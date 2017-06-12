import logging
from subprocess import check_output
from timeit import default_timer as timer
import functools
import shelve
import gensim
import bisect
import numpy as np
from typing import List, Dict
import pandas as pd
import uuid
import xmltodict
import itertools
from util.Util import clean_text, smart_split_in_words


class SoundsDict:
    """
        Handles a set of 'shelves' as part of a unified dictionary
    """

    def __file_name_2_shelf_or_None__(self, shelf_filename: str):
        try:
            return shelve.open(shelf_filename, flag='r')
        except Exception as e:
            self.alogger.error("Impossible to open '{}': {}".format(shelf_filename, str(e)))
            return None

    def __init__(self, file_names, alogger: logging.Logger):
        self.alogger = alogger
        self.all_shelves = []
        self.all_shelves = list(map(self.__file_name_2_shelf_or_None__, file_names))
        self.all_shelves = [ a_shelf for a_shelf in self.all_shelves if a_shelf is not None ]

    def __getitem__(self, key):
        """
        Index operator 
        Args:
            key: 

        Returns: the value 
        """
        def get_from_dict_in_head(set_of_dict):
            if len(set_of_dict) == 0:
                raise KeyError("key '{}' is not in this set".format(key))
            else:
                v_in_head = set_of_dict[0].get(key)
                if v_in_head is not None:
                    return v_in_head
                else:
                    get_from_dict_in_head(set_of_dict[1:])
        return get_from_dict_in_head(self.all_shelves)

    def __del__(self):
        """
        On 'destruction', close all references to shelves 
        Returns:

        """
        self.alogger.debug("Closing all open shelves")
        for a_shelf in self.all_shelves:
            a_shelf.close()
        self.alogger.debug("Closing all open shelves => DONE")

class PhonemesFromGraphemes:

    MAX_LENGTH_TO_SPEAK = 10  # if I give more than this, espeak fails to do a good job

    def __init__(self, alogger: logging.Logger):
        self.alogger = alogger

    def set_log_level(self, log_level):
        """
            log_level: one of logging.{WARNING, ...} 

        """
        self.alogger.setLevel(log_level)

    def graphs2phones(self, s):
        """
            Graphemes to Phonemes: 
            Takes a sentence, returns an array of graphemes strings (one per number of words in original sentence)
            Example(s): 
            > graphs2phones('hello world bla and ble')
            > graphs2phones(' wasuuuppp!')
        """
        phs = check_output(["/usr/local/bin/speak", "-q", "-x" ,'-v', 'en-us' ,s]).decode('utf-8')
        return [w for w in phs.strip().split(" ") if w != ' ']

    def take_time(self, code_snippet_as_string):
        """
            Measures the time it takes to execute the code snippet
            provided as string. 
            Returns: the value of the invocation + number of seconds it took. 
            Example(s): 
            > r, secs = take_time("2 + 2")
            > print("result = {}, time = {} secs".format(r, secs))
        """
        start = timer()
        r = eval(code_snippet_as_string)
        end = timer()
        return (r, end - start)

    def graphemes_to_phonemes(self, words_in_sent):
        """
            Takes a list of words and returns a list of tuples
            (grapheme: phoneme)
            Example:
            > graphemes_to_phonemes(["luis", "papa"])
            [('luis', "lj'u:Iz"), ('papa', "pa#p'A:")]
        """
        # First step: generate all sounds of words as if they were "alone" (ie, not in a sentence)
        # We want to avoid a combination of words making only 1 sound
        # For example (depending on accent): "what's up?"
        # So in order to do that we'll introduce a word with a unique sound between the words,
        # generate phonemes and then process them smartly:
        # separator for words in sentence
        separator = {"str": "XXX"}
        separator["sound"] = ''.join(self.graphs2phones(separator["str"]))
        #
        how_many_words = len(words_in_sent)
        num_batches = (how_many_words // self.MAX_LENGTH_TO_SPEAK) + int(how_many_words % self.MAX_LENGTH_TO_SPEAK != 0)
        result_array = [] # {}
        for i in range(num_batches):
            words_in_batch = words_in_sent[i * self.MAX_LENGTH_TO_SPEAK: (i + 1 ) *self.MAX_LENGTH_TO_SPEAK]
            sent_augm = ' '.join \
                ([w1 + ' ' + w2 for w1, w2 in list(zip([separator["str"] ] *len(words_in_batch), words_in_batch))]) + " " + separator["str"]
            phonemes_strs_augm = self.graphs2phones(sent_augm)
            # there we go: all (indexes of) sounds that we are interested in.
            seps_idxs = [i for i ,v in enumerate(phonemes_strs_augm) if v.endswith(separator["sound"]) or v.startswith(separator["sound"]) ]
            how_many_separators = len(seps_idxs)

            all_sounds = list(map(
                lambda t: ' '.join(phonemes_strs_augm[t[0] + 1: t[1]]),
                list(zip(seps_idxs[:-1], seps_idxs[1:]))))
            result_array += list(zip(words_in_batch, all_sounds))
        return result_array


    def dict_graphemes_to_phonemes(self, words_in_sent) -> dict:
        as_phon_graph_list = self.graphemes_to_phonemes(words_in_sent)
        return {ph: graph for (graph, ph) in as_phon_graph_list}


    def graphemes_to_phonemes_to_shelves(self, words_in_sent, shelf_filename):
        """
            Takes a list of words and returns a list of tuples
            (grapheme: phoneme)
            Example:
            > graphemes_to_phonemes(["luis", "papa"])
            [('luis', "lj'u:Iz"), ('papa', "pa#p'A:")]
        """
        # let's do this in batches:
        how_many_words = len(words_in_sent)
        num_batches = (how_many_words // self.MAX_LENGTH_TO_SPEAK) + int(how_many_words % self.MAX_LENGTH_TO_SPEAK != 0)
        result_dict = shelve.open(shelf_filename, flag='c')

        try:
            for i in range(num_batches):
                batch_begin = i * self.MAX_LENGTH_TO_SPEAK
                batch_end = batch_begin + self.MAX_LENGTH_TO_SPEAK
                words_in_batch = words_in_sent[batch_begin: batch_end]
                result_for_batch = self.graphemes_to_phonemes(words_in_batch)
                if i % 1000 == 0:
                    self.alogger.info("batch {} out of {}: words_in_sent[{}:{}] => {}".format(i + 1, num_batches, batch_begin, batch_end, result_for_batch))
                #
                for word, sound in result_for_batch:
                    existing_set = result_dict.get(sound)
                    result_dict[sound] = (set(existing_set) if existing_set is not None else set()).union({ word })
                result_dict.sync()
        finally:
            self.alogger.info("Closing shelf '{}'".format(shelf_filename))
            result_dict.close()

class ModelWrapper():
    default_shelf_filename = 'shelf_from0_for2999999.shelf'

    def __init__(self, data_dir: str, alogger: logging.Logger, m = None, sounds_dict: SoundsDict = None):
        self.data_dir = data_dir
        self.alogger = alogger

        if m is None:
            f_name = '{}/GoogleNews-vectors-negative300.bin.gz'.format(self.data_dir)
            self.alogger.info("Loading model from {}...".format(f_name))
            self.model = gensim.models.word2vec.KeyedVectors.load_word2vec_format(f_name, binary=True)
            self.alogger.info("Model succesfully loaded")
        else:
            self.alogger.info(
                "[init] Model provided. If you want me to FORCE re-load it, call ModelWrapper's constructor with 'None'")
            self.model = m
        #
        self.sounds_dict = sounds_dict
        # sort all the words in the model, so that we can auto-complete queries quickly
        self.alogger.info("Sort all the words in the model, so that we can auto-complete queries quickly...")
        self.orig_words = [gensim.utils.to_unicode(word) for word in self.model.index2word]
        indices = [i for i, _ in sorted(enumerate(self.orig_words), key=lambda item: item[1].lower())]
        self.all_words = [self.orig_words[i].lower() for i in indices]  # lowercased, sorted as lowercased
        self.orig_words = [self.orig_words[i] for i in indices]  # original letter casing, but sorted as if lowercased



    def suggest(self, term):
        """
        For a given prefix, return 10 words that exist in the model start start with that prefix
        """
        prefix = gensim.utils.to_unicode(term).strip().lower()
        count = 10
        pos = bisect.bisect_left(self.all_words, prefix)
        result = self.orig_words[pos: pos + count]
        return result

    def most_similar(self, positive, negative):
        """
            positive: an array of positive words
            negative: an array of negative words 
        """
        try:
            result = self.model.most_similar(
                positive=[word.strip() for word in positive if word],
                negative=[word.strip() for word in negative if word],
                topn=5)
        except:
            result = []
        return {'similars': result}

    def vec_repr(self, word):
        """
            If 'word' belongs in the vocabulary, returns its 
            word2vec representation. Otherwise returns a vector of 0's
            of the same length of the other words. 
        """
        try:
            return self.model.word_vec(word)
        except KeyError:
            self.alogger.debug("'{}' not in Model. Returning [0]'s vector.".format(word))
            return np.zeros(self.model.vector_size)

    def set_sounds_dict(self, sounds_dict: SoundsDict):
        self.sounds_dict = sounds_dict

    def sound_to_word(self, a_sound: str) -> List[str]:
        """
        Does a mpa sound -> word
        :param a_sound: 
        :return: 
        """
        if self.sounds_dict is None:
            self.alogger.error('Sounds dictionary not set')
            raise RuntimeError('Sounds dictionary not set')
        return self.sounds_dict[a_sound]

    def safe_sound_to_word(self, ph: str) -> List[str]:
        """
            Returns words that have the sound passed as parameter; 
            if there is no such map, returns the empty list 
        :param ph: 
        :return: 
        """
        if ph is None or len(ph) == 0:
            return []
        try:
            r = self.sound_to_word(ph)
            if r is None:
                return []
            else:
                return r
        except:
            return []

    def sound_to_vec(self, a_sound: str) -> str:
        return self.vec_repr(self.sound_to_word(a_sound))

    def sound_repr(self, a_sound: str) -> Dict:
        return {'word': self.sound_to_word(a_sound), 'vec': self.sound_to_vec(a_sound)}


class Formspring_Data_Parser():
    def __init__(self, path_to_xml, pg: PhonemesFromGraphemes, mw: ModelWrapper, alogger: logging.Logger):
        self.path = None
        self.doc = None
        with open(path_to_xml) as fd:
            self.path = path_to_xml
            self.doc = xmltodict.parse(fd.read())
        self.alogger = alogger
        self.pg = pg
        self.mw = mw

    def how_many_entries(self) -> int: return len(self.doc['dataset']['FORMSPRINGID'])

    def all_entries(self) -> List: return self.doc['dataset']['FORMSPRINGID']

    def posts_for_id(self, an_id: int) -> List:
        r = self.doc['dataset']['FORMSPRINGID'][an_id]['POST']
        # is this a list or a dict?
        try:
            r[0]  # wil this throw an Exception?
            return r
        except:
            return list([r])

    def doc2dict(self, an_id: int) -> List[Dict]:
        """
            Parses the information on an XML record and converts it into a dictionary entry 
            with fields 
            * uuid
            * txt_orig
            * question_raw
            * answer_raw
            * labels 
        :param an_id: id of a participant in the data 
        :return: a list of dictionaries 
        """
        some_posts = self.posts_for_id(an_id)
        r = []
        for post in some_posts:
            q_and_a_orig = post['TEXT']
            q_and_a = clean_text(post['TEXT'])
            # parse question
            beg_q = q_and_a.index("Q:") + 2
            end_q = q_and_a.index("A:")
            the_q_raw = q_and_a[beg_q:end_q].strip()
            # parse answer
            the_a_raw = q_and_a[end_q + 2:].strip()
            raw_labels = [lab['ANSWER'] for lab in post['LABELDATA']]
            labels = list(map(lambda txt: txt.lower(), [lab for lab in raw_labels if lab is not None]))
            r.append({
                "uuid": uuid.uuid4(),
                "txt_orig": q_and_a_orig,
                "question_raw": the_q_raw,
                "answer_raw": the_a_raw,
                "labels": labels,
            })
        return r

    def __raw2cleans__(self, raw: str) -> List[str]:
        graphs_and_phons = self.pg.graphemes_to_phonemes(words_in_sent=smart_split_in_words(raw))
        graphs_and_phons_ext = [(graph, ph, list(set(map(lambda w: w.lower(), self.mw.safe_sound_to_word(ph))))) for graph, ph
                                in graphs_and_phons]
        graphs_and_phons_ext_winners = [(graph, ph, cands, [graph] if graph.lower() in cands else cands) for graph, ph, cands in
                                        graphs_and_phons_ext]
        sounds_to_words = [winners for _, _, _, winners in graphs_and_phons_ext_winners if len(winners) > 0]
        alls_as_tuples = [elt for elt in itertools.product(*sounds_to_words)]
        alls_as_strings = [' '.join(a_tuple) for a_tuple in alls_as_tuples]
        return alls_as_strings

    def questions_answers_labels(self, an_id: int) -> pd.DataFrame: #  List[Dict]:
        raw_posts = self.doc2dict(an_id)
        r = []
        for post in raw_posts:
            # read pre-calculated fields
            the_uuid = post['uuid']
            q_and_a_orig = post['txt_orig']
            the_q_raw = post['question_raw']
            the_a_raw = post['answer_raw']
            labels = post['labels']
            # parse question
            questions_from_raw = self.__raw2cleans__(the_q_raw)
            list_of_questions = questions_from_raw
            # parse answer
            answers_from_raw = self.__raw2cleans__(the_a_raw)
            list_of_answers = answers_from_raw
            # parse votes
            all_votes_as_yes = [l for l in labels if l == 'yes']
            is_threat = (len(all_votes_as_yes) / len(labels)) >= 0.5
            #
            q_and_a_s = [list_of_questions, list_of_answers]
            for a_q, an_a in [elt for elt in itertools.product(*q_and_a_s)]:
                r.append({
                    "uuid": the_uuid,
                    "question": a_q,
                    "question_raw": the_q_raw,
                    # "question_as_sounds": the_q_as_sounds,
                    "answer": an_a,
                    "answer_raw": the_a_raw,
                    # "answer_as_sounds": the_a_raw,
                    "threat": is_threat
                })
        self.alogger.debug("Generated data for entry = {}".format(an_id))
        return pd.DataFrame(r)
        # return r

    def all_questions_answers_labels(self) -> pd.DataFrame:
        return functools.reduce(lambda df1, df2: pd.concat([df1, df2]), [self.questions_answers_labels(an_id) for an_id in range(self.how_many_entries())])
