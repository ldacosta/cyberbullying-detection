import logging
from subprocess import check_output
from timeit import default_timer as timer
import functools
import shelve
import gensim
import bisect
import numpy as np
from typing import List, Dict, Set



def get_logger(name: str):
    alogger = logging.getLogger(name)
    if not len(alogger.handlers):
        alogger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        alogger.addHandler(ch)
        print("Logger created")
    else:
        print("Logger retrieved")
    return alogger

class SoundsDict:
    """
        Handles a set of 'shelves' as part of a unified dictionary
    """

    def __file_name_2_shelf_or_None__(self, shelf_filename: str):
        try:
            return shelve.open(shelf_filename, flag='r')
        except Exception as e:
            self.alogger.debug("Impossible to open '{}': {}".format(shelf_filename, str(e)))
            return None

    def __init__(self, file_names, logger_name = None):
        if logger_name is None:
            self.alogger = get_logger(name = __name__)
        else:
            self.alogger = get_logger(name = logger_name)
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

    def __init__(self, logger_name = None):
        if logger_name is None:
            self.alogger = get_logger(name = __name__)
        else:
            self.alogger = get_logger(name = logger_name)

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
        self.alogger.debug("Return {} strings: {}".format(len(phs.split()), phs))
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
            self.alogger.debug("{}: {} to {}".format(i, i * self.MAX_LENGTH_TO_SPEAK, (i + 1 ) *self.MAX_LENGTH_TO_SPEAK))
            words_in_batch = words_in_sent[i * self.MAX_LENGTH_TO_SPEAK: (i + 1 ) *self.MAX_LENGTH_TO_SPEAK]
            self.alogger.debug("words_in_batch = {}".format(words_in_batch))
            sent_augm = ' '.join \
                ([w1 + ' ' + w2 for w1, w2 in list(zip([separator["str"] ] *len(words_in_batch), words_in_batch))]) + " " + separator["str"]
            self.alogger.debug("sent_augm = {}".format(sent_augm))
            phonemes_strs_augm = self.graphs2phones(sent_augm)
            self.alogger.debug("phonemes_strs_augm = {}".format(phonemes_strs_augm))
            # there we go: all (indexes of) sounds that we are interested in.
            seps_idxs = [i for i ,v in enumerate(phonemes_strs_augm) if v.endswith(separator["sound"])]
            self.alogger.debug("seps_idxs = {}".format(seps_idxs))
            how_many_separators = len(seps_idxs)
            self.alogger.debug("how_many_separators = {}".format(how_many_separators))

            all_sounds = list(map(
                lambda t: ' '.join(phonemes_strs_augm[t[0] + 1: t[1]]),
                list(zip(seps_idxs[:-1], seps_idxs[1:]))))
            self.alogger.debug("all sounds = {}".format(all_sounds))
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
                    self.alogger.debug("After inserting word '{}' => '{}' :: {}".format(word, sound, result_dict[sound]))
                result_dict.sync()
        finally:
            self.alogger.info("Closing shelf '{}'".format(shelf_filename))
            result_dict.close()



class ModelWrapper():
    default_shelf_filename = 'shelf_from0_for2999999.shelf'

    def __init__(self, data_dir: str, m = None, logger_name: str = None):
        self.data_dir = data_dir
        if logger_name is None:
            self.alogger = get_logger(name = __name__)
        else:
            self.alogger = get_logger(name = logger_name)

        if m is None:
            f_name = '{}/GoogleNews-vectors-negative300.bin.gz'.format(self.data_dir)
            self.alogger.info("Loading model from {}...".format(f_name))
            self.model = gensim.models.word2vec.KeyedVectors.load_word2vec_format(f_name, binary=True)
            self.alogger.info("Model succesfully loaded")
        else:
            self.alogger.info(
                "[init] Model provided. If you want me to FORCE re-load it, call ModelWrapper's constructor with 'None'")
            self.model = m
        # sort all the words in the model, so that we can auto-complete queries quickly
        self.alogger.info("Sort all the words in the model, so that we can auto-complete queries quickly...")
        self.orig_words = [gensim.utils.to_unicode(word) for word in self.model.index2word]
        indices = [i for i, _ in sorted(enumerate(self.orig_words), key=lambda item: item[1].lower())]
        self.all_words = [self.orig_words[i].lower() for i in indices]  # lowercased, sorted as lowercased
        self.orig_words = [self.orig_words[i] for i in indices]  # original letter casing, but sorted as if lowercased


    def set_sounds_dictionary(self, sounds_dict) -> bool:
        if sounds_dict is not None:
            self.sounds_dict = sounds_dict
            return True
        else:
            self.alogger.error("Sounds dictionary is 'None'. Setting ignored")
            return False

    def get_sounds_dictionary_from(self, file_name) -> bool:
        try:
            self.alogger.info("Loading default sounds dictionary from '{}'...".format(file_name))
            self.sounds_dict = shelve.open(file_name, flag='r')
            self.alogger.info("Sounds dictionary succesfully loaded")
            return True
        except Exception as e:
            self.alogger.error("Impossible to load sounds dictionary from {}: {}".format(file_name, str(e)))
            return False




    def suggest(self, term):
        """
        For a given prefix, return 10 words that exist in the model start start with that prefix
        """
        prefix = gensim.utils.to_unicode(term).strip().lower()
        count = 10
        pos = bisect.bisect_left(self.all_words, prefix)
        result = self.orig_words[pos: pos + count]
        self.alogger.debug("suggested %r: %s" % (prefix, result))
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
        self.alogger.debug("similars for %s vs. %s: %s" % (positive, negative, result))
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

    def sound_to_word(self, a_sound: str) -> str:
        return self.sounds_dict[a_sound]

    def sound_to_vec(self, a_sound: str) -> str:
        return self.vec_repr(self.sound_to_word(a_sound))

    def sound_repr(self, a_sound: str) -> Dict:
        return {'word': self.sound_to_word(a_sound), 'vec': self.sound_to_vec(a_sound)}

