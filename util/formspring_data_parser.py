import functools
import itertools
import logging
import uuid

import pandas as pd
import xmltodict
from typing import List, Dict

from phonemes_from_graphemes import PhonemesFromGraphemes
from words_2_vectors import ModelWrapper
from util.Util import clean_text, smart_split_in_words


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
        return functools.reduce(
            lambda df1, df2: pd.concat([df1, df2]),
            [self.questions_answers_labels(an_id) for an_id in range(self.how_many_entries())])