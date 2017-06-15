import functools
import itertools
import logging
import uuid

import pandas as pd
import xmltodict
from typing import List, Dict

from phonemes_from_graphemes import PhonemesFromGraphemes
from util.Util import clean_text
from util.parser.data_parser import *
from words_2_vectors import ModelWrapper

def how_many_entries(a_doc) -> int: return len(a_doc['dataset']['FORMSPRINGID'])

def all_entries(a_doc) -> List: return a_doc['dataset']['FORMSPRINGID']

def all_questions_answers_labels(pg, mw, a_doc, how_many_entries: int, alogger: logging.Logger) -> pd.DataFrame:
    def questions_answers_labels(an_id: int) -> pd.DataFrame:  # List[Dict]:
        def posts_for_id(an_id: int) -> List:
            r = a_doc['dataset']['FORMSPRINGID'][an_id]['POST']
            # is this a list or a dict?
            try:
                r[0]  # wil this throw an Exception?
                return r
            except:
                return list([r])

        def doc2dict(an_id: int) -> List[Dict]:
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
            some_posts = posts_for_id(an_id)
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

        raw_posts = doc2dict(an_id)
        r = []
        for post in raw_posts:
            # read pre-calculated fields
            the_uuid = post['uuid']
            q_and_a_orig = post['txt_orig']
            the_q_raw = post['question_raw']
            the_a_raw = post['answer_raw']
            labels = post['labels']
            # parse question
            questions_from_raw = clean_raw_string(pg, mw, the_q_raw)
            list_of_questions = questions_from_raw
            # parse answer
            answers_from_raw = clean_raw_string(pg, mw, the_a_raw)
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
        alogger.debug("Generated data for entry = {}".format(an_id))
        return pd.DataFrame(r)
        # return r

    print([an_id for an_id in range(how_many_entries)])
    return functools.reduce(
        lambda df1, df2: pd.concat([df1, df2]),
        [questions_answers_labels(an_id) for an_id in range(how_many_entries)])


class Formspring_Data_Parser(Data_Parser):
    def __init__(self, all_data: pd.DataFrame, alogger: logging.Logger):
        Data_Parser.__init__(self)
        assert {'answer', 'question', 'threat'}.issubset(all_data.columns)
        self.really_all = all_data
        self.alogger = alogger

    def save(self, file_name: str):
        self.really_all.to_csv(file_name)
        self.alogger.info("Saved to '{}'".format(file_name))

    @classmethod
    def from_raw_xml(cls, path_to_xml, pg: PhonemesFromGraphemes, mw: ModelWrapper, alogger: logging.Logger):
        alogger.debug("Raw parsing '{}'...".format(path_to_xml))
        with open(path_to_xml) as fd:
            the_doc = xmltodict.parse(fd.read())
        alogger.debug("Launching construction of dataframe")
        really_all = all_questions_answers_labels(pg, mw, the_doc, how_many_entries=how_many_entries(the_doc), alogger=alogger)
        return cls(all_data=really_all, alogger=alogger)

    def interactions(self):
        return ["{}; {}".format(q, a) for q, a in list(zip(self.really_all['question'].tolist(), self.really_all['answer'].tolist()))]

    def labels(self):
        return ["THREAT" if threat else "CLEAN" for threat in self.really_all['threat'].tolist()]