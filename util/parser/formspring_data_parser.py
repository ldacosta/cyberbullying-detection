import functools
import itertools
import logging
import uuid

import pandas as pd
import xmltodict
from typing import List, Dict

from phonemes_from_graphemes import PhonemesFromGraphemes, SoundsDict
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

def properly_create_model_wrapper(mw: ModelWrapper, sounds_dict: SoundsDict, common_logger: logging.Logger) -> (ModelWrapper, SoundsDict):
    if mw is None:
        mw = ModelWrapper.from_google_news_model(data_dir=data_dir, alogger=common_logger)
    # sounds dictionary
    if sounds_dict is None:
        sounds_dict = SoundsDict(a_dir=data_dir, alogger=common_logger)
    mw.set_sounds_dict(sounds_dict=sounds_dict)
    return mw, sounds_dict

def build_formspring_parser_and_load_everyone(
    read_from_here: str,
    mw: ModelWrapper,
    sounds_dict: SoundsDict,
    phonemesFactory: PhonemesFromGraphemes,
    data_dir: str,
    common_logger: logging.Logger) -> (Formspring_Data_Parser, ModelWrapper, SoundsDict, PhonemesFromGraphemes):
    """
    Returns the parser + everyhting we needed to build it: 
    * the Model Wrapper
    * the Sounds Dictionary 
    * the phonemes_from_graphemes class
    """
    assert (read_from_here is not None) or (data_dir is not None)
    if read_from_here is not None:
        a_formspring_data_parser = Formspring_Data_Parser(all_data = pd.read_csv(read_from_here), alogger = common_logger)
    else:
        # model wrapper and sounds dictionary
        mw, sounds_dict = properly_create_model_wrapper(mw, sounds_dict, common_logger)
        # phonemes factory
        if phonemesFactory is None:
            phonemesFactory = PhonemesFromGraphemes(alogger=common_logger)
        # parser
        xml_file_name = '/Users/luisd/Downloads/FormspringLabeledForCyberbullying/XMLMergedFile.xml'
        a_formspring_data_parser = Formspring_Data_Parser.from_raw_xml(xml_file_name, pg = phonemesFactory, mw = mw, alogger = common_logger)
    #
    return (a_formspring_data_parser, mw, sounds_dict, phonemesFactory)

def formspring_load_or_create(
        formspring_parser_load: bool,
        common_logger: logging.Logger,
        mw,
        sounds_dict,
        phonemesFactory,
        data_dir) -> (Formspring_Data_Parser, ModelWrapper, SoundsDict, PhonemesFromGraphemes):
    if formspring_parser_load:
        print("Please monitor the debug file (ie, run 'tail -f {}')".format(common_logger.handlers[1].baseFilename))
        (a_formspring_data_parser, mw, sounds_dict, phonemesFactory) = build_formspring_parser_and_load_everyone(
            read_from_here = "really_all_of_them.csv",
            mw = mw,
            sounds_dict = sounds_dict,
            phonemesFactory = phonemesFactory,
            data_dir = data_dir,
            common_logger = common_logger)
    else:
        (a_formspring_data_parser, mw, sounds_dict, phonemesFactory) = build_formspring_parser_and_load_everyone(
            read_from_here = None,
            mw = mw,
            sounds_dict = sounds_dict,
            phonemesFactory = phonemesFactory,
            data_dir = data_dir,
            common_logger = common_logger)
    common_logger.info("Formspring parser should have been loaded")
    return (a_formspring_data_parser, mw, sounds_dict, phonemesFactory)

from sklearn import model_selection
import util.Data as util_data


def run_neural_networks(sents_encoder: util_data.reviews_labels_encoder, questions_answers_formspring, labels_formspring):
    # fully connected
    # TODO: the datasets are very imbalanced - so we must do some over(/under) sampling. Maybe https://github.com/scikit-learn-contrib/imbalanced-learn ?
    X_train, X_test, y_train, y_test = model_selection.train_test_split(questions_answers_formspring, labels_formspring,
                                                                        test_size=0.33, random_state=42)
    cyber_fc = fc_nn.CyberbullyingFullyConnectedNetwork(reviews=questions_answers_formspring, labels=labels_formspring,hidden_nodes=300,learning_rate=0.01)
    cyber_fc.train(X_train, y_train)
    cyber_fc.test(X_test, y_test)

    # P-CNN
    cnn_k = nn.CyberbullyingDetectionnNN(features_in_words=300, words_in_review=10)
    # nn.sanity_check()
    reviews_as_matrix = sents_encoder.reviews_as_matrix()
    labels_as_matrix = sents_encoder.labels_as_matrix()

    # Split the data
    # TODO: the datasets are very imbalanced - so we must do some over(/under) sampling. Maybe https://github.com/scikit-learn-contrib/imbalanced-learn ?
    X_train, X_test, y_train, y_test = model_selection.train_test_split(reviews_as_matrix, labels_as_matrix,
                                                                        test_size=0.33, random_state=42)
    an_x_subset = X_train[:500]
    a_y_subset = y_train[:500]
    cnn_k.fit(x_train=an_x_subset.reshape(an_x_subset.shape + (1,)), y_train=a_y_subset, batch_size=78, epochs=20)
    # cnn_k.fit(x_train=X_train.reshape(X_train.shape + (1,)), y_train=y_train, batch_size=78, epochs=20)

    print(cnn_k.evaluate(x = X_test.reshape(X_test.shape + (1,)), y = y_test, batch_size=80))


if __name__ == "__main__":
    from util.Util import *
    from phonemes_from_graphemes import *
    from words_2_vectors import *
    import os
    import nn
    import fc_nn
    import spacy  # nl processing

    WORDS_FOR_TEXT = 25 # how many words to take on each text to do the classification

    #
    # let's go to the appropriate directory
    root_dir = get_git_root()
    os.chdir(root_dir)
    data_dir = "./data"
    common_logger = get_logger(name="common_logger", debug_log_file_name="common_logger.log")
    common_logger.info("Switched to directory '{}'".format(root_dir))
    common_logger.info("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))
    #
    #
    formspring_parser_load = True
    existing_components = False
    print("Please monitor the debug file (ie, run 'tail -f {}')".format(common_logger.handlers[1].baseFilename))
    (a_formspring_data_parser, mw, sounds_dict, phonemesFactory) = formspring_load_or_create(
        formspring_parser_load,
        common_logger,
        mw = None,
        sounds_dict = None,
        phonemesFactory = None,
        data_dir = data_dir)

    questions_answers_formspring = a_formspring_data_parser.interactions()
    labels_formspring = a_formspring_data_parser.labels()


    mw, _ = properly_create_model_wrapper(mw, sounds_dict, common_logger)
    nlp = spacy.load('en')
    sents_encoder = util_data.reviews_labels_encoder(
        mw,
        n_words_in_review = 10,
        reviews = list(map(lambda x: util_data.review(x), questions_answers_formspring)),
        labels = list(map(lambda x: util_data.label(x), labels_formspring)),
        spacy_nlp = nlp)
    run_neural_networks(sents_encoder, questions_answers_formspring, labels_formspring)
