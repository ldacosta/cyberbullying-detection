from util.parser.data_parser import Data_Parser, clean_raw_string
import logging
import pandas as pd
import os.path
from util import Util
from tqdm import tqdm
from words_2_vectors import ModelWrapper
from phonemes_from_graphemes import PhonemesFromGraphemes, SoundsDict

class Wikipedia_Talk_Parser(Data_Parser):
    """
    Inspired by https://figshare.com/articles/Wikipedia_Talk_Labels_Personal_Attacks/4054689 
    """


    def __init__(self, comments: pd.DataFrame, alogger: logging.Logger):
        Data_Parser.__init__(self)
        assert {'comment', 'attack'}.issubset(comments.columns)
        self.comments = comments
        self.alogger = alogger

    def save(self, file_name: str):
        self.comments.to_csv(file_name)
        self.alogger.info("Saved to '{}'".format(file_name))

    @classmethod
    def from_sources(cls, pg, mw, num_words: int, alogger: logging.Logger):
        """
        
        :param pg: 
        :param mw: 
        :param num_words: how many words should we keep for each comment 
        :param alogger: 
        :return: 
        """

        comments_file_name = 'attack_annotated_comments.tsv'
        annotations_file_name = 'attack_annotations.tsv'
        ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
        ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'

        if not os.path.isfile(comments_file_name):
            Util.download_file(ANNOTATED_COMMENTS_URL, comments_file_name)
        alogger.info("'{}' is downloaded as '{}'".format(ANNOTATED_COMMENTS_URL, comments_file_name))
        #
        if not os.path.isfile(annotations_file_name):
            Util.download_file(ANNOTATIONS_URL, annotations_file_name)
        alogger.info("'{}' is downloaded as '{}'".format(ANNOTATIONS_URL, annotations_file_name))
        #
        comments = pd.read_csv(comments_file_name, sep = '\t', index_col = 0)
        annotations = pd.read_csv(annotations_file_name,  sep = '\t')
        alogger.debug("Read comments and annotations")
        # labels a comment as an atack if the majority of annotators did so
        labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
        # join labels and comments
        comments['attack'] = labels
        alogger.debug("Labels properly set")
        # remove newline and tab tokens
        comments['comment'] = comments['comment'].\
            apply(lambda x: x.replace("NEWLINE_TOKEN", " ").replace("TAB_TOKEN", " ").replace("`", "'"))
        # comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
        # comments['comment'] = comments['comment'].apply(lambda x: x.replace("`", "'"))
        comments['comment'] = comments['comment'].apply(lambda x: x[1:-1] if (x[0] == x[-1]) and (x[0] == "'") else x)
        comments['comment'] = comments['comment'].apply(lambda x: x.strip())
        comments = comments[['comment', 'attack']] # .head(n = 100)
        alogger.debug("Starting cleaning up all {} data elements by phonetics (keeping max of {} words per text...".format(comments.shape[0], num_words))
        comments['comment'] = comments['comment'].apply(lambda x: clean_raw_string(pg, mw, raw = ' '.join(x.split()[:num_words])))
        alogger.debug("Re-building dataframe now...")
        comments = pd.concat([pd.Series(row['attack'], row['comment'])
                                    for _, row in tqdm(comments.iterrows(),total=comments.shape[0])]).reset_index()
        comments.columns = ['comment', 'attack']

        alogger.info(
            "Read all comments; there are {} out of {} that are considered 'attack'".
                format(comments.loc[comments['attack'] == True].shape[0], comments.shape[0]))
        return cls(comments, alogger)

    def interactions(self):
        return [c for c in self.comments['comment'].tolist()]

    def labels(self):
        return ["THREAT" if threat else "CLEAN" for threat in self.comments['attack'].tolist()]


def build_wiki_parser_and_load_everyone(
    read_from_here: str,
    mw: ModelWrapper,
    sounds_dict: SoundsDict,
    phonemesFactory: PhonemesFromGraphemes,
    data_dir: str,
    common_logger: logging.Logger):
    """
    Returns the parser + everyhting we needed to build it: 
    * the Model Wrapper
    * the Sounds Dictionary 
    * the phonemes_from_graphemes class
    """
    assert (read_from_here is not None) or (data_dir is not None)
    if read_from_here is not None:
        all_wikip = pd.read_csv('dataframe_wiki.csv')
        wiki_parser = Wikipedia_Talk_Parser(comments = all_wikip, alogger = common_logger)
    else:
        # model wrapper
        if mw is None:
            mw = ModelWrapper.from_google_news_model(data_dir=data_dir, alogger=common_logger)
        # sounds dictionary
        if sounds_dict is None:
            sounds_dict = SoundsDict(a_dir=data_dir, alogger=common_logger)
        mw.set_sounds_dict(sounds_dict=sounds_dict)
        # phonemes factory
        if phonemesFactory is None:
            phonemesFactory = PhonemesFromGraphemes(alogger=common_logger)
        # parser
        wiki_parser = Wikipedia_Talk_Parser.from_sources(pg = phonemesFactory, mw = mw, alogger=common_logger)
    #
    return (wiki_parser, mw, sounds_dict, phonemesFactory)


def wikinastigrams_load_or_create(
        wiki_parser_load: bool,
        common_logger: logging.Logger,
        mw,
        sounds_dict,
        phonemesFactory,
        data_dir) -> (Wikipedia_Talk_Parser, ModelWrapper, SoundsDict, PhonemesFromGraphemes):
    print("Please monitor the debug file (ie, run 'tail -f {}')".format(common_logger.handlers[1].baseFilename))
    if wiki_parser_load:
        (wiki_data_parser, mw, sounds_dict, phonemesFactory) = build_wiki_parser_and_load_everyone(
            read_from_here = "dataframe_wiki.csv",
            mw = mw,
            sounds_dict = sounds_dict,
            phonemesFactory = phonemesFactory,
            data_dir = data_dir,
            common_logger = common_logger)
    else:
        (wiki_data_parser, mw, sounds_dict, phonemesFactory) = build_wiki_parser_and_load_everyone(
            read_from_here = None,
            mw = mw,
            sounds_dict = sounds_dict,
            phonemesFactory = phonemesFactory,
            data_dir = data_dir,
            common_logger = common_logger)
    common_logger.info("Wiki parser should have been loaded")
    return (wiki_data_parser, mw, sounds_dict, phonemesFactory)

import util.parser.formspring_data_parser as formspring_data_parser
import util.Data as util_data

if __name__ == "__main__":
    # from util.Util import *
    # from phonemes_from_graphemes import *
    # from words_2_vectors import *
    # import os
    #
    # WORDS_FOR_TEXT = 25 # how many words to take on each text to do the classification
    #
    # #
    # # let's go to the appropriate directory
    # root_dir = get_git_root()
    # os.chdir(root_dir)
    # data_dir = "./data"
    # common_logger = get_logger(name="common_logger", debug_log_file_name="common_logger.log")
    # common_logger.info("Switched to directory '{}'".format(root_dir))
    # common_logger.info("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))
    # mw = ModelWrapper.from_google_news_model(data_dir=data_dir, alogger=common_logger)
    # sounds_dict = SoundsDict(a_dir=data_dir, alogger=common_logger)
    # mw.set_sounds_dict(sounds_dict=sounds_dict)
    # phonemesFactory = PhonemesFromGraphemes(alogger=common_logger)
    # wiki_parser = Wikipedia_Talk_Parser.from_sources(pg=phonemesFactory, mw=mw, num_words=WORDS_FOR_TEXT, alogger=common_logger)
    # wiki_parser.save("dataframe_wiki.csv")


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
    wiki_parser_load = True
    existing_components = False
    print("Please monitor the debug file (ie, run 'tail -f {}')".format(common_logger.handlers[1].baseFilename))

    (wiki_data_parser, mw, sounds_dict, phonemesFactory) = wikinastigrams_load_or_create(
        wiki_parser_load,
        common_logger,
        mw = None,
        sounds_dict = None,
        phonemesFactory = None,
        data_dir = data_dir)

    interactions_wiki = wiki_data_parser.interactions()
    labels_wiki = wiki_data_parser.labels()
    print("There are {} data points".format(len(interactions_wiki)))

    mw, _ = formspring_data_parser.properly_create_model_wrapper(mw, sounds_dict, common_logger)
    nlp = spacy.load('en')
    sents_encoder = util_data.reviews_labels_encoder(
        mw,
        n_words_in_review = 10,
        reviews = list(map(lambda x: util_data.review(x), interactions_wiki)),
        labels = list(map(lambda x: util_data.label(x), labels_wiki)),
        spacy_nlp = nlp)
    formspring_data_parser.run_neural_networks(sents_encoder, interactions_wiki, labels_wiki)




