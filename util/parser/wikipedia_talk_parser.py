from util.parser.data_parser import Data_Parser, clean_raw_string
import logging
import pandas as pd
import os.path
from util import Util
from tqdm import tqdm

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
    def from_sources(cls, pg, mw, alogger: logging.Logger):
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
        comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
        alogger.debug("Starting cleaning up data by phonetics...")
        final_comments = comments.copy()
        del comments # let's free some memory
        final_comments['comment'] = final_comments['comment'].apply(lambda x: clean_raw_string(pg, mw, raw = x))
        alogger.debug("Re-building dataframe now...")
        final_comments = pd.concat([pd.Series(row['attack'], row['comment'])
                                    for _, row in tqdm(final_comments.iterrows(),total=final_comments.shape[0])]).reset_index()
        final_comments.columns = ['comment', 'attack']

        alogger.info(
            "Read all comments; there are {} out of {} that are considered 'attack'".
                format(comments.loc[comments['attack'] == True].shape[0], comments.shape[0]))
        return cls(comments, final_comments, alogger)

    def interactions(self):
        return [c for c in self.comments['comment'].tolist()]

    def labels(self):
        return ["THREAT" if threat else "CLEAN" for threat in self.comments['attack'].tolist()]

if __name__ == "__main__":
    # import sys
    #
    # sys.path.append("/Users/luisd/dev/cyberbullying-detection")

    from util.Util import *
    from phonemes_from_graphemes import *
    from words_2_vectors import *
    import os

    #
    data_dir = "./data"
    common_logger = get_logger(name="common_logger", debug_log_file_name="common_logger.log")
    print("Debug will be written in {}".format(common_logger.handlers[1].baseFilename))
    # let's go to the appropriate directory
    root_dir = get_git_root()
    common_logger.info("Switching to directory '{}'".format(root_dir))
    os.chdir(root_dir)
    mw = ModelWrapper.from_google_news_model(data_dir=data_dir, alogger=common_logger)
    sounds_dict = SoundsDict(a_dir=data_dir, alogger=common_logger)
    mw.set_sounds_dict(sounds_dict=sounds_dict)
    phonemesFactory = PhonemesFromGraphemes(alogger=common_logger)
    wiki_parser = Wikipedia_Talk_Parser.from_sources(pg=phonemesFactory, mw=mw, alogger=common_logger)
    wiki_parser.save("dataframe_wiki.csv")
