import bisect
import logging

import gensim
import numpy as np
from typing import List, Dict

from phonemes_from_graphemes import SoundsDict
import os
from util import Util

class ModelWrapper():
    default_shelf_filename = 'shelf_from0_for2999999.shelf'

    def __init__(self, alogger: logging.Logger, m, sounds_dict: SoundsDict = None):
        """
        
        :param alogger: 
        :param m: 
        :param sounds_dict: 
        """
        # am I really being initialized with a proper model?
        try:
            dummy = m.word_vec("word")
        except AttributeError:
            err_msg = "Object is not being initialized with a valid model"
            alogger.error(err_msg)
            raise RuntimeError(err_msg)
        # ok then!
        # self.data_dir = data_dir
        self.alogger = alogger
        self.model = m
        self.sounds_dict = sounds_dict
        # sort all the words in the model, so that we can auto-complete queries quickly
        self.alogger.info("Sort all the words in the model, so that we can auto-complete queries quickly...")
        self.orig_words = [gensim.utils.to_unicode(word) for word in self.model.index2word]
        indices = [i for i, _ in sorted(enumerate(self.orig_words), key=lambda item: item[1].lower())]
        self.all_words = [self.orig_words[i].lower() for i in indices]  # lowercased, sorted as lowercased
        self.orig_words = [self.orig_words[i] for i in indices]  # original letter casing, but sorted as if lowercased
        self.alogger.debug("Model wrapper successfully initialized")


    @classmethod
    def from_google_news_model(cls, data_dir: str, alogger: logging.Logger):
        if not os.path.exists(data_dir):
            alogger.info("Creating directory '{}'".format(data_dir))
            os.makedirs(data_dir)
        f_name = '{}/GoogleNews-vectors-negative300.bin.gz'.format(data_dir)
        MODEL_ON_GOOGLE_NEWS = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
        if not os.path.isfile(f_name):
            alogger.info("Downloading '{}' to '{}'".format(MODEL_ON_GOOGLE_NEWS, f_name))
            Util.download_file(MODEL_ON_GOOGLE_NEWS, f_name)
        alogger.info("'{}' is downloaded as '{}'".format(MODEL_ON_GOOGLE_NEWS, f_name))
        alogger.info("Loading model from {}...".format(f_name))
        model = gensim.models.word2vec.KeyedVectors.load_word2vec_format(f_name, binary=True)
        alogger.info("Model succesfully loaded")
        return cls(alogger=alogger, m=model)

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