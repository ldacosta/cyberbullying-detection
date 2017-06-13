import spacy # nl processing
from typing import List
import numpy as np
import random
import keras

empty_string = '<word_not_in_model>' # need a fill-up word for too-short sentences

class review(object):
    """
    Just a wrapper class for reviews.
    """
    def __init__(self, review_as_str: str):
        self.as_str = review_as_str

    def as_word_list(self) -> List[str]:
        return self.as_str.split()

class label(object):
    """
    A wrapper class for labels (threat or not) 
    """

    THREAT = "threat"
    CLEAN = "clean"

    def __init__(self, is_threat: bool):
        self.value = self.THREAT if is_threat else self.CLEAN

    @classmethod
    def from_string(cls, s: str):
        assert s in { cls.THREAT, cls.CLEAN }, "String value of label must be either '{}' or {}".format(cls.THREAT, cls.CLEAN)
        return cls(is_threat=True if s.lower() == cls.THREAT else False)

    def as_categories(self) -> List[int]:
        return keras.utils.to_categorical([0 if self.value == self.CLEAN else 1], num_classes=2)

class reviews_labels_encoder(object):

    # TODO: specify a logger
    def __init__(self, mw, n_words_in_review: int, reviews: List[review], labels: List[label], spacy_nlp = None):
        """
        Creates an instance of this object. 
        :param mw: Model to transform word -> vector 
        :param n_words_in_review: maximum number of words I will take from the review
        :param reviews: all reviews; label[i] corresponds ro review[i]
        :param labels: all labels 
        """
        # nlp stuff
        if spacy_nlp is None:
            self.nlp = spacy.load('en')
        else:
            self.nlp = spacy_nlp
        self.probs = [lex.prob for lex in self.nlp.vocab]
        #
        self.mw = mw
        self.n_words_in_review = n_words_in_review  #
        self.reviews = reviews
        self.labels = labels

    def reviews_as_vec(self) -> List[np.array]:
        return [self.words2repr(r.as_word_list()) for r in self.reviews]

    def reviews_as_matrix(self) -> np.array:
        return np.array(self.reviews_as_vec())

    def labels_as_matrix(self) -> np.array:
        return np.concatenate(list(map(lambda l: l.as_categories(), self.labels)))

    def __padWords2size__(self, rev_as_list: List[str]):
        avail_words = len(rev_as_list[:self.n_words_in_review])
        num_empty_words = self.n_words_in_review - avail_words
        return rev_as_list[:self.n_words_in_review] + [empty_string]*num_empty_words
        # return np.array([mw.vec_repr(w) for w in rev_right_size])

    def words2repr(self, rev_as_list: List[str]):
        rev_right_size = self.__padWords2size__(rev_as_list)
        return np.array([self.mw.vec_repr(w) for w in rev_right_size])


    def get_x_and_y_from(self, idxs, a_size):
        assert a_size <= len(idxs), "Can't choose {} elts from set of {}".format(a_size, len(idxs))
        random.shuffle(idxs)
        batch_idxs = idxs[:a_size]
        # process data: x
        reviews_in_batch = [self.reviews[i] for i in batch_idxs] # reviews[batch_idxs]
        batch_as_words = [ [w.text for w in self.nlp(rev) if self.nlp.vocab[w.text].prob < self.probs[-1000]] for rev in reviews_in_batch]
        batch_x = np.array([self.words2repr(words_in_review) for words_in_review in batch_as_words])
    #     batch_as_words = [ [w.text for w in nlp(rev) if nlp.vocab[w.text].prob < probs[-1000]] for rev in tqdm(reviews_in_batch, total=len(reviews_in_batch))]
    #     batch_x = np.array([words2repr(words_in_review, n_words_in_review) for words_in_review in tqdm(batch_as_words, total=len(batch_as_words))])
        # y
        labels_slice = [self.labels[i] for i in batch_idxs] # labels[batch_idxs]
        batch_y = np.array([[1, 0] if (w == 'POSITIVE') else [0, 1] for w in labels_slice]).reshape([len(labels_slice),2])
        assert batch_x.shape[0] == batch_y.shape[0], "Sanity check failed: reviews and labels have different sizes"
        return (batch_x, batch_y)

