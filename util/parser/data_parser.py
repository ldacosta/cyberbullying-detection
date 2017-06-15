from typing import List
from util.Util import smart_split_in_words
import itertools

def clean_raw_string(pg, mw, raw: str) -> List[str]:
    graphs_and_phons = pg.graphemes_to_phonemes(words_in_sent=smart_split_in_words(raw))
    graphs_and_phons_ext = [(graph, ph, list(set(map(lambda w: w.lower(), mw.safe_sound_to_word(ph))))) for graph, ph
                            in graphs_and_phons]
    graphs_and_phons_ext_winners = [(graph, ph, cands, [graph] if graph.lower() in cands else cands) for graph, ph, cands in
                                    graphs_and_phons_ext]
    sounds_to_words = [winners for _, _, _, winners in graphs_and_phons_ext_winners if len(winners) > 0]
    alls_as_tuples = [elt for elt in itertools.product(*sounds_to_words)]
    alls_as_strings = [' '.join(a_tuple) for a_tuple in alls_as_tuples]
    return alls_as_strings


class Data_Parser(object):

    def __init__(self):
        pass

    def interactions(self) -> List[str]:
        raise RuntimeError("Not defined")

    def labels(self) -> List[str]:
        raise RuntimeError("Not defined")