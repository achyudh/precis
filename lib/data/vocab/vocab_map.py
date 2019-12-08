from argparse import Namespace
from itertools import chain
from typing import Iterable, Optional, Dict

from lib.util.preprocessing import get_unique_list
from .vocab_type import VocabType


class VocabMap:
    def __init__(self, vocab_type: VocabType, words: Iterable[str], special_words: Optional[Namespace] = None):
        if special_words is None:
            special_words = Namespace()

        self.vocab_type = vocab_type
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: Dict[int, str] = {}
        self._word_to_index_lookup_table = None
        self._index_to_word_lookup_table = None
        self.special_words: Namespace = special_words

        for index, word in enumerate(chain(get_unique_list(special_words.__dict__.values()), words)):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        self.size = len(self.word_to_index)

    @classmethod
    def from_word_freq_dict(cls, vocab_type: VocabType, word_freq_dict: Dict[str, int], max_size: int,
                            special_words: Optional[Namespace] = None):
        words_sorted_by_counts = sorted(word_freq_dict, key=word_freq_dict.get, reverse=True)
        words_sorted_by_counts_and_limited = words_sorted_by_counts[:max_size]
        return cls(vocab_type, words_sorted_by_counts_and_limited, special_words)

    def lookup_index(self, word: str) -> int:
        return self.word_to_index.get(word, self.word_to_index[self.special_words.OOV])

    def lookup_word(self, index: int) -> str:
        return self.index_to_word.get(index, self.special_words.OOV)


