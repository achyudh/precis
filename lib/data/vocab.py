import abc
import pickle
from argparse import Namespace
from collections import OrderedDict
from enum import Enum
from itertools import chain
from typing import Optional, Dict, Iterable, Set, Tuple


class VocabType(Enum):
    Token = 1
    Target = 2
    Path = 3
    Sequence = 4


_SpecialVocabWords_OnlyOov = Namespace(OOV='<OOV>')
_SpecialVocabWords_OovAndPad = Namespace(PAD='<PAD>', OOV='<OOV>')
_SpecialVocabWords_OovPadSeq = Namespace(PAD='<PAD>', OOV='<OOV>', SOS='<SOS>', EOS='<EOS>')


def get_unique_list(lst: Iterable) -> list:
    return list(OrderedDict(((item, 0) for item in lst)).keys())


def get_special_words_by_vocab_type(vocab_type: VocabType) -> Namespace:
    if vocab_type == VocabType.Target:
        return _SpecialVocabWords_OnlyOov
    elif vocab_type == VocabType.Sequence:
        return _SpecialVocabWords_OovPadSeq
    else:
        return _SpecialVocabWords_OovAndPad


class Vocab:
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
    def create_from_freq_dict(cls, vocab_type: VocabType, word_to_count: Dict[str, int], max_size: int,
                              special_words: Optional[Namespace] = None):
        if special_words is None:
            special_words = Namespace()
        words_sorted_by_counts = sorted(word_to_count, key=word_to_count.get, reverse=True)
        words_sorted_by_counts_and_limited = words_sorted_by_counts[:max_size]
        return cls(vocab_type, words_sorted_by_counts_and_limited, special_words)

    def lookup_index(self, word: str) -> int:
        return self.word_to_index.get(word, self.word_to_index[self.special_words.OOV])

    def lookup_word(self, index: int) -> str:
        return self.index_to_word.get(index, self.special_words.OOV)


class AbstractVocabContainer(abc.ABC):
    def __init__(self, config: Namespace):
        self.config = config

    def _load_word_freq_dict(self) -> Tuple:
        with open(self.config.word_freq_dict_path, 'rb') as file:
            token_to_count = pickle.load(file)
            path_to_count = pickle.load(file)
            target_to_count = pickle.load(file)
        return token_to_count, path_to_count, target_to_count


class Code2VecVocabContainer(AbstractVocabContainer):
    def __init__(self, config: Namespace):
        super().__init__(config)
        self.token_vocab: Optional[Vocab] = None
        self.path_vocab: Optional[Vocab] = None
        self.target_vocab: Optional[Vocab] = None
        self._create_from_word_freq_dict()

    def _create_from_word_freq_dict(self):
        token_to_count, path_to_count, target_to_count = self._load_word_freq_dict()

        self.token_vocab = Vocab.create_from_freq_dict(
            VocabType.Token, token_to_count, self.config.max_token_vocab_size,
            special_words=get_special_words_by_vocab_type(VocabType.Token))
        print('Token vocab. size: %d' % self.token_vocab.size)

        self.path_vocab = Vocab.create_from_freq_dict(
            VocabType.Path, path_to_count, self.config.max_path_vocab_size,
            special_words=get_special_words_by_vocab_type(VocabType.Path))
        print('Path vocab. size: %d' % self.path_vocab.size)

        self.target_vocab = Vocab.create_from_freq_dict(
            VocabType.Target, target_to_count, self.config.max_target_vocab_size,
            special_words=get_special_words_by_vocab_type(VocabType.Target))
        print('Target vocab. size: %d' % self.target_vocab.size)

    def get(self, vocab_type: VocabType) -> Vocab:
        if vocab_type == VocabType.Token:
            return self.token_vocab
        if vocab_type == VocabType.Target:
            return self.target_vocab
        if vocab_type == VocabType.Path:
            return self.path_vocab


class Code2SeqVocabContainer(AbstractVocabContainer):
    def __init__(self, config: Namespace):
        super().__init__(config)
        self.subtoken_vocab: Optional[Vocab] = None
        self.node_vocab: Optional[Vocab] = None
        self.target_vocab: Optional[Vocab] = None
        self._create_from_word_freq_dict()

    def _create_from_word_freq_dict(self):
        subtoken_to_count, node_to_count, target_to_count = self._load_word_freq_dict()

        self.subtoken_vocab = Vocab.create_from_freq_dict(
            VocabType.Token, subtoken_to_count, self.config.max_subtoken_vocab_size,
            special_words=get_special_words_by_vocab_type(VocabType.Token))
        print('Subtoken vocab. size: %d' % self.subtoken_vocab.size)

        self.node_vocab = Vocab.create_from_freq_dict(
            VocabType.Path, node_to_count, self.config.max_node_vocab_size,
            special_words=get_special_words_by_vocab_type(VocabType.Path))
        print('Node vocab. size: %d' % self.node_vocab.size)

        self.target_vocab = Vocab.create_from_freq_dict(
            VocabType.Sequence, target_to_count, self.config.max_target_vocab_size,
            special_words=get_special_words_by_vocab_type(VocabType.Sequence))
        print('Target vocab. size: %d' % self.target_vocab.size)
