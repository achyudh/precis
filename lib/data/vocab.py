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
    SeqTarget = 5


_SpecialVocabWords_OnlyOov = Namespace(OOV='<OOV>')
_SpecialVocabWords_OovAndPad = Namespace(PAD='<PAD>', OOV='<OOV>')
_SpecialVocabWords_OovPadSos = Namespace(PAD='<PAD>', OOV='<OOV>', SOS='<S>')


def get_unique_list(lst: Iterable) -> list:
    return list(OrderedDict(((item, 0) for item in lst)).keys())


def get_special_words_by_vocab_type(vocab_type: VocabType) -> Namespace:
    if vocab_type == VocabType.Target:
        return _SpecialVocabWords_OnlyOov
    elif vocab_type == VocabType.SeqTarget:
        return _SpecialVocabWords_OovPadSos
    else:
        return _SpecialVocabWords_OovAndPad


class Vocab:
    def __init__(self, vocab_type: VocabType, words: Iterable[str],
                 special_words: Optional[Namespace] = None):
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

    def save_to_file(self, file):
        # Remove special words before saving vocab
        special_words_as_unique_list = get_unique_list(self.special_words.__dict__.values())
        nr_special_words = len(special_words_as_unique_list)
        word_to_index_wo_specials = {word: idx for word, idx in self.word_to_index.items() if idx >= nr_special_words}
        index_to_word_wo_specials = {idx: word for idx, word in self.index_to_word.items() if idx >= nr_special_words}
        size_wo_specials = self.size - nr_special_words
        pickle.dump(word_to_index_wo_specials, file)
        pickle.dump(index_to_word_wo_specials, file)
        pickle.dump(size_wo_specials, file)

    @classmethod
    def load_from_file(cls, vocab_type: VocabType, file, special_words: Namespace) -> 'Vocab':
        special_words_as_unique_list = get_unique_list(special_words.__dict__.values())

        # Add special words after loading vocab
        word_to_index_wo_specials = pickle.load(file)
        index_to_word_wo_specials = pickle.load(file)
        size_wo_specials = pickle.load(file)
        assert len(index_to_word_wo_specials) == len(word_to_index_wo_specials) == size_wo_specials
        min_word_idx_wo_specials = min(index_to_word_wo_specials.keys())

        if min_word_idx_wo_specials != len(special_words_as_unique_list):
            raise ValueError("Error while attempting to load vocabulary")

        vocab = cls(vocab_type, [], special_words)
        vocab.word_to_index = {**word_to_index_wo_specials,
                               **{word: idx for idx, word in enumerate(special_words_as_unique_list)}}
        vocab.index_to_word = {**index_to_word_wo_specials,
                               **{idx: word for idx, word in enumerate(special_words_as_unique_list)}}
        vocab.size = size_wo_specials + len(special_words_as_unique_list)
        return vocab

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

        self._already_saved_in_paths: Set[str] = set()  # To avoid re-saving a non-modified vocabulary
        self._load_or_create()

    def _load_or_create(self):
        if self.config.pretrained_model:
            vocabularies_load_path = self.config.get_vocabularies_path_from_model_path(self.config.pretrained_model)
            self._load_from_path(vocabularies_load_path)
        else:
            self._create_from_word_freq_dict()

    def _load_from_path(self, vocabularies_load_path: str):
        with open(vocabularies_load_path, 'rb') as file:
            self.token_vocab = Vocab.load_from_file(
                VocabType.Token, file, get_special_words_by_vocab_type(VocabType.Token))
            self.target_vocab = Vocab.load_from_file(
                VocabType.Target, file, get_special_words_by_vocab_type(VocabType.Target))
            self.path_vocab = Vocab.load_from_file(
                VocabType.Path, file, get_special_words_by_vocab_type(VocabType.Path))
        self._already_saved_in_paths.add(vocabularies_load_path)

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

    def save(self, vocabularies_save_path: str):
        if vocabularies_save_path in self._already_saved_in_paths:
            return
        with open(vocabularies_save_path, 'wb') as file:
            self.token_vocab.save_to_file(file)
            self.target_vocab.save_to_file(file)
            self.path_vocab.save_to_file(file)
        self._already_saved_in_paths.add(vocabularies_save_path)

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
            VocabType.SeqTarget, target_to_count, self.config.max_target_vocab_size,
            special_words=get_special_words_by_vocab_type(VocabType.SeqTarget))
        print('Target vocab. size: %d' % self.target_vocab.size)
