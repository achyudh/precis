import pickle
from argparse import Namespace
from typing import Optional, Tuple

from .vocab_container import VocabContainer
from .vocab_map import VocabMap
from .vocab_type import VocabType, get_special_words


class PathContextVocabContainer(VocabContainer):
    def __init__(self, config: Namespace, target_vocab_type=VocabType.Target):
        super().__init__(config)
        self.token_vocab: Optional[VocabMap] = None
        self.path_vocab: Optional[VocabMap] = None
        self.target_vocab: Optional[VocabMap] = None
        self._from_word_freq_dict(target_vocab_type)

    def get(self, vocab_type: VocabType) -> VocabMap:
        if vocab_type == VocabType.Token:
            return self.token_vocab
        if vocab_type == VocabType.Target:
            return self.target_vocab
        if vocab_type == VocabType.Path:
            return self.path_vocab

    def _from_word_freq_dict(self, target_vocab_type):
        token_to_count, path_to_count, target_to_count = self._load_word_freq_dict()

        self.token_vocab = VocabMap.from_word_freq_dict(
            VocabType.Token, token_to_count, self.config.max_token_vocab_size,
            special_words=get_special_words(VocabType.Token))
        print('Token vocab. size: %d' % self.token_vocab.size)

        self.path_vocab = VocabMap.from_word_freq_dict(
            VocabType.Path, path_to_count, self.config.max_path_vocab_size,
            special_words=get_special_words(VocabType.Path))
        print('Path vocab. size: %d' % self.path_vocab.size)

        self.target_vocab = VocabMap.from_word_freq_dict(
            target_vocab_type, target_to_count, self.config.max_target_vocab_size,
            special_words=get_special_words(target_vocab_type))
        print('Target vocab. size: %d' % self.target_vocab.size)

    def _load_word_freq_dict(self) -> Tuple:
        with open(self.config.word_freq_dict_path, 'rb') as file:
            token_to_count = pickle.load(file)
            path_to_count = pickle.load(file)
            target_to_count = pickle.load(file)
        return token_to_count, path_to_count, target_to_count
