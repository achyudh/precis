import abc
from argparse import Namespace

from .vocab_type import VocabType
from .vocab_map import VocabMap


class VocabContainer(abc.ABC):
    def __init__(self, config: Namespace):
        self.config = config

    @abc.abstractmethod
    def get(self, vocab_type: VocabType) -> VocabMap:
        pass
