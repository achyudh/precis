import abc
from argparse import Namespace
from typing import Tuple


class ContextReader(abc.ABC):
    def __init__(self, config: Namespace):
        self.config = config

    def process_input_row(self, index, row_placeholder) -> Tuple:
        parts = row_placeholder.split()
        return self._get_input_tensors(index, *parts)

    @abc.abstractmethod
    def _get_input_tensors(self, *row_parts) -> Tuple:
        pass
