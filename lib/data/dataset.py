import abc
from argparse import Namespace
from enum import Enum

from torch.utils.data import Dataset

from lib.data.reader import PathContextReader, ReaderInputTensors


class AbstractInputTransformer(abc.ABC):
    """
    Should be inherited by the model implementation.
    An instance of the inherited class is passed by the model to the reader in order to help the reader
        to construct the input in the form that the model expects to receive it.
    This class also enables conveniently & clearly access input parts by their field names.
        eg: 'tensors.path_indices' instead if 'tensors[1]'.
    This allows the input tensors to be passed as pure tuples along the computation graph, while the
        python functions that construct the graph can easily (and clearly) access tensors.
    """

    @abc.abstractmethod
    def to_model_input_form(self, input_tensors: ReaderInputTensors):
        pass

    @abc.abstractmethod
    def from_model_input_form(self, input_row) -> ReaderInputTensors:
        pass


class DatasetSplit(Enum):
    Train = 'train'
    Dev = 'dev'
    Test = 'test'

    @property
    def is_train(self):
        return self is DatasetSplit.Train

    @property
    def is_dev(self):
        return self is DatasetSplit.Dev

    @property
    def is_test(self):
        return self is DatasetSplit.Test


class JavaSummarizationDataset(Dataset):
    def __init__(self, config: Namespace, reader: PathContextReader, split: DatasetSplit):
        self.input_tensors = list()
        with open(config.data_path(split)) as csv_file:
            for input_row in csv_file:
                input_tensor = reader.process_input_row(input_row.strip())
                if reader.is_valid_input_row(input_tensor, split):
                    self.input_tensors.append(input_tensor)

    def __getitem__(self, idx: int) -> ReaderInputTensors:
        return self.input_tensors[idx]

    def __len__(self) -> int:
        return len(self.input_tensors)
