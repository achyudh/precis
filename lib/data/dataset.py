from argparse import Namespace
from enum import Enum

import torch
from torch.utils.data import Dataset

from lib.data.reader import PathContextReader, PathContextInput


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
        self.target_labels = list()

        with open(config.data_path(split)) as csv_file:
            for input_row in csv_file:
                input_tensor, target_label = reader.process_input_row(index=len(self.input_tensors),
                                                                      row_placeholder=input_row.strip())
                if reader.is_valid_input_row(input_tensor, split):
                    self.input_tensors.append(input_tensor)
                    self.target_labels.append(target_label)

                if len(self.input_tensors) == 512:
                    break

    def __getitem__(self, idx: int) -> PathContextInput:
        return self.input_tensors[idx]

    def __len__(self) -> int:
        return len(self.input_tensors)
