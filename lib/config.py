import os
from argparse import ArgumentParser

from lib.data.dataset import DatasetSplit


class Config:
    @classmethod
    def get_args(cls) -> ArgumentParser:
        parser = ArgumentParser()

        parser.add_argument('--no-cuda', action='store_false', dest='cuda')
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--seed', type=int, default=3435)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--patience', type=int, default=5)
        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--weight-decay', type=float, default=0)
        parser.add_argument('--dropout-rate', type=float, default=0.25)

        parser.add_argument('--dataset', type=str, default='java-small', choices=['java-small'])
        parser.add_argument('--data-dir', default=os.path.join(os.pardir, 'precis-data', 'datasets'))
        parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'code2vec'))

        parser.add_argument('--resume-snapshot', type=str)
        parser.add_argument('--pretrained-model', type=str)

        return parser

    def data_path(self, split: DatasetSplit):
        if split.is_train:
            return self.train_data_path
        elif split.is_dev:
            return self.dev_data_path
        else:
            return self.test_data_path
