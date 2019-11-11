import os
import random

import numpy as np
import torch.onnx

from lib import Config
from lib.common.evaluators import Code2VecEvaluator
from lib.common.trainers import Code2VecTrainer
from lib.data.dataset import JavaSummarizationDataset, DatasetSplit
from lib.data.reader import PathContextReader
from lib.data.vocab import Code2VecVocab
from lib.model import Code2Vec

# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))


def evaluate_split(config, model, reader, split):
    evaluator = Code2VecEvaluator(config, model, reader, split)
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))


if __name__ == '__main__':
    config = Config()
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    print('Number of GPUs:', n_gpu)
    print('Device:', str(device).upper())

    # Set random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    dataset_map = {
        'java-small': JavaSummarizationDataset
    }

    if config.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    config.device = device
    config.n_gpu = n_gpu
    train_examples = None
    dataset = dataset_map[config.dataset]

    if not  config.pretrained_model:
        save_path = os.path.join(config.save_path, config.dataset)
        os.makedirs(save_path, exist_ok=True)

    vocab = Code2VecVocab(config)
    reader = PathContextReader(config, vocab)
    model = Code2Vec(config, vocab)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=config.lr, weight_decay=config.weight_decay)
    trainer = Code2VecTrainer(config, model, dataset, reader, optimizer)

    if not config.pretrained_model:
        trainer.train()
        model = torch.load(trainer.snapshot_path)
    else:
        model = torch.load(config.trained_model, map_location=lambda storage, location: storage)
        model = model.to(device)

    evaluate_split(config, model, reader, split=DatasetSplit.Dev)
    evaluate_split(config, model, reader, split=DatasetSplit.Test)
