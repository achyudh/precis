import datetime
import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from lib.common.evaluators import ConvPathAttnEvaluator
from lib.data.dataset import DatasetSplit


class ConvPathAttnTrainer(object):
    def __init__(self, config, model, dataset, reader, optimizer):
        self.config = config
        self.model = model
        self.vocab = reader.vocab
        self.optimizer = optimizer

        self.loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
        self.train_dataset = dataset(config, reader, split=DatasetSplit.Train)
        self.dev_evaluator = ConvPathAttnEvaluator(config, model, reader, split=DatasetSplit.Dev)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.config.save_path, self.config.dataset, '%s.pt' % timestamp)

        self.nb_train_steps = 0
        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

    def train_epoch(self, train_dataloader):
        for batch in tqdm(train_dataloader, desc="Training"):
            self.model.train()
            self.optimizer.zero_grad()

            source_subtoken_indices = batch.source_subtoken_indices.to(self.config.device)
            node_indices = batch.node_indices.to(self.config.device)
            target_subtoken_indices = batch.target_subtoken_indices.to(self.config.device)

            source_subtoken_lengths = batch.source_subtoken_lengths.to(self.config.device)
            node_lengths = batch.node_lengths.to(self.config.device)
            target_subtoken_lengths = batch.target_subtoken_lengths.to(self.config.device)

            context_valid_mask = batch.context_valid_mask.to(self.config.device)
            label_indices = batch.label_index.to(self.config.device)

            logits = self.model(source_subtoken_indices, node_indices, target_subtoken_indices,
                                       source_subtoken_lengths, node_lengths, target_subtoken_lengths,
                                       context_valid_mask)

            loss = self.loss_function(logits, label_indices)
            if self.config.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            self.optimizer.step()
            self.nb_train_steps += 1

    def train(self):
        best_dev_f1 = 0
        unimproved_epochs = 0
        print("Batch size:", self.config.batch_size)
        print("Number of examples: ", len(self.train_dataset))

        # Initialize a dataloader with train_dataset
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.config.batch_size)

        for epoch in trange(int(self.config.epochs), desc="Epoch"):
            self.train_epoch(train_dataloader)
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = self.dev_evaluator.get_scores()[0]

            # Print validation results
            tqdm.write(self.log_header)
            tqdm.write(self.log_template.format(epoch + 1, self.nb_train_steps, epoch + 1, self.config.epochs,
                                                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss))

            # Update validation results
            if dev_f1 > best_dev_f1:
                unimproved_epochs = 0
                best_dev_f1 = dev_f1
                torch.save(self.model, self.snapshot_path)
            else:
                unimproved_epochs += 1
                if unimproved_epochs >= self.config.patience:
                    break
