import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.data.dataset import JavaSummarizationDataset

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class Code2VecEvaluator(object):
    def __init__(self, config, model, reader, split):
        self.config = config
        self.model = model
        self.eval_data = JavaSummarizationDataset(config, reader, split)

    def get_scores(self, silent=False):
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_data, shuffle=True, batch_size=self.config.batch_size)

        total_loss = 0
        nb_eval_steps = 0
        target_labels = list()
        predicted_labels = list()

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            source_token_indices = batch.source_token_indices.to(self.config.device)
            path_indices = batch.path_indices.to(self.config.device)
            target_token_indices = batch.target_token_indices.to(self.config.device)
            context_valid_mask = batch.context_valid_mask.to(self.config.device)
            target_indices = batch.target_index.to(self.config.device)

            with torch.no_grad():
                logits = self.model(source_token_indices, path_indices, target_token_indices, context_valid_mask)

            predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
            target_labels.extend(target_indices.cpu().detach().numpy())
            loss = F.cross_entropy(logits, target_indices)

            if self.config.n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            nb_eval_steps += 1

        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
        recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
        f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
        avg_loss = total_loss / nb_eval_steps

        return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'avg_loss']
