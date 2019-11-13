import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.common.metrics import TopKAccuracyMetric, SubtokenCompositionMetric
from lib.common.processors import Code2VecMetricOutputProcessor
from lib.data.dataset import JavaSummarizationDataset

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class Code2VecEvaluator(object):
    def __init__(self, config, model, reader, split):
        self.config = config
        self.model = model
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
        self.eval_data = JavaSummarizationDataset(config, reader, split)
        self.metric_output_processor = Code2VecMetricOutputProcessor(config.top_k, reader.vocab)
        self.top_k_accuracy_metric = TopKAccuracyMetric(config.top_k, reader.vocab.target_vocab.special_words)
        self.subtoken_composition_metric = SubtokenCompositionMetric(reader.vocab.target_vocab.special_words)

    def get_scores(self, silent=False):
        total_loss = 0
        nb_eval_steps = 0

        self.model.eval()
        eval_dataloader = DataLoader(self.eval_data, shuffle=True, batch_size=self.config.batch_size)

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            source_token_indices = batch.source_token_indices.to(self.config.device)
            path_indices = batch.path_indices.to(self.config.device)
            target_token_indices = batch.target_token_indices.to(self.config.device)
            context_valid_mask = batch.context_valid_mask.to(self.config.device)
            target_indices = batch.target_index.to(self.config.device)

            with torch.no_grad():
                logits = self.model(source_token_indices, path_indices, target_token_indices, context_valid_mask)

            loss = self.loss_function(logits, target_indices)
            top_k_output = self.metric_output_processor.process(logits, target_indices)
            self.top_k_accuracy_metric.update_batch(top_k_output)
            self.subtoken_composition_metric.update_batch(top_k_output)

            if self.config.n_gpu > 1:
                loss = loss.mean()

            nb_eval_steps += 1
            total_loss += loss.item()

        accuracy = self.top_k_accuracy_metric.top_k_accuracy[0]
        precision = self.subtoken_composition_metric.precision
        recall = self.subtoken_composition_metric.recall
        f1 = self.subtoken_composition_metric.f1
        avg_loss = total_loss / nb_eval_steps

        return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'avg_loss']
