import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.common.metrics import TopKAccuracyMetric, SubtokenCompositionMetric
from lib.common.processors import Code2VecMetricOutputProcessor
from lib.data.dataset import JavaSummarizationDataset

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class Code2SeqEvaluator(object):
    def __init__(self, config, model, reader, split):
        self.model = model
        self.config = config
        self.reader = reader
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
        self.eval_data = JavaSummarizationDataset(config, reader, split)
        self.metric_output_processor = Code2VecMetricOutputProcessor(config.top_k, reader.vocab, self.eval_data)

    def get_scores(self, silent=False):
        total_loss = 0
        nb_eval_steps = 0

        self.model.eval()
        eval_dataloader = DataLoader(self.eval_data, shuffle=True, batch_size=self.config.batch_size)
        top_k_accuracy_metric = TopKAccuracyMetric(self.config.top_k, self.reader.vocab.target_vocab.special_words)
        subtoken_composition_metric = SubtokenCompositionMetric(self.reader.vocab.target_vocab.special_words)

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            source_token_indices = batch.source_token_indices.to(self.config.device)
            path_indices = batch.path_indices.to(self.config.device)
            target_token_indices = batch.target_token_indices.to(self.config.device)
            context_valid_mask = batch.context_valid_mask.to(self.config.device)
            target_indices = batch.target_index.to(self.config.device)

            with torch.no_grad():
                logits = self.model(source_token_indices, path_indices, target_token_indices, context_valid_mask)

            loss = self.loss_function(logits, target_indices)
            top_k_output = self.metric_output_processor.process(logits, batch.sample_index)
            top_k_accuracy_metric.update_batch(top_k_output)
            subtoken_composition_metric.update_batch(top_k_output)

            if self.config.n_gpu > 1:
                loss = loss.mean()

            nb_eval_steps += 1
            total_loss += loss.item()

        accuracy = top_k_accuracy_metric.top_k_accuracy[0]
        precision = subtoken_composition_metric.precision
        recall = subtoken_composition_metric.recall
        f1 = subtoken_composition_metric.f1
        avg_loss = total_loss / nb_eval_steps

        return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'avg_loss']
