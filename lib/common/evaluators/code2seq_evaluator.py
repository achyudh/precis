import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.common.metrics import TopKAccuracyMetric, SubtokenCompositionMetric
from lib.common.processors import Code2SeqMetricOutputProcessor
from lib.data.dataset import JavaSummarizationDataset
from lib.util import BeamSearch


class Code2SeqEvaluator(object):
    def __init__(self, config, model, reader, split):
        self.config = config
        self.model = model
        self.vocab = reader.vocab

        target_pad_index = self.vocab.target_vocab.word_to_index[self.vocab.target_vocab.special_words.PAD]
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=target_pad_index)

        self.eval_data = JavaSummarizationDataset(config, reader, split)
        self.beam_search = BeamSearch(config, model.decoder, reader.vocab)
        self.metric_output_processor = Code2SeqMetricOutputProcessor(config, self.vocab, self.eval_data)

    def get_scores(self, silent=False):
        total_loss = 0
        nb_eval_steps = 0

        self.model.eval()
        eval_dataloader = DataLoader(self.eval_data, shuffle=True, batch_size=self.config.batch_size)
        top_k_accuracy_metric = TopKAccuracyMetric(1, self.vocab.target_vocab.special_words)
        subtoken_composition_metric = SubtokenCompositionMetric(self.vocab.target_vocab.special_words)

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            source_subtoken_indices = batch.source_subtoken_indices.to(self.config.device)
            node_indices = batch.node_indices.to(self.config.device)
            target_subtoken_indices = batch.target_subtoken_indices.to(self.config.device)

            source_subtoken_lengths = batch.source_subtoken_lengths.to(self.config.device)
            node_lengths = batch.node_lengths.to(self.config.device)
            target_subtoken_lengths = batch.target_subtoken_lengths.to(self.config.device)

            context_valid_mask = batch.context_valid_mask.to(self.config.device)
            label_indices = batch.label_indices.to(self.config.device)

            with torch.no_grad():
                context_embed = self.model(source_subtoken_indices, node_indices, target_subtoken_indices,
                                           source_subtoken_lengths, node_lengths, target_subtoken_lengths,
                                           context_valid_mask)

                top_k_sequences = self.beam_search.decode(context_embed, context_valid_mask)
                logits = torch.cat([sequence[0].logits[0].unsqueeze(0) for sequence in top_k_sequences])

            loss = self.loss_function(logits, label_indices[:, 0])
            top_k_output = self.metric_output_processor.process(top_k_sequences, batch.sample_index)
            top_k_accuracy_metric.update_batch(top_k_output)
            subtoken_composition_metric.update_batch(top_k_output)

            nb_eval_steps += 1
            total_loss += loss.item()

        accuracy = top_k_accuracy_metric.top_k_accuracy[0]
        precision = subtoken_composition_metric.precision
        recall = subtoken_composition_metric.recall
        f1 = subtoken_composition_metric.f1
        avg_loss = total_loss / nb_eval_steps

        return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'avg_loss']
