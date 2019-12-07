# Adapted from github.com/eladhoffer/seq2seq.pytorch
# Licensed under the MIT License

import heapq

import torch
import torch.nn.functional as F


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, indices, state, logits, score, context):
        """Initializes the Sequence.

        Args:
          indices: List of word ids in the sequence.
          state: Model state after generating the previous word.
          score: Score of the sequence.
        """
        self.indices = indices
        self.state = state
        self.logits = logits
        self.score = score
        self.context = context

    def __cmp__(self, other):
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class BeamSearch:
    def __init__(self, config, decoder, vocab):
        self.config = config
        self.decoder = decoder
        self.vocab_size = vocab.target_vocab.size
        self.eos_index = vocab.target_vocab.word_to_index[vocab.target_vocab.special_words.EOS]

    def decode(self, contexts, context_valid_mask):
        batch_size = contexts.shape[0]
        initial_inputs = self.decoder.init_input(batch_size)
        initial_state = self.decoder.init_state(contexts, context_valid_mask, batch_size)

        partial_sequences = [TopN(self.config.beam_width) for _ in range(batch_size)]
        complete_sequences = [TopN(self.config.beam_width) for _ in range(batch_size)]

        logits, h_0, c_0 = self.decoder(initial_inputs, *initial_state, contexts)
        top_k = F.softmax(logits).topk(self.config.beam_width)

        for b in range(batch_size):
            # Create first beam_size candidate hypotheses for each entry in batch
            for k in range(self.config.beam_width):
                seq = Sequence(
                    indices=[top_k.indices[b][k]],
                    state=(h_0[b], c_0[b]),
                    logits=[logits[b]],
                    score=top_k.values[b][k],
                    context=contexts[b])
                partial_sequences[b].push(seq)

        # Run beam search for rest of the iterations
        for i0 in range(1, self.config.max_target_length):
            partial_sequences_list = [p.extract() for p in partial_sequences]
            for p in partial_sequences:
                p.reset()

            # Keep a flattened list of partial hypotheses to easily feed through as a batch
            flattened_partial = [s for sub_partial in partial_sequences_list for s in sub_partial]
            inputs = torch.cat([c.indices[-1].unsqueeze(0) for c in flattened_partial])
            h_0 = torch.cat([c.state[0].unsqueeze(0) for c in flattened_partial])
            c_0 = torch.cat([c.state[1].unsqueeze(0) for c in flattened_partial])
            contexts = torch.cat([c.context.unsqueeze(0) for c in flattened_partial])

            if not flattened_partial:
                # We have run out of partial candidates
                break

            # Feed current hypotheses through the model
            logits, h_0, c_0 = self.decoder(inputs, h_0, c_0, contexts)
            top_k = F.softmax(logits).topk(self.config.beam_width)

            for b in range(batch_size):
                # For every entry in batch, find and trim to the most likely beam_size hypotheses
                for partial in partial_sequences_list[b]:
                    state = h_0[b], c_0[b]
                    for k in range(self.config.beam_width):
                        w = top_k.indices[b][k]
                        indices = partial.indices + [w]
                        score = partial.score + top_k.values[b][k]
                        logit_list = partial.logits + [logits[b]]

                        if w.item() == self.eos_index:
                            if self.config.target_length_norm > 0:
                                factor = self.config.target_length_norm_factor
                                length_penalty = (factor + len(indices)) / (factor + 1)
                                score /= length_penalty ** self.config.target_length_norm
                            beam = Sequence(indices, state, logit_list, score, context=partial.context)
                            complete_sequences[b].push(beam)
                        else:
                            beam = Sequence(indices, state, logit_list, score, context=partial.context)
                            partial_sequences[b].push(beam)

        # If we have no complete sequences then fall back to the partial sequences, but never output a
        # mixture of complete and partial sequences because a partial sequence could have a higher score
        # than all the complete sequences.

        for b in range(batch_size):
            if not complete_sequences[b].size():
                complete_sequences[b] = partial_sequences[b]

        return [complete.extract(sort=True) for complete in complete_sequences]
