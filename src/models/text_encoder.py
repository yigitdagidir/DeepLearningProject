"""
Text branch of the model.

Pipeline:  word ids  ->  embedding layer  ->  RNN (LSTM or GRU)  ->  text vector

* The embedding layer turns each word id into a dense `embed_dim` vector.  It
  can be trained from scratch (default) or initialised from pre-trained GloVe
  vectors (see `load_glove`).
* The RNN reads the sequence of embeddings one step at a time and keeps a hidden
  state that summarises everything seen so far.  We use the final hidden state
  as the encoding of the whole caption.
* `pack_padded_sequence` makes the RNN ignore the <pad> tokens so that padding
  does not influence the final state.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 embed_dim: int = 128,
                 rnn_type: str = "lstm",
                 hidden: int = 256,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.num_directions = 2 if bidirectional else 1
        self.out_dim = hidden * self.num_directions

    def load_glove(self, glove_path: str, vocab) -> int:
        """Optionally overwrite the embedding weights with GloVe vectors."""
        import numpy as np
        emb = self.embedding.weight.data
        found = 0
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab.stoi and len(parts) - 1 == emb.size(1):
                    vec = torch.tensor([float(x) for x in parts[1:]])
                    emb[vocab.stoi[word]] = vec
                    found += 1
        print(f"[text] loaded {found} GloVe vectors")
        return found

    def forward(self, text):                         # text: (B, T) of word ids
        lengths = (text != self.pad_idx).sum(dim=1)  # true length of each caption
        lengths = lengths.clamp(min=1)               # avoid zero-length sequences
        embedded = self.embedding(text)              # (B, T, embed_dim)

        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(packed)           # h_n: (layers*dir, B, hidden)
        else:
            _, h_n = self.rnn(packed)

        # take the last layer's hidden state(s); concat the two directions
        if self.num_directions == 2:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (B, 2*hidden)
        else:
            last = h_n[-1]                                  # (B, hidden)
        return last                                         # (B, out_dim)
