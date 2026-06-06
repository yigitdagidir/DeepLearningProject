"""
Everything related to turning a caption (a raw string) into a list of integer
token ids that the embedding layer can consume.

We keep the tokenizer deliberately simple (lower-case + split on non letters)
because the captions in this kind of dataset are short and clean.  A heavier
tokenizer (e.g. the one that comes with BERT) could be plugged in here without
changing the rest of the code.
"""

import re
from collections import Counter
from typing import List

# special tokens
PAD = "<pad>"     # padding so every caption in a batch has the same length
UNK = "<unk>"     # words that are not in the vocabulary
SOS = "<sos>"     # start of sentence
EOS = "<eos>"     # end of sentence
SPECIALS = [PAD, UNK, SOS, EOS]


def tokenize(text: str) -> List[str]:
    """'A blue Cotton T-shirt!' -> ['a', 'blue', 'cotton', 't', 'shirt']"""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)   # keep only letter sequences
    return tokens


class Vocabulary:
    """Maps words <-> integer ids.  Built from the training captions only."""

    def __init__(self):
        self.itos = {}          # id  -> string
        self.stoi = {}          # string -> id
        for tok in SPECIALS:
            self._add_word(tok)

    def _add_word(self, word: str) -> int:
        if word not in self.stoi:
            idx = len(self.itos)
            self.itos[idx] = word
            self.stoi[word] = idx
        return self.stoi[word]

    def build(self, captions: List[str], min_freq: int = 1) -> "Vocabulary":
        counter = Counter()
        for cap in captions:
            counter.update(tokenize(cap))
        # add words sorted by frequency (deterministic ordering)
        for word, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq:
                self._add_word(word)
        return self

    def __len__(self):
        return len(self.itos)

    @property
    def pad_idx(self):
        return self.stoi[PAD]

    def encode(self, text: str, max_len: int) -> List[int]:
        """Caption -> fixed length list of ids (with <sos>/<eos>, padded/truncated)."""
        tokens = [SOS] + tokenize(text) + [EOS]
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens]
        if len(ids) < max_len:
            ids = ids + [self.pad_idx] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            ids[-1] = self.stoi[EOS]      # keep an <eos> at the end even if truncated
        return ids

    def decode(self, ids: List[int]) -> str:
        words = [self.itos.get(i, UNK) for i in ids
                 if i not in (self.pad_idx,)]
        return " ".join(words)
