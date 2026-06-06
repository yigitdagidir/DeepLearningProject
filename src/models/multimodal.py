"""
The full model = image branch + text branch + fusion head.

It supports three modes through `cfg.modality`:
    "image" -> only the CNN is used   (image-only baseline)
    "text"  -> only the RNN is used   (text-only baseline)
    "both"  -> the two are fused      (our proposed multi-modal model)

Having all three in one class means the baselines and the full model share the
exact same code, which makes the ablation comparison fair.
"""

import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .text_encoder import TextEncoder
from .fusion import FusionClassifier


class MultiModalNet(nn.Module):
    def __init__(self, cfg, vocab_size: int, pad_idx: int, num_classes: int):
        super().__init__()
        self.modality = cfg.modality

        # image branch (only built if we actually need images)
        if cfg.modality in ("image", "both"):
            self.cnn = CNNEncoder(cfg.cnn_backbone, cfg.pretrained,
                                  cfg.freeze_backbone, cfg.img_feat_dim)
            img_dim = self.cnn.out_dim
        else:
            self.cnn = None
            img_dim = 0

        # text branch
        if cfg.modality in ("text", "both"):
            self.txt = TextEncoder(vocab_size, pad_idx, cfg.embed_dim,
                                   cfg.rnn_type, cfg.rnn_hidden, cfg.rnn_layers,
                                   cfg.bidirectional)
            txt_dim = self.txt.out_dim
        else:
            self.txt = None
            txt_dim = 0

        self.head = FusionClassifier(img_dim, txt_dim, num_classes,
                                     cfg.fusion_hidden, cfg.dropout, cfg.modality)

    def forward(self, images, text):
        img_feat = self.cnn(images) if self.cnn is not None else None
        txt_feat = self.txt(text) if self.txt is not None else None
        return self.head(img_feat, txt_feat)

    def trainable_parameters(self):
        """Only the parameters that actually require gradients (backbone may be frozen)."""
        return [p for p in self.parameters() if p.requires_grad]
