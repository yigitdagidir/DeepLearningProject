"""
Fusion head.

After the CNN gives us an image vector and the RNN gives us a text vector, we
combine ("fuse") them.  The simplest and most common strategy, and the one the
project asks for, is *concatenation*: we stick the two vectors next to each
other to form one long vector

        z = [ v_image ; v_text ]

and then pass it through a few fully-connected (dense) layers.  Those FC layers
are what actually let the network learn cross-modal interactions: a hidden unit
can fire only when, say, the image looks like a shoe AND the caption mentions
"sneakers".  A final linear layer produces one logit per class; softmax (applied
inside the loss) turns the logits into class probabilities.
"""

import torch
import torch.nn as nn
from typing import List


class FusionClassifier(nn.Module):
    def __init__(self,
                 img_dim: int,
                 txt_dim: int,
                 num_classes: int,
                 hidden_dims: List[int],
                 dropout: float = 0.5,
                 modality: str = "both"):
        super().__init__()
        self.modality = modality

        if modality == "both":
            in_dim = img_dim + txt_dim
        elif modality == "image":
            in_dim = img_dim
        elif modality == "text":
            in_dim = txt_dim
        else:
            raise ValueError(f"unknown modality '{modality}'")

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),       # regularisation to fight over-fitting
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))   # final logits
        self.mlp = nn.Sequential(*layers)

    def forward(self, img_feat, txt_feat):
        if self.modality == "both":
            z = torch.cat([img_feat, txt_feat], dim=1)
        elif self.modality == "image":
            z = img_feat
        else:
            z = txt_feat
        return self.mlp(z)                 # (B, num_classes) raw logits
