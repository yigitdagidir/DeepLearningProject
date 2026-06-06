"""
Image branch of the model.

We take a CNN that was pre-trained on ImageNet (ResNet / EfficientNet), throw
away its 1000-class classification head, and use the remaining "backbone" as a
feature extractor.  This is classic transfer learning: the early conv layers
already know how to detect edges, textures and parts, which transfer well to
clothing images, so we only need to learn the last bit on top.

`freeze_backbone=True` keeps the pre-trained weights fixed and only trains the
small projection layer we add -> fast and works even with little data.
Setting it to False fine-tunes the whole network.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class CNNEncoder(nn.Module):
    def __init__(self,
                 backbone: str = "resnet18",
                 pretrained: bool = True,
                 freeze: bool = True,
                 out_dim: int = 256):
        super().__init__()
        self.backbone_name = backbone

        if backbone in ("resnet18", "resnet34", "resnet50"):
            weights = "DEFAULT" if pretrained else None
            net = getattr(tvm, backbone)(weights=weights)
            feat_dim = net.fc.in_features        # 512 for r18/r34, 2048 for r50
            net.fc = nn.Identity()               # drop the ImageNet classifier
            self.cnn = net
        elif backbone == "efficientnet_b0":
            weights = "DEFAULT" if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
            feat_dim = net.classifier[1].in_features   # 1280
            net.classifier = nn.Identity()
            self.cnn = net
        else:
            raise ValueError(f"unsupported backbone '{backbone}'")

        if freeze:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # small trainable head that maps the backbone features to `out_dim`
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, images):                  # images: (B, 3, H, W)
        feats = self.cnn(images)                # (B, feat_dim)
        return self.proj(feats)                 # (B, out_dim)
