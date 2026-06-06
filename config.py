"""
Central place for all the settings we use across the project.

We keep everything in one dataclass so that the training script, the
cross-validation script and the hyper-parameter search all read the same
default values.  Anything here can be overridden from the command line
(see train.py) so we don't have to touch the code to run a new experiment.
"""

from dataclasses import dataclass, field, asdict
from typing import List
import json


@dataclass
class Config:
    # ----- data -----
    dataset: str = "demo"          # "demo"  or  "fashiongen"
    data_root: str = "data"        # where the demo data / real .h5 files live
    image_size: int = 64           # 64 for the demo, 224 for real Fashion-Gen
    max_text_len: int = 20         # captions are cut / padded to this many tokens
    min_word_freq: int = 1         # words rarer than this are mapped to <unk>
    val_split: float = 0.15        # fraction of training data used for validation
    num_workers: int = 0           # 0 is the safe default on Windows

    # ----- image branch (CNN) -----
    cnn_backbone: str = "resnet18"     # resnet18 / resnet34 / efficientnet_b0
    pretrained: bool = True            # use ImageNet weights (transfer learning)
    freeze_backbone: bool = True       # only fine-tune the top layers (faster, less overfit)
    img_feat_dim: int = 256            # CNN features are projected down to this size

    # ----- text branch (RNN) -----
    embed_dim: int = 128               # word embedding size
    rnn_type: str = "lstm"             # "lstm" or "gru"
    rnn_hidden: int = 256              # hidden size of the RNN
    rnn_layers: int = 1
    bidirectional: bool = True
    use_pretrained_embeddings: bool = False   # set True + give glove_path to load GloVe
    glove_path: str = ""

    # ----- fusion head -----
    fusion_hidden: List[int] = field(default_factory=lambda: [256])  # FC layers after concat
    dropout: float = 0.5
    modality: str = "both"             # "image", "text" or "both"  (used for the ablation)

    # ----- optimisation -----
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"            # "adam" or "sgd"
    lr_step: int = 7                   # StepLR: divide lr by 10 every lr_step epochs
    lr_gamma: float = 0.1
    label_smoothing: float = 0.0

    # ----- misc -----
    seed: int = 42
    out_dir: str = "outputs"
    experiment_name: str = "run"

    # convenience ----------------------------------------------------------
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "Config":
        with open(path) as f:
            data = json.load(f)
        return Config(**data)
