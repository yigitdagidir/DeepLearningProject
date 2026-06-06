"""
The PyTorch `Dataset` that feeds the model, plus the helper that builds the
train / val / test dataloaders.

Both data sources (the synthetic demo and the real Fashion-Gen) are funnelled
into the same `MultiModalDataset`, so the model and the training loop never need
to know where the data came from.
"""

import os
import json
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from .text_utils import Vocabulary
from . import demo_dataset


# ImageNet statistics – required because we use ImageNet-pretrained CNNs.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool):
    """Standard pre-processing: resize, (light) augmentation, normalise."""
    if train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MultiModalDataset(Dataset):
    """
    Holds (image, caption, label) triples.

    Images can be given either as file paths (demo set, saves RAM) or as
    in-memory numpy arrays (real Fashion-Gen).  Captions are encoded to fixed
    length integer id tensors using the shared Vocabulary.
    """

    def __init__(self,
                 images,                       # list of paths (str) OR numpy arrays
                 captions: List[str],
                 labels: List[int],
                 vocab: Vocabulary,
                 max_text_len: int,
                 transform,
                 image_root: str = ""):
        assert len(images) == len(captions) == len(labels)
        self.images = images
        self.captions = captions
        self.labels = labels
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.labels)

    def _load_image(self, item) -> Image.Image:
        if isinstance(item, str):
            return Image.open(os.path.join(self.image_root, item)).convert("RGB")
        # numpy array (Fashion-Gen)
        return Image.fromarray(item).convert("RGB")

    def __getitem__(self, idx):
        img = self.transform(self._load_image(self.images[idx]))
        text = torch.tensor(self.vocab.encode(self.captions[idx], self.max_text_len),
                            dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, text, label


# --------------------------------------------------------------------------- #
#  Loading raw records from the two possible sources                          #
# --------------------------------------------------------------------------- #
def _load_demo(cfg):
    """Generate the demo set if needed, then read its metadata."""
    meta_path = os.path.join(cfg.data_root, "demo", "demo_metadata.json")
    if not os.path.exists(meta_path):
        demo_dataset.generate(root=os.path.join(cfg.data_root, "demo"),
                              image_size=cfg.image_size, seed=cfg.seed)
    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]
    root = os.path.join(cfg.data_root, "demo")

    def subset(split):
        recs = [r for r in meta["records"] if r["split"] == split]
        imgs = [r["image"] for r in recs]
        caps = [r["caption"] for r in recs]
        labs = [r["label"] for r in recs]
        return imgs, caps, labs

    train = subset("train")
    test = subset("test")
    return train, test, classes, root


def _load_fashiongen(cfg):
    from .fashiongen import load_fashiongen
    train_h5 = os.path.join(cfg.data_root, "fashiongen_256_256_train.h5")
    val_h5 = os.path.join(cfg.data_root, "fashiongen_256_256_validation.h5")
    tr_imgs, tr_caps, tr_labs, classes = load_fashiongen(train_h5)
    te_imgs, te_caps, te_labs, _ = load_fashiongen(val_h5)
    return (tr_imgs, tr_caps, tr_labs), (te_imgs, te_caps, te_labs), classes, ""


# --------------------------------------------------------------------------- #
#  Public entry point                                                         #
# --------------------------------------------------------------------------- #
def build_dataloaders(cfg, return_raw: bool = False):
    """
    Build train / val / test dataloaders + the vocabulary + class names.

    The vocabulary is built ONLY from the training captions (standard practice –
    we must not peek at validation/test text).
    """
    if cfg.dataset == "demo":
        (tr_imgs, tr_caps, tr_labs), (te_imgs, te_caps, te_labs), classes, root = _load_demo(cfg)
    elif cfg.dataset == "fashiongen":
        (tr_imgs, tr_caps, tr_labs), (te_imgs, te_caps, te_labs), classes, root = _load_fashiongen(cfg)
    else:
        raise ValueError(f"unknown dataset '{cfg.dataset}'")

    # ---- train / val split (stratify-free random split is fine here) ----
    rng = np.random.default_rng(cfg.seed)
    n = len(tr_labs)
    perm = rng.permutation(n)
    n_val = int(round(cfg.val_split * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def take(lst, idx):
        return [lst[i] for i in idx]

    # ---- vocabulary from TRAIN captions only ----
    vocab = Vocabulary().build(take(tr_caps, train_idx), min_freq=cfg.min_word_freq)

    tf_train = build_transforms(cfg.image_size, train=True)
    tf_eval = build_transforms(cfg.image_size, train=False)

    train_ds = MultiModalDataset(take(tr_imgs, train_idx), take(tr_caps, train_idx),
                                 take(tr_labs, train_idx), vocab, cfg.max_text_len,
                                 tf_train, image_root=root)
    val_ds = MultiModalDataset(take(tr_imgs, val_idx), take(tr_caps, val_idx),
                               take(tr_labs, val_idx), vocab, cfg.max_text_len,
                               tf_eval, image_root=root)
    test_ds = MultiModalDataset(te_imgs, te_caps, te_labs, vocab, cfg.max_text_len,
                                tf_eval, image_root=root)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=False),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers),
    }
    info = {"vocab": vocab, "classes": classes, "num_classes": len(classes),
            "vocab_size": len(vocab), "pad_idx": vocab.pad_idx}
    if return_raw:
        info["raw_train"] = (take(tr_imgs, train_idx), take(tr_caps, train_idx),
                             take(tr_labs, train_idx))
        info["root"] = root
    return loaders, info
