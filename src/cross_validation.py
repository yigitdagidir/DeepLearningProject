"""
K-fold cross-validation.

The project asks us to "evaluate model performance with cross-validation to
ensure generalisation".  We use stratified K-fold (so every fold has the same
class balance) over the training set.  For each fold we rebuild the vocabulary
from that fold's training captions only, train a fresh model, and evaluate on
the held-out fold.  We report the mean and standard deviation of the accuracy.

Run:
    python -m src.cross_validation --dataset demo --folds 5 --epochs 8
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import Config
from .utils import set_seed, get_device
from .data.dataset import build_dataloaders, MultiModalDataset, build_transforms
from .data.text_utils import Vocabulary
from .models.multimodal import MultiModalNet
from .engine import fit, evaluate


def run_cv(cfg: Config, folds: int = 5):
    set_seed(cfg.seed)
    device = get_device()

    # grab the raw (un-split) training data once
    _, info = build_dataloaders(cfg, return_raw=True)
    images, captions, labels = info["raw_train"]
    root = info["root"]
    classes = info["classes"]
    labels_arr = np.array(labels)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
    fold_acc = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels_arr), 1):
        # build vocab from this fold's training captions only
        tr_caps = [captions[i] for i in tr_idx]
        vocab = Vocabulary().build(tr_caps, min_freq=cfg.min_word_freq)

        tf_tr = build_transforms(cfg.image_size, train=True)
        tf_ev = build_transforms(cfg.image_size, train=False)

        def make_ds(idx, tf):
            return MultiModalDataset([images[i] for i in idx],
                                     [captions[i] for i in idx],
                                     [labels[i] for i in idx],
                                     vocab, cfg.max_text_len, tf, image_root=root)

        loaders = {
            "train": DataLoader(make_ds(tr_idx, tf_tr), batch_size=cfg.batch_size,
                                shuffle=True, num_workers=cfg.num_workers),
            "val": DataLoader(make_ds(va_idx, tf_ev), batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers),
        }

        model = MultiModalNet(cfg, len(vocab), vocab.pad_idx, len(classes)).to(device)
        fit(model, loaders, cfg, device, verbose=False)

        criterion = nn.CrossEntropyLoss()
        _, acc = evaluate(model, loaders["val"], criterion, device)
        fold_acc.append(acc)
        print(f"fold {fold}/{folds}: val accuracy = {acc:.4f}")

    fold_acc = np.array(fold_acc)
    print("\n========== cross-validation summary ==========")
    print(f"accuracy per fold : {np.round(fold_acc, 4).tolist()}")
    print(f"mean accuracy     : {fold_acc.mean():.4f}")
    print(f"std  accuracy     : {fold_acc.std():.4f}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, "cross_validation.json")
    with open(out_path, "w") as f:
        json.dump({"folds": folds, "modality": cfg.modality,
                   "fold_acc": fold_acc.tolist(),
                   "mean": float(fold_acc.mean()),
                   "std": float(fold_acc.std())}, f, indent=2)
    print(f"saved CV results to {out_path}")
    return fold_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="demo")
    p.add_argument("--data_root", default="data")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--modality", default="both")
    p.add_argument("--image_size", type=int, default=64)
    args = p.parse_args()

    cfg = Config(dataset=args.dataset, data_root=args.data_root, epochs=args.epochs,
                 modality=args.modality, image_size=args.image_size)
    run_cv(cfg, folds=args.folds)


if __name__ == "__main__":
    main()
