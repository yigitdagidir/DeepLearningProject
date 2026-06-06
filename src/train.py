"""
Main training script.

Examples
--------
# quick demo run (generates the synthetic data automatically, ~1-2 min on CPU):
    python -m src.train --dataset demo --epochs 15

# image-only / text-only baselines (for the fusion ablation):
    python -m src.train --modality image --experiment_name image_only
    python -m src.train --modality text  --experiment_name text_only

# real Fashion-Gen (after putting the .h5 files in data/fashiongen/):
    python -m src.train --dataset fashiongen --data_root data/fashiongen \
                        --image_size 224 --freeze_backbone --epochs 20

Everything it produces (config, best weights, training curves, confusion matrix,
metrics.json, classification report) is written to outputs/<experiment_name>/.
"""

import os
import json
import argparse
import dataclasses

import torch
import torch.nn as nn

from .config import Config
from .utils import (set_seed, get_device, ensure_dir, plot_curves,
                    plot_confusion, text_report)
from .data.dataset import build_dataloaders
from .models.multimodal import MultiModalNet
from .engine import fit, evaluate


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Train the multi-modal classifier.")
    # we expose every field of the Config dataclass as a CLI flag
    for f in dataclasses.fields(Config):
        if f.type == bool or f.name in ("pretrained", "freeze_backbone",
                                        "bidirectional", "use_pretrained_embeddings"):
            p.add_argument(f"--{f.name}", action="store_true", default=None)
            p.add_argument(f"--no_{f.name}", dest=f.name, action="store_false")
        elif f.name == "fusion_hidden":
            p.add_argument("--fusion_hidden", type=int, nargs="+", default=None)
        else:
            p.add_argument(f"--{f.name}", type=type(getattr(cfg, f.name)), default=None)
    args = p.parse_args()
    # override defaults only where the user actually passed something
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    out_dir = ensure_dir(os.path.join(cfg.out_dir, cfg.experiment_name))
    print(f"device = {device} | experiment = {cfg.experiment_name} | modality = {cfg.modality}")

    # ---- data ----
    loaders, info = build_dataloaders(cfg)
    print(f"classes ({info['num_classes']}): {info['classes']}")
    print(f"vocab size: {info['vocab_size']}")

    # ---- model ----
    model = MultiModalNet(cfg, info["vocab_size"], info["pad_idx"],
                          info["num_classes"]).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable parameters: {n_train:,}")

    # ---- train ----
    history, best_val = fit(model, loaders, cfg, device)

    # ---- test ----
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, loaders["test"],
                                                   criterion, device, return_preds=True)
    print(f"\nTEST: loss {test_loss:.4f} | accuracy {test_acc:.4f}")

    # ---- save everything ----
    cfg.to_json(os.path.join(out_dir, "config.json"))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    plot_curves(history, os.path.join(out_dir, "training_curves.png"))
    plot_confusion(y_true, y_pred, info["classes"],
                   os.path.join(out_dir, "confusion_matrix.png"))

    report = text_report(y_true, y_pred, info["classes"])
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)

    metrics = {
        "experiment": cfg.experiment_name,
        "modality": cfg.modality,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "num_classes": info["num_classes"],
        "history": history,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAll artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
