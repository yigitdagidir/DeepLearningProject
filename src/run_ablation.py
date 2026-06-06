"""
Fusion ablation study.

Trains three models with identical settings, changing only the modality:
    1. image-only   (CNN + classifier)
    2. text-only    (RNN + classifier)
    3. fusion        (CNN + RNN + classifier)

and writes a small bar chart + json comparing their test accuracy.  This is the
experiment that answers the project question "what is the impact of multi-modal
fusion on performance?".

Run:
    python -m src.run_ablation --epochs 15
"""

import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .config import Config
from .utils import set_seed, get_device, ensure_dir
from .data.dataset import build_dataloaders
from .models.multimodal import MultiModalNet
from .engine import fit, evaluate


def train_one(modality: str, base_kwargs: dict):
    cfg = Config(modality=modality, **base_kwargs)
    set_seed(cfg.seed)
    device = get_device()
    loaders, info = build_dataloaders(cfg)
    model = MultiModalNet(cfg, info["vocab_size"], info["pad_idx"],
                          info["num_classes"]).to(device)
    history, best_val = fit(model, loaders, cfg, device, verbose=False)
    crit = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, loaders["test"], crit, device)
    print(f"[{modality:>5}] best_val={best_val:.4f}  test_acc={test_acc:.4f}")
    return {"modality": modality, "val_acc": best_val, "test_acc": test_acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="demo")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--out_dir", default="outputs/ablation")
    args = p.parse_args()

    base_kwargs = dict(dataset=args.dataset, epochs=args.epochs,
                       image_size=args.image_size)
    out_dir = ensure_dir(args.out_dir)

    results = [train_one(m, base_kwargs) for m in ["image", "text", "both"]]

    with open(os.path.join(out_dir, "ablation.json"), "w") as f:
        json.dump(results, f, indent=2)

    # bar chart
    names = {"image": "Image only\n(CNN)", "text": "Text only\n(RNN)",
             "both": "Fusion\n(CNN+RNN)"}
    labels = [names[r["modality"]] for r in results]
    accs = [r["test_acc"] for r in results]
    colors = ["#6fa8dc", "#93c47d", "#e06666"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, accs, color=colors)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Effect of multi-modal fusion")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}",
                ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_bar.png"), dpi=130)
    print(f"\nsaved ablation chart to {out_dir}/ablation_bar.png")


if __name__ == "__main__":
    main()
