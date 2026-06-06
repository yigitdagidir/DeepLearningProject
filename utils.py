"""
Small helper functions that are used all over the place:
seeding, picking the device, an averaging meter and the plotting helpers
for the confusion matrix and the training curves.
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # we never show windows, we only save .png files
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def set_seed(seed: int = 42) -> None:
    """Make a run reproducible (as much as PyTorch allows on CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """Keeps a running average of a value (loss / accuracy) during an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def accuracy(logits, targets) -> float:
    """Top-1 accuracy for a batch."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
#  Plotting helpers                                                           #
# --------------------------------------------------------------------------- #
def plot_curves(history: dict, out_path: str) -> None:
    """Plot train/val loss and accuracy over the epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "s-", label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="train")
    ax2.plot(epochs, history["val_acc"], "s-", label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, out_path: str) -> None:
    """Save a confusion-matrix heat-map."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    # write the counts inside the cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def text_report(y_true, y_pred, class_names) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, digits=4,
                                 zero_division=0)
