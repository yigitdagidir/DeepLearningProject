"""Evaluate a trained model on the held-out test split.

Usage:
    python -m src.evaluate --model image
    python -m src.evaluate --model text
    python -m src.evaluate --model fusion

Loads ``artifacts/<run-name>/model.keras``, runs it over the test set, and writes:
    * ``test_metrics.json``    - accuracy, macro precision/recall/F1, per-class report
    * ``confusion_matrix.png`` - normalised confusion-matrix heatmap

These ``test_metrics.json`` files are what ``src.compare`` reads to build the
non-negotiable 3-way comparison.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src import config
from src.data.dataset import make_dataset
from src.train import adapt_for_model


def _collect_labels(ds: tf.data.Dataset) -> np.ndarray:
    """Materialise the integer labels from a ``(x, y)`` dataset, in order."""
    return np.concatenate([y.numpy() for _, y in ds], axis=0)


def plot_confusion_matrix(cm: np.ndarray, class_list: list[str], out_path) -> None:
    """Save a row-normalised confusion-matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_list, yticklabels=class_list, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate(model_variant: str, run_name: str | None = None) -> dict:
    """Evaluate a trained variant on test and persist metrics + confusion matrix."""
    config.set_seeds()
    run_name = run_name or model_variant
    out_dir = config.artifacts_dir_for(run_name)
    model_path = out_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Train it first: python -m src.train --model {model_variant}"
        )

    class_list = config.load_class_list()
    test_ds = adapt_for_model(
        make_dataset(config.TEST_MANIFEST, training=False), model_variant
    )

    model = tf.keras.models.load_model(model_path)
    y_true = _collect_labels(test_ds)
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=class_list, output_dict=True, zero_division=0
    )

    metrics = {
        "model": model_variant,
        "run_name": run_name,
        "test_accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "n_test": int(len(y_true)),
        "per_class": report,
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_list))))
    plot_confusion_matrix(cm, class_list, out_dir / "confusion_matrix.png")

    print(f"[evaluate] {model_variant}: acc={acc:.4f} macro-F1={f1:.4f} "
          f"-> {out_dir/'test_metrics.json'}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained variant on test.")
    p.add_argument("--model", required=True, choices=config.MODEL_VARIANTS)
    p.add_argument("--run-name", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.run_name)
