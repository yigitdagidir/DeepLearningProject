"""Assemble the 3-way comparison: image-only vs text-only vs fusion.

Usage:
    python -m src.compare

Reads each variant's ``artifacts/<model>/test_metrics.json`` (written by
``src.evaluate``) and produces the **core deliverable** of the whole project:

    * ``artifacts/comparison.csv``  - tidy table of test metrics
    * ``artifacts/comparison.md``   - the same table in Markdown (drop into the report)
    * ``artifacts/comparison.png``  - grouped bar chart (accuracy & macro-F1)

The headline result the report/presentation lead with — does multi-modal fusion
beat either modality alone? — is computed and printed here.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

METRIC_KEYS = ["test_accuracy", "macro_f1", "macro_precision", "macro_recall"]
PRETTY = {"image": "Image-only", "text": "Text-only", "fusion": "Fusion (multi-modal)"}


def _load_metrics(model: str) -> dict | None:
    path = config.ARTIFACTS_DIR / model / "test_metrics.json"
    if not path.exists():
        print(f"[compare] WARNING: {path} missing — run `python -m src.evaluate "
              f"--model {model}` first.")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_table() -> pd.DataFrame:
    """Collect the three variants' test metrics into a tidy DataFrame."""
    rows = []
    for model in config.MODEL_VARIANTS:
        m = _load_metrics(model)
        if m is None:
            continue
        rows.append({
            "model": PRETTY.get(model, model),
            "accuracy": m["test_accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
        })
    if not rows:
        raise RuntimeError("No test_metrics.json found. Train+evaluate all models first.")
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Render the comparison DataFrame as a Markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [
            row[c] if isinstance(row[c], str) else f"{row[c]:.4f}" for c in cols
        ]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def plot_comparison(df: pd.DataFrame, out_path) -> None:
    """Grouped bar chart of accuracy and macro-F1 across the three models."""
    labels = df["model"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, df["macro_f1"], width, label="Macro-F1")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Single-modal baselines vs multi-modal fusion (test set)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend()
    for i, (a, f) in enumerate(zip(df["accuracy"], df["macro_f1"])):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compare() -> pd.DataFrame:
    """Build, persist, and print the 3-way comparison; return the table."""
    config.ensure_dirs()
    df = build_table()

    csv_path = config.ARTIFACTS_DIR / "comparison.csv"
    md_path = config.ARTIFACTS_DIR / "comparison.md"
    png_path = config.ARTIFACTS_DIR / "comparison.png"

    df.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(df), encoding="utf-8")
    plot_comparison(df, png_path)

    print("\n=== 3-way comparison (test set) ===")
    print(df.to_string(index=False))

    # Headline verdict.
    if {"Image-only", "Text-only", "Fusion (multi-modal)"}.issubset(set(df["model"])):
        fus = df.set_index("model").loc["Fusion (multi-modal)", "macro_f1"]
        best_single = df[df["model"] != "Fusion (multi-modal)"]["macro_f1"].max()
        delta = fus - best_single
        verdict = "BEATS" if delta > 0 else "does NOT beat"
        print(f"\n[compare] Fusion macro-F1 {verdict} the best single modality "
              f"by {delta:+.4f}.")
    print(f"[compare] Wrote {csv_path.name}, {md_path.name}, {png_path.name} to artifacts/.")
    return df


if __name__ == "__main__":
    compare()
