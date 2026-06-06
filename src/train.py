"""Train one model variant: image-only, text-only, or fusion.

Usage:
    python -m src.train --model image
    python -m src.train --model text
    python -m src.train --model fusion

    # Phase-4 hyperparameter overrides (saved under a custom run name):
    python -m src.train --model fusion --lr 5e-4 --dropout 0.5 --run-name fusion_lr5e4

Each run saves, under ``artifacts/<run-name>/`` (default ``<model>``):
    * ``model.keras``        - the trained model (weights + architecture + vocab)
    * ``history.json``       - per-epoch train/val loss & accuracy
    * ``metrics.json``       - summary (best val metrics, epochs, hyperparameters)
    * ``training_curves.png``- loss & accuracy curves

Test-set metrics + confusion matrix are produced separately by ``src.evaluate``.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless (Colab / CI) — no display needed.
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config
from src.data.dataset import get_text_vectorizer, make_dataset


# --------------------------------------------------------------------------- #
# Dataset adaptation per variant
# --------------------------------------------------------------------------- #
def adapt_for_model(ds: tf.data.Dataset, model_variant: str) -> tf.data.Dataset:
    """Project the shared ``((image, text), label)`` dataset onto a variant's inputs."""
    if model_variant == "image":
        return ds.map(lambda inputs, y: (inputs[0], y), num_parallel_calls=tf.data.AUTOTUNE)
    if model_variant == "text":
        return ds.map(lambda inputs, y: (inputs[1], y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds  # fusion consumes (image, text) directly (input order matches)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_and_compile(
    model_variant: str,
    learning_rate: float,
    dropout: float,
    fusion_units: int,
    trainable_backbone: bool,
) -> tf.keras.Model:
    """Build the requested variant and compile it (Adam + sparse CE + accuracy)."""
    if model_variant == "image":
        from src.models.image_branch import build_image_model

        model = build_image_model()
    elif model_variant == "text":
        from src.models.text_branch import build_text_model

        model = build_text_model(get_text_vectorizer())
    elif model_variant == "fusion":
        from src.models.fusion import build_fusion_model

        model = build_fusion_model(
            get_text_vectorizer(),
            fusion_dense_units=fusion_units,
            dropout=dropout,
            trainable_backbone=trainable_backbone,
        )
    else:
        raise ValueError(f"Unknown model '{model_variant}'. Use one of {config.MODEL_VARIANTS}.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_curves(history: dict, out_path) -> None:
    """Save loss & accuracy training curves side by side."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], label="val")
    ax_loss.set_title("Loss"); ax_loss.set_xlabel("epoch"); ax_loss.legend()

    ax_acc.plot(history["accuracy"], label="train")
    ax_acc.plot(history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy"); ax_acc.set_xlabel("epoch"); ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(
    model_variant: str,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    dropout: float = config.DROPOUT,
    fusion_units: int = config.FUSION_DENSE_UNITS,
    trainable_backbone: bool = False,
    run_name: str | None = None,
) -> dict:
    """Train a variant end to end and persist model + metrics + curves."""
    config.set_seeds()
    run_name = run_name or model_variant
    out_dir = config.artifacts_dir_for(run_name)

    train_ds = adapt_for_model(make_dataset(config.TRAIN_MANIFEST, training=True,
                                            batch_size=batch_size), model_variant)
    val_ds = adapt_for_model(make_dataset(config.VAL_MANIFEST, training=False,
                                          batch_size=batch_size), model_variant)

    model = build_and_compile(model_variant, learning_rate, dropout,
                              fusion_units, trainable_backbone)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    print(f"[train] Training '{run_name}' for up to {epochs} epochs ...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Persist artifacts.
    model.save(out_dir / "model.keras")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    best_epoch = int(min(range(len(hist["val_loss"])), key=lambda i: hist["val_loss"][i]))
    summary = {
        "model": model_variant,
        "run_name": run_name,
        "epochs_trained": len(hist["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": hist["val_loss"][best_epoch],
        "best_val_accuracy": hist["val_accuracy"][best_epoch],
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "fusion_units": fusion_units,
            "trainable_backbone": trainable_backbone,
            "seed": config.SEED,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_curves(hist, out_dir / "training_curves.png")

    print(f"[train] Done. Best val acc={summary['best_val_accuracy']:.4f} "
          f"(epoch {best_epoch}). Artifacts -> {out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train image/text/fusion classifier.")
    p.add_argument("--model", required=True, choices=config.MODEL_VARIANTS)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--fusion-units", type=int, default=config.FUSION_DENSE_UNITS)
    p.add_argument("--trainable-backbone", action="store_true",
                   help="Stage-2 fine-tuning: unfreeze the EfficientNet backbone.")
    p.add_argument("--run-name", default=None,
                   help="Artifacts subdir name (default: the model name).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        fusion_units=args.fusion_units,
        trainable_backbone=args.trainable_backbone,
        run_name=args.run_name,
    )
