"""``tf.data`` input pipeline yielding ``((image, text), label)`` and the shared
``TextVectorization`` layer.

Design choices that matter for the report:

* **Raw strings flow through the pipeline.** ``TextVectorization`` is the *first
  layer of the text model* (per ``CLAUDE.md`` §5), so the dataset yields the raw
  ``productDisplayName`` string and tokenisation happens inside the graph. This
  guarantees the image-only, text-only and fusion models all see the *identical*
  vectoriser — an apples-to-apples comparison with no leakage.
* **The vectoriser is adapted on the TRAIN split only** and its vocabulary is
  persisted to ``artifacts/text_vocab.txt`` so every model/run reuses it.
* **EfficientNetB0 wants pixels in [0, 255].** ``efficientnet.preprocess_input``
  is a pass-through for B0 (normalisation lives inside the model); we still call
  it so the contract is explicit and correct if the backbone is ever swapped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization

from src import config

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def load_manifest(path: Path) -> pd.DataFrame:
    """Load a split manifest written by preprocess.py."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.preprocess` first."
        )
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Image decoding
# --------------------------------------------------------------------------- #
def decode_image(path: tf.Tensor) -> tf.Tensor:
    """Read, decode, resize to the CNN input size, and EfficientNet-preprocess."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=config.IMAGE_CHANNELS)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    # Pass-through for EfficientNetB0 (kept explicit on purpose).
    img = efficientnet.preprocess_input(img)
    return img


# --------------------------------------------------------------------------- #
# tf.data pipeline
# --------------------------------------------------------------------------- #
def make_dataset(
    manifest_path: Path,
    training: bool = False,
    batch_size: int = config.BATCH_SIZE,
    shuffle_buffer: int = 4096,
) -> tf.data.Dataset:
    """Build a batched, prefetched dataset of ``((image, text), label)``.

    Args:
        manifest_path: CSV split manifest (train/val/test).
        training: if True, shuffle each epoch (val/test stay ordered).
        batch_size: batch size.
        shuffle_buffer: shuffle buffer size for the training split.
    """
    df = load_manifest(manifest_path)
    paths = df["image_path"].astype(str).to_numpy()
    texts = df[config.TEXT_COLUMN].astype(str).to_numpy()
    labels = df["label"].astype("int32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, texts, labels))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(df)), seed=config.SEED,
                        reshuffle_each_iteration=True)

    def _map(path, text, label):
        return (decode_image(path), text), label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# --------------------------------------------------------------------------- #
# TextVectorization (shared, train-only adaptation, persisted)
# --------------------------------------------------------------------------- #
def _new_vectorizer() -> TextVectorization:
    return TextVectorization(
        max_tokens=config.VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=config.SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )


def build_text_vectorizer(persist: bool = True) -> TextVectorization:
    """Adapt a TextVectorization on the **train split only** and persist its vocab."""
    train_df = load_manifest(config.TRAIN_MANIFEST)
    texts = train_df[config.TEXT_COLUMN].astype(str).to_numpy()
    vectorizer = _new_vectorizer()
    vectorizer.adapt(texts)

    if persist:
        config.ensure_dirs()
        vocab = vectorizer.get_vocabulary()  # includes "" (pad) and "[UNK]"
        config.VOCAB_FILE.write_text("\n".join(vocab), encoding="utf-8")
        print(f"[dataset] Adapted vectoriser on train; vocab ({len(vocab)}) "
              f"-> {config.VOCAB_FILE}")
    return vectorizer


def load_text_vectorizer() -> TextVectorization:
    """Load the persisted vectoriser so every model shares the same vocabulary."""
    if not config.VOCAB_FILE.exists():
        raise FileNotFoundError(
            f"{config.VOCAB_FILE} not found. Build it via build_text_vectorizer()."
        )
    vocab = config.VOCAB_FILE.read_text(encoding="utf-8").split("\n")
    # set_vocabulary() must NOT receive the reserved mask/OOV tokens.
    vocab = [t for t in vocab if t not in ("", "[UNK]")]
    vectorizer = _new_vectorizer()
    vectorizer.set_vocabulary(vocab)
    return vectorizer


def get_text_vectorizer() -> TextVectorization:
    """Return the shared vectoriser: load if persisted, else build (and persist)."""
    if config.VOCAB_FILE.exists():
        return load_text_vectorizer()
    return build_text_vectorizer(persist=True)


if __name__ == "__main__":
    # Sanity check: build the vectoriser and print one batch's shapes/dtypes.
    config.set_seeds()
    vec = get_text_vectorizer()
    print("[dataset] vocab size:", len(vec.get_vocabulary()))
    train_ds = make_dataset(config.TRAIN_MANIFEST, training=True)
    (img, txt), y = next(iter(train_ds))
    print("[dataset] image batch:", img.shape, img.dtype)
    print("[dataset] text  batch:", txt.shape, txt.dtype, "(raw strings)")
    print("[dataset] label batch:", y.shape, y.dtype)
    print("[dataset] example text:", txt[0].numpy().decode("utf-8"))
    print("[dataset] vectorised  :", vec(txt[:1]).numpy()[0])
