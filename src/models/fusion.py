"""Fusion model: concatenate the image (256-d) and text (128-d) encodings (Person C).

This is the multi-modal model whose whole point is to beat both single-modal
baselines. It **reuses the exact encoder builders** from the image and text
branches, so the only thing that differs from the baselines is the fusion head —
making the 3-way comparison a fair, apples-to-apples test.

Architecture (``CLAUDE.md`` §5):
    [image_encoder(256) , text_encoder(128)]
        -> Concatenate -> Dense(256, relu) -> Dropout(0.3) -> Dense(n, softmax)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Input,
    TextVectorization,
)

from src import config
from src.models.image_branch import build_image_encoder
from src.models.text_branch import build_text_encoder


def build_fusion_model(
    vectorizer: TextVectorization,
    n_classes: int | None = None,
    fusion_dense_units: int | None = None,
    dropout: float | None = None,
    trainable_backbone: bool = False,
) -> Model:
    """Build the multi-modal fusion classifier.

    Args:
        vectorizer: the shared, train-adapted ``TextVectorization`` layer.
        n_classes: number of output classes (defaults to ``config.N_CLASSES``).
        fusion_dense_units: width of the post-concat FC layer (HP-tunable).
        dropout: dropout rate before the softmax (HP-tunable).
        trainable_backbone: forwarded to the image encoder (Stage-2 only).
    """
    n_classes = n_classes or config.N_CLASSES
    fusion_dense_units = fusion_dense_units or config.FUSION_DENSE_UNITS
    dropout = config.DROPOUT if dropout is None else dropout

    image_input = Input(
        shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS), name="image"
    )
    text_input = Input(shape=(), dtype=tf.string, name="text")

    image_encoding = build_image_encoder(image_input, trainable_backbone)  # (b, 256)
    text_encoding = build_text_encoder(text_input, vectorizer)             # (b, 128)

    x = Concatenate(name="fusion_concat")([image_encoding, text_encoding])  # (b, 384)
    x = Dense(fusion_dense_units, activation="relu", name="fusion_dense")(x)
    x = Dropout(dropout, name="fusion_dropout")(x)
    outputs = Dense(n_classes, activation="softmax", name="predictions")(x)

    return Model(inputs=[image_input, text_input], outputs=outputs, name="fusion")


if __name__ == "__main__":
    from src.data.dataset import get_text_vectorizer

    config.set_seeds()
    model = build_fusion_model(get_text_vectorizer())
    model.summary()
