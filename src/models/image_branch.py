"""Image branch: frozen EfficientNetB0 backbone -> projection (Person A).

Exposes two things so the fusion model (Phase 3) reuses the *exact* same encoder
architecture as the image-only baseline (Phase 2):

* :func:`build_image_encoder` — wires ``Input -> EfficientNetB0(frozen) ->
  GlobalAveragePooling2D -> Dense(256, relu)`` and returns the 256-d encoding
  tensor (for fusion to concatenate).
* :func:`build_image_model` — wraps that encoder with a softmax head into a
  standalone image-only classifier.

Architecture (``CLAUDE.md`` §5):
    EfficientNetB0(weights="imagenet", include_top=False)  [FROZEN]
        -> GlobalAveragePooling2D -> Dense(256, relu)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

from src import config


def build_image_encoder(image_input: tf.Tensor, trainable_backbone: bool = False) -> tf.Tensor:
    """Build the image encoder subgraph and return its 256-d output tensor.

    Args:
        image_input: a Keras ``Input`` of shape ``IMAGE_SIZE + (3,)``.
        trainable_backbone: keep False for the frozen Stage-1 design; set True
            only for the optional Stage-2 fine-tuning (top blocks).
    """
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
    )
    # Freeze the whole backbone for Stage 1 (BatchNorm then runs in inference
    # mode, which is exactly what we want for a frozen feature extractor).
    backbone.trainable = trainable_backbone

    x = GlobalAveragePooling2D(name="image_gap")(backbone.output)
    x = Dense(config.IMAGE_DENSE_UNITS, activation="relu", name="image_encoding")(x)
    return x


def build_image_model(n_classes: int | None = None) -> Model:
    """Standalone image-only classifier (Phase 2 baseline)."""
    n_classes = n_classes or config.N_CLASSES
    image_input = Input(
        shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS), name="image"
    )
    encoding = build_image_encoder(image_input)
    outputs = Dense(n_classes, activation="softmax", name="predictions")(encoding)
    return Model(inputs=image_input, outputs=outputs, name="image_only")


if __name__ == "__main__":
    model = build_image_model()
    model.summary()
