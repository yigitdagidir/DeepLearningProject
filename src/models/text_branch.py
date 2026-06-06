"""Text branch: TextVectorization -> Embedding -> GRU -> projection (Person B).

Mirrors :mod:`src.models.image_branch`: a reusable encoder builder plus a
standalone baseline model, so the fusion model reuses the identical text encoder.

Architecture (``CLAUDE.md`` §5):
    TextVectorization -> Embedding -> GRU(128) -> Dense(128, relu)

The ``TextVectorization`` layer is **passed in** (shared, adapted on train only),
so the text-only baseline and the fusion model tokenise text identically — no
vocabulary mismatch, no leakage. The GRU returns only its final hidden state,
which is the text encoding used downstream (the brief requires "use the final RNN
state as the text encoding").
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, GRU, Input, TextVectorization

from src import config


def build_text_encoder(
    text_input: tf.Tensor, vectorizer: TextVectorization
) -> tf.Tensor:
    """Build the text encoder subgraph and return its 128-d output tensor.

    Args:
        text_input: a Keras ``Input(shape=(), dtype=tf.string)`` of raw strings.
        vectorizer: the shared, train-adapted ``TextVectorization`` layer.
    """
    x = vectorizer(text_input)                       # (batch, SEQUENCE_LENGTH) ints
    x = Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.EMBEDDING_DIM,
        mask_zero=True,                              # let the GRU skip padding
        name="text_embedding",
    )(x)
    x = GRU(config.GRU_UNITS, name="text_gru")(x)     # final hidden state only
    x = Dense(config.TEXT_DENSE_UNITS, activation="relu", name="text_encoding")(x)
    return x


def build_text_model(
    vectorizer: TextVectorization, n_classes: int | None = None
) -> Model:
    """Standalone text-only classifier (Phase 2 baseline)."""
    n_classes = n_classes or config.N_CLASSES
    text_input = Input(shape=(), dtype=tf.string, name="text")
    encoding = build_text_encoder(text_input, vectorizer)
    outputs = Dense(n_classes, activation="softmax", name="predictions")(encoding)
    return Model(inputs=text_input, outputs=outputs, name="text_only")


if __name__ == "__main__":
    from src.data.dataset import get_text_vectorizer

    config.set_seeds()
    model = build_text_model(get_text_vectorizer())
    model.summary()
