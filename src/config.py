"""Central configuration: paths, hyperparameters, the class list, and the seed.

Every other module imports from here so there are **no magic numbers scattered
around** the codebase (a hard rule in ``CLAUDE.md`` §9). Keeping one source of
truth also makes the report's "experimental setup" section a direct copy of this
file.

The module is intentionally import-safe *without* TensorFlow installed: the only
``tensorflow`` use lives inside :func:`set_seeds`, where the import is lazy. That
lets ``python -c "import src.config"`` succeed on any machine (the Phase 0
Definition of Done), while real training happens on Colab where TF is present.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# ``config.py`` lives at ``<root>/src/config.py``; the project root is two up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

# Where the Kaggle dataset is unpacked (download.py records the resolved path
# here so preprocess/dataset don't have to re-resolve the kagglehub cache).
DATASET_PATH_FILE: Path = DATA_DIR / "dataset_path.txt"

# Split manifests (CSV) produced by preprocess.py.
SPLITS_DIR: Path = DATA_DIR / "splits"
TRAIN_MANIFEST: Path = SPLITS_DIR / "train.csv"
VAL_MANIFEST: Path = SPLITS_DIR / "val.csv"
TEST_MANIFEST: Path = SPLITS_DIR / "test.csv"

# Persisted class list (index -> subCategory name). Written by preprocess.py and
# read everywhere else so label encoding is identical across all models/runs.
CLASS_LIST_FILE: Path = ARTIFACTS_DIR / "class_list.json"

# Persisted TextVectorization vocabulary (adapted on TRAIN ONLY -> no leakage).
VOCAB_FILE: Path = ARTIFACTS_DIR / "text_vocab.txt"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
KAGGLE_DATASET: str = "paramaggarwal/fashion-product-images-small"
STYLES_CSV_NAME: str = "styles.csv"
IMAGES_DIRNAME: str = "images"

# Classification target and text field (see docs/PROJECT_BRIEF.md §2).
LABEL_COLUMN: str = "subCategory"
TEXT_COLUMN: str = "productDisplayName"
ID_COLUMN: str = "id"

# Keep the top-N most frequent subCategory classes. ~10 is the sweet spot:
# masterCategory is too easy (fusion contribution becomes invisible) and
# articleType has 140+ classes (too hard for the timeline).
N_CLASSES: int = 10

# Optional cap on the number of rows used while building the pipeline, to keep
# Colab fast. ``None`` uses every row of the top-N classes.
SUBSET_CAP: int | None = 20_000

# --------------------------------------------------------------------------- #
# Split (stratified, fixed seed) — 70 / 15 / 15
# --------------------------------------------------------------------------- #
SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# --------------------------------------------------------------------------- #
# Image branch
# --------------------------------------------------------------------------- #
# EfficientNetB0's native input is 224x224x3. Small-version images (~60x80) are
# upscaled to this; acceptable for a course project (documented as a limitation).
IMAGE_SIZE: tuple[int, int] = (224, 224)
IMAGE_CHANNELS: int = 3
IMAGE_DENSE_UNITS: int = 256  # projection after GlobalAveragePooling2D

# --------------------------------------------------------------------------- #
# Text branch
# --------------------------------------------------------------------------- #
VOCAB_SIZE: int = 10_000       # max tokens kept by TextVectorization
SEQUENCE_LENGTH: int = 20      # productDisplayName is short (~5-12 words)
EMBEDDING_DIM: int = 128
GRU_UNITS: int = 128
TEXT_DENSE_UNITS: int = 128    # projection after the GRU

# --------------------------------------------------------------------------- #
# Fusion head
# --------------------------------------------------------------------------- #
FUSION_DENSE_UNITS: int = 256
DROPOUT: float = 0.3

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE: int = 32
EPOCHS: int = 15               # upper bound; EarlyStopping usually stops sooner
LEARNING_RATE: float = 1e-3
EARLY_STOPPING_PATIENCE: int = 3

# Opt-in only: forcing fully deterministic ops can raise UnimplementedError mid-
# training if a GPU op lacks a deterministic kernel. Fixed seeds already give us
# practical reproducibility; flip this on only if you need bit-exact runs.
DETERMINISTIC_OPS: bool = False

# Small, manual hyperparameter grid for Phase 4 (no AutoML). Each entry is a
# partial override applied on top of the defaults above.
HP_GRID: list[dict] = [
    {"learning_rate": 1e-3, "dropout": 0.3, "fusion_dense_units": 256},
    {"learning_rate": 5e-4, "dropout": 0.3, "fusion_dense_units": 256},
    {"learning_rate": 1e-3, "dropout": 0.5, "fusion_dense_units": 256},
    {"learning_rate": 1e-3, "dropout": 0.3, "fusion_dense_units": 512},
]

# Recognised model variants (the three required by the brief).
MODEL_VARIANTS: tuple[str, ...] = ("image", "text", "fusion")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow for reproducibility.

    TensorFlow is imported lazily so this module stays importable on machines
    without TF (e.g. for ``import src.config`` smoke tests). Call this at the
    top of every entry point (download/preprocess/train/evaluate).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Opt-in only (see DETERMINISTIC_OPS): enabling this can make fit() raise
        # if a GPU op has no deterministic implementation.
        if DETERMINISTIC_OPS:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:  # pragma: no cover - older TF without this API
                pass
    except ImportError:
        # TF absent (local dev / config import test) — Python+NumPy seeded above.
        pass


def ensure_dirs() -> None:
    """Create the data/artifacts directory tree if missing (idempotent)."""
    for d in (DATA_DIR, ARTIFACTS_DIR, SPLITS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_class_list(class_list: list[str]) -> None:
    """Persist the ordered class list (index == integer label)."""
    ensure_dirs()
    CLASS_LIST_FILE.write_text(json.dumps(class_list, indent=2), encoding="utf-8")


def load_class_list() -> list[str]:
    """Load the persisted class list written by preprocess.py.

    Raises a clear error if preprocessing has not been run yet.
    """
    if not CLASS_LIST_FILE.exists():
        raise FileNotFoundError(
            f"{CLASS_LIST_FILE} not found. Run `python -m src.data.preprocess` first."
        )
    return json.loads(CLASS_LIST_FILE.read_text(encoding="utf-8"))


def artifacts_dir_for(model: str) -> Path:
    """Return (and create) the per-model artifacts directory, e.g. artifacts/fusion/."""
    d = ARTIFACTS_DIR / model
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    # Quick self-check: prints the resolved config so a teammate can confirm
    # paths/hyperparameters on their machine (Phase 0 sign-off).
    set_seeds()
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("DATA_DIR     :", DATA_DIR)
    print("ARTIFACTS_DIR:", ARTIFACTS_DIR)
    print("SEED         :", SEED)
    print("IMAGE_SIZE   :", IMAGE_SIZE)
    print("N_CLASSES    :", N_CLASSES)
    print("BATCH_SIZE   :", BATCH_SIZE, "| EPOCHS:", EPOCHS, "| LR:", LEARNING_RATE)
    print("config import + set_seeds() OK")
