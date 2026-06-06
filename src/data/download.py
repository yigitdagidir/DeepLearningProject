"""Download the Kaggle *Fashion Product Images (small)* dataset.

Usage:
    python -m src.data.download

Uses ``kagglehub`` (no manual kaggle.json juggling on Colab) and is **idempotent**:
kagglehub caches the dataset, and we additionally record the resolved dataset root
in ``data/dataset_path.txt`` so the rest of the pipeline never has to re-resolve the
cache. The dataset ships ``styles.csv`` plus an ``images/`` folder; depending on the
mirror these may sit under a nested subfolder, so we locate ``styles.csv`` by search
and derive the images directory relative to it.
"""

from __future__ import annotations

from pathlib import Path

from src import config


def _find_styles_csv(root: Path) -> Path:
    """Locate styles.csv anywhere under ``root`` (handles nested mirror layouts)."""
    # Prefer a top-level styles.csv; otherwise take the shallowest match.
    candidates = sorted(root.rglob(config.STYLES_CSV_NAME), key=lambda p: len(p.parts))
    if not candidates:
        raise FileNotFoundError(f"Could not find {config.STYLES_CSV_NAME} under {root}")
    return candidates[0]


def _find_images_dir(styles_csv: Path) -> Path:
    """Find the images directory that pairs with ``styles_csv``."""
    sibling = styles_csv.parent / config.IMAGES_DIRNAME
    if sibling.is_dir():
        return sibling
    # Fall back to a recursive search for an 'images' folder near the csv.
    for cand in styles_csv.parent.rglob(config.IMAGES_DIRNAME):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find an '{config.IMAGES_DIRNAME}' directory near {styles_csv}"
    )


def resolve_dataset_paths() -> tuple[Path, Path]:
    """Return ``(styles_csv, images_dir)`` for the downloaded dataset.

    Reads the recorded root from ``data/dataset_path.txt`` when available;
    otherwise raises with a hint to run the download first.
    """
    if config.DATASET_PATH_FILE.exists():
        root = Path(config.DATASET_PATH_FILE.read_text(encoding="utf-8").strip())
        if root.exists():
            styles = _find_styles_csv(root)
            return styles, _find_images_dir(styles)
    raise FileNotFoundError(
        "Dataset path not recorded. Run `python -m src.data.download` first."
    )


def download() -> tuple[Path, Path]:
    """Download (or reuse cached) dataset and record its resolved root.

    Returns ``(styles_csv, images_dir)``.
    """
    config.ensure_dirs()

    # Fast path: already recorded and present.
    if config.DATASET_PATH_FILE.exists():
        try:
            styles, images = resolve_dataset_paths()
            print(f"[download] Already present. styles.csv={styles}")
            print(f"[download] images dir = {images}")
            return styles, images
        except FileNotFoundError:
            pass  # recorded path went stale — re-download below.

    import kagglehub  # imported here so `import src.data.download` works without it.

    print(f"[download] Fetching '{config.KAGGLE_DATASET}' via kagglehub ...")
    root = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    print(f"[download] kagglehub cache root: {root}")

    styles = _find_styles_csv(root)
    images = _find_images_dir(styles)

    # Record the *root* (parent of styles.csv) for fast resolution later.
    config.DATASET_PATH_FILE.write_text(str(styles.parent), encoding="utf-8")

    print(f"[download] styles.csv = {styles}")
    print(f"[download] images dir = {images}")
    n_images = sum(1 for _ in images.glob('*.jpg'))
    print(f"[download] {n_images} jpg images found.")
    return styles, images


if __name__ == "__main__":
    config.set_seeds()
    download()
