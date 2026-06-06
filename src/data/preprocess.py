"""Build reproducible, stratified train/val/test splits over the top-N classes.

Usage:
    python -m src.data.preprocess

Pipeline:
    1. Load ``styles.csv`` (skipping the handful of malformed rows it is known for).
    2. Drop rows with a missing id / label / text.
    3. Keep the ``N_CLASSES`` most frequent ``subCategory`` values.
    4. Attach ``images/{id}.jpg`` paths and drop rows whose image is missing.
    5. Integer-encode labels and persist the ordered ``CLASS_LIST``.
    6. Optionally cap the dataset size (stratified) for faster Colab iteration.
    7. Stratified **70/15/15** split with the fixed seed.
    8. Write ``train.csv`` / ``val.csv`` / ``test.csv`` manifests under ``data/splits``.

The per-step functions take/return DataFrames so they can be unit-tested with a
synthetic frame (no dataset download required) — see ``tests``/the exploration
notebook.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_styles(styles_csv: Path) -> pd.DataFrame:
    """Load styles.csv, skipping the malformed rows it is famous for.

    A few rows have unescaped commas in ``productDisplayName`` and break the
    naive parser; ``on_bad_lines='skip'`` drops only those rows.
    """
    df = pd.read_csv(styles_csv, on_bad_lines="skip")
    return df


def filter_top_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop incomplete rows and keep the top-N most frequent subCategory classes.

    Returns the filtered frame plus the ordered class list (most frequent first;
    list index == integer label).
    """
    needed = [config.ID_COLUMN, config.LABEL_COLUMN, config.TEXT_COLUMN]
    df = df.dropna(subset=needed).copy()
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).str.strip()
    df = df[df[config.TEXT_COLUMN].str.len() > 0]

    counts = df[config.LABEL_COLUMN].value_counts()
    class_list = counts.head(config.N_CLASSES).index.tolist()

    df = df[df[config.LABEL_COLUMN].isin(class_list)].copy()
    return df, class_list


def attach_image_paths(
    df: pd.DataFrame, images_dir: Path, verify: bool = True
) -> pd.DataFrame:
    """Add an ``image_path`` column and (optionally) drop rows with no image file."""
    df = df.copy()
    df["image_path"] = df[config.ID_COLUMN].apply(
        lambda i: str(images_dir / f"{int(i)}.jpg")
    )
    if verify:
        exists = df["image_path"].apply(lambda p: Path(p).is_file())
        dropped = int((~exists).sum())
        if dropped:
            print(f"[preprocess] Dropping {dropped} rows with missing image files.")
        df = df[exists].copy()
    return df


def encode_labels(df: pd.DataFrame, class_list: list[str]) -> pd.DataFrame:
    """Add an integer ``label`` column from the ordered class list."""
    index = {name: i for i, name in enumerate(class_list)}
    df = df.copy()
    df["label"] = df[config.LABEL_COLUMN].map(index).astype(int)
    return df


def cap_subset(df: pd.DataFrame, cap: int | None = config.SUBSET_CAP) -> pd.DataFrame:
    """Stratified down-sample to ``cap`` rows (keeps class proportions). No-op if None."""
    if cap is None or len(df) <= cap:
        return df
    frac = cap / len(df)
    # GroupBy.sample does stratified per-class sampling directly — avoids the
    # groupby(...).apply(...) pitfall where pandas >=3.0 drops the grouping
    # ('label') column from the result, which would later break the split.
    capped = (
        df.groupby("label", group_keys=False)
        .sample(frac=frac, random_state=config.SEED)
        .reset_index(drop=True)
    )
    print(f"[preprocess] Capped dataset {len(df)} -> {len(capped)} rows (subset).")
    return capped


def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split using two sklearn calls and the fixed seed."""
    # First peel off the test set (15%).
    train_val, test = train_test_split(
        df,
        test_size=config.TEST_RATIO,
        stratify=df["label"],
        random_state=config.SEED,
    )
    # From the remaining 85%, take val so it is 15% of the *original* total.
    val_fraction = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val["label"],
        random_state=config.SEED,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _print_distribution(name: str, df: pd.DataFrame, class_list: list[str]) -> None:
    """Print per-class proportions so splits can be eyeballed for balance."""
    props = df["label"].value_counts(normalize=True).sort_index()
    pretty = ", ".join(
        f"{class_list[i]}={props.get(i, 0):.3f}" for i in range(len(class_list))
    )
    print(f"[preprocess] {name:5s} n={len(df):6d} | {pretty}")


def preprocess() -> None:
    """Run the full preprocessing pipeline and write split manifests."""
    from src.data.download import resolve_dataset_paths

    config.ensure_dirs()
    styles_csv, images_dir = resolve_dataset_paths()
    print(f"[preprocess] styles.csv = {styles_csv}")
    print(f"[preprocess] images dir = {images_dir}")

    df = load_styles(styles_csv)
    print(f"[preprocess] Loaded {len(df)} rows.")

    df, class_list = filter_top_classes(df)
    print(f"[preprocess] Top-{config.N_CLASSES} classes: {class_list}")

    df = attach_image_paths(df, images_dir, verify=True)
    df = encode_labels(df, class_list)
    df = cap_subset(df)

    config.save_class_list(class_list)
    print(f"[preprocess] Saved class list -> {config.CLASS_LIST_FILE}")

    train, val, test = stratified_split(df)
    _print_distribution("train", train, class_list)
    _print_distribution("val", val, class_list)
    _print_distribution("test", test, class_list)

    cols = [config.ID_COLUMN, "image_path", config.TEXT_COLUMN, "label", config.LABEL_COLUMN]
    train[cols].to_csv(config.TRAIN_MANIFEST, index=False)
    val[cols].to_csv(config.VAL_MANIFEST, index=False)
    test[cols].to_csv(config.TEST_MANIFEST, index=False)
    print(
        f"[preprocess] Wrote manifests to {config.SPLITS_DIR} "
        f"(train={len(train)}, val={len(val)}, test={len(test)})."
    )


if __name__ == "__main__":
    config.set_seeds()
    preprocess()
