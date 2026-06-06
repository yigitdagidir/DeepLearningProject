"""
Loader for the *real* Fashion-Gen dataset.

Fashion-Gen ships as HDF5 files, e.g.:
    fashiongen_256_256_train.h5
    fashiongen_256_256_validation.h5

with (among others) these datasets inside the file:
    input_image        (N, 256, 256, 3)  uint8
    input_description  (N, 1)             bytes   -> the caption / product text
    input_category     (N, 1)             bytes   -> e.g. "SWEATERS", "SHOES", ...
    input_subcategory  (N, 1)             bytes

We use `input_category` as the classification target.  The function below reads
the file lazily (it does NOT load all images into RAM) and returns plain python
lists that the common `MultiModalDataset` wrapper understands, so the rest of
the code does not care whether the data came from the demo set or the real one.

To use it:
    1) download the .h5 files from the Fashion-Gen page,
    2) put them under  data/fashiongen/,
    3) run training with  --dataset fashiongen  --data_root data/fashiongen
"""

import os
from typing import List, Tuple


def _decode(x) -> str:
    """HDF5 stores strings as bytes; turn them into a normal python string."""
    if isinstance(x, bytes):
        return x.decode("latin-1", errors="ignore")
    # sometimes it is a 1-element array of bytes
    try:
        return x[0].decode("latin-1", errors="ignore")
    except Exception:
        return str(x)


def load_fashiongen(h5_path: str, max_samples: int = -1):
    """
    Returns (images, captions, labels, class_names).

    images   : list of HxWx3 uint8 numpy arrays
    captions : list of str
    labels   : list of int
    class_names : list of str (index == label id)
    """
    try:
        import h5py
    except ImportError as e:
        raise ImportError("Reading Fashion-Gen needs h5py:  pip install h5py") from e
    import numpy as np

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Could not find {h5_path}. Download the Fashion-Gen .h5 files first.")

    with h5py.File(h5_path, "r") as f:
        n = f["input_image"].shape[0]
        n = n if max_samples < 0 else min(n, max_samples)

        categories = [_decode(f["input_category"][i]) for i in range(n)]
        class_names = sorted(set(c.strip() for c in categories))
        cls_to_idx = {c: i for i, c in enumerate(class_names)}

        images, captions, labels = [], [], []
        for i in range(n):
            images.append(np.array(f["input_image"][i]))            # HxWx3 uint8
            captions.append(_decode(f["input_description"][i]))
            labels.append(cls_to_idx[categories[i].strip()])

    return images, captions, labels, class_names
