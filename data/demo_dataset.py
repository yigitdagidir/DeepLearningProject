"""
Synthetic "Fashion-Gen-like" dataset generator.

Why this file exists
--------------------
The real Fashion-Gen dataset is ~10 GB of HDF5 files behind a registration
form, which is impractical to download just to check that our pipeline runs.
So we generate a *small* dataset that has the SAME STRUCTURE as Fashion-Gen:

    * a clothing image                (the CNN input)
    * a textual style/category caption (the RNN input)
    * a category label                (what we classify)

The images are simple coloured shapes, but they are genuinely learnable by a
CNN, and the captions are genuinely informative for the RNN, so the numbers we
report are *real* (the network actually has to learn something).

Fusion experiment
-----------------
To study the effect of multi-modal fusion (the project explicitly asks for
this), every sample has exactly one of its two modalities "corrupted":

    * image corrupted  -> the image is replaced by pure noise
    * text  corrupted  -> the caption is replaced by a generic, useless string

The two corrupted subsets are disjoint, so for any sample at least one modality
is still informative.  Consequently an image-only or text-only model has an
upper bound well below 100 %, while a model that fuses both modalities can in
principle recover the information -> we can measure the benefit of fusion.
"""

import os
import json
import random
from typing import List, Dict

import numpy as np
from PIL import Image, ImageDraw


# The six clothing categories we use for the demo (a subset of Fashion-Gen style
# categories).  Order matters: the index is the integer label.
CLASSES: List[str] = ["t_shirt", "dress", "pants", "shoes", "bag", "hat"]

# small vocabularies used to build varied captions
COLORS = ["red", "blue", "green", "black", "white", "yellow", "pink", "grey", "brown"]
MATERIALS = ["cotton", "denim", "leather", "wool", "silk", "linen", "polyester"]
STYLES = ["casual", "elegant", "sporty", "vintage", "modern", "classic"]

# per-class caption templates (the {} are filled from the vocabularies above)
TEMPLATES: Dict[str, List[str]] = {
    "t_shirt": [
        "a {color} {material} short sleeve t shirt with a round neck",
        "{style} {color} t shirt made of {material}",
    ],
    "dress": [
        "an {style} {color} {material} dress with a long skirt",
        "a {color} summer dress in soft {material}",
    ],
    "pants": [
        "a pair of {color} {material} pants with a {style} fit",
        "{style} {color} trousers made of {material}",
    ],
    "shoes": [
        "a pair of {color} {material} shoes with flat soles",
        "{style} {color} sneakers in {material}",
    ],
    "bag": [
        "a {color} {material} handbag with a long strap",
        "{style} {color} shoulder bag made of {material}",
    ],
    "hat": [
        "a {color} {material} hat with a wide brim",
        "{style} {color} cap made of {material}",
    ],
}

GENERIC_CAPTION = "a fashion item"     # used when the text modality is corrupted


# --------------------------------------------------------------------------- #
#  Image drawing                                                              #
# --------------------------------------------------------------------------- #
def _rand_color(rng) -> tuple:
    return tuple(int(c) for c in rng.integers(30, 226, size=3))


def _draw_shape(label: str, size: int, rng) -> Image.Image:
    """Draw a simple but class-distinctive coloured shape on a noisy background."""
    # light noisy background so the task is not trivially solved by the mean colour
    bg = rng.integers(200, 256, size=(size, size, 3)).astype(np.uint8)
    img = Image.fromarray(bg, mode="RGB")
    d = ImageDraw.Draw(img)
    col = _rand_color(rng)
    m = size // 8                          # margin
    jx, jy = int(rng.integers(-m // 2, m // 2 + 1)), int(rng.integers(-m // 2, m // 2 + 1))

    if label == "t_shirt":
        d.rectangle([m + jx, 2 * m + jy, size - m + jx, size - 2 * m + jy], fill=col)
        d.rectangle([jx, 2 * m + jy, m + jx, 4 * m + jy], fill=col)             # left sleeve
        d.rectangle([size - m + jx, 2 * m + jy, size + jx, 4 * m + jy], fill=col)  # right sleeve
    elif label == "dress":
        d.polygon([(3 * m + jx, m + jy), (size - 3 * m + jx, m + jy),
                   (size - m + jx, size - m + jy), (m + jx, size - m + jy)], fill=col)  # flared
    elif label == "pants":
        d.rectangle([3 * m + jx, m + jy, size // 2 + jx, size - m + jy], fill=col)      # left leg
        d.rectangle([size // 2 + jx, m + jy, size - 3 * m + jx, size - m + jy], fill=col)  # right leg
    elif label == "shoes":
        d.ellipse([m + jx, size - 3 * m + jy, size - m + jx, size - m + jy], fill=col)  # flat ellipse
    elif label == "bag":
        d.rectangle([2 * m + jx, 3 * m + jy, size - 2 * m + jx, size - 2 * m + jy], fill=col)
        d.arc([3 * m + jx, m + jy, size - 3 * m + jx, 4 * m + jy], 180, 360, fill=col, width=3)  # handle
    elif label == "hat":
        d.pieslice([2 * m + jx, 2 * m + jy, size - 2 * m + jx, size + 2 * m + jy], 180, 360, fill=col)
        d.rectangle([m + jx, size // 2 + jy, size - m + jx, size // 2 + m + jy], fill=col)  # brim
    return img


def _noise_image(size: int, rng) -> Image.Image:
    """Pure noise -> used when the image modality is corrupted."""
    arr = rng.integers(0, 256, size=(size, size, 3)).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_caption(label: str, rng) -> str:
    tmpl = TEMPLATES[label][int(rng.integers(0, len(TEMPLATES[label])))]
    return tmpl.format(color=rng.choice(COLORS),
                       material=rng.choice(MATERIALS),
                       style=rng.choice(STYLES))


# --------------------------------------------------------------------------- #
#  Dataset generation                                                         #
# --------------------------------------------------------------------------- #
def generate(root: str = "data/demo",
             n_train_per_class: int = 120,
             n_test_per_class: int = 30,
             image_size: int = 64,
             corrupt_frac: float = 0.45,
             seed: int = 42) -> str:
    """
    Create the demo dataset on disk and return the path of the metadata file.

    corrupt_frac: fraction of samples whose IMAGE is replaced by noise, plus the
                  same fraction whose TEXT is replaced by the generic caption
                  (disjoint groups).  0.45 + 0.45 leaves ~10 % "clean" samples.
    """
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for split, n_per in [("train", n_train_per_class), ("test", n_test_per_class)]:
        os.makedirs(os.path.join(img_dir, split), exist_ok=True)
        for label_idx, label in enumerate(CLASSES):
            for k in range(n_per):
                # decide corruption: 0 = clean, 1 = image noise, 2 = text generic
                r = rng.random()
                corrupt_img = r < corrupt_frac
                corrupt_txt = (not corrupt_img) and (r < 2 * corrupt_frac)

                if corrupt_img:
                    img = _noise_image(image_size, rng)
                else:
                    img = _draw_shape(label, image_size, rng)

                caption = GENERIC_CAPTION if corrupt_txt else _make_caption(label, rng)

                fname = f"{split}/{label}_{k:04d}.png"
                img.save(os.path.join(img_dir, fname))
                records.append({
                    "image": os.path.join("images", fname).replace("\\", "/"),
                    "caption": caption,
                    "label": label_idx,
                    "label_name": label,
                    "split": split,
                    "corrupt": "image" if corrupt_img else ("text" if corrupt_txt else "none"),
                })

    meta_path = os.path.join(root, "demo_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"classes": CLASSES, "records": records,
                   "image_size": image_size}, f, indent=2)
    print(f"[demo] wrote {len(records)} samples to {root}")
    return meta_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate the synthetic Fashion-Gen-like demo set.")
    p.add_argument("--root", default="data/demo")
    p.add_argument("--train_per_class", type=int, default=120)
    p.add_argument("--test_per_class", type=int, default=30)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    generate(args.root, args.train_per_class, args.test_per_class,
             args.image_size, seed=args.seed)
