"""
Helper that renders the mathematical equations (as small PNG images via
matplotlib's mathtext) and the architecture diagram used in the report and the
slides.  We render the maths as images so the equations look clean and identical
in Word, in PowerPoint and in a PDF export.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)


def render_eq(lines, name, fontsize=20):
    """
    Render one or more LaTeX lines to a PNG and return (path, width_in, height_in).
    `lines` is a list of raw mathtext strings (without surrounding $).
    """
    n = len(lines)
    fig_h = 0.55 * n + 0.15
    fig_w = 6.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    for i, line in enumerate(lines):
        y = 1 - (i + 0.5) / n
        fig.text(0.5, y, f"${line}$", ha="center", va="center", fontsize=fontsize)
    path = os.path.join(ASSETS, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def architecture_diagram():
    """A simple block diagram of the multi-modal pipeline."""
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                           linewidth=1.5, edgecolor="#333333", facecolor=color)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, linewidth=1.5, color="#333333"))

    # image path (top)
    box(0.2, 4.2, 1.7, 1.1, "Image\n(H x W x 3)", "#dce9f7")
    box(2.3, 4.2, 2.1, 1.1, "Pretrained CNN\n(ResNet-18)", "#bcd6f0")
    box(4.8, 4.2, 1.9, 1.1, "Image vector\nv_img (256)", "#a7c8ea")

    # text path (bottom)
    box(0.2, 0.6, 1.7, 1.1, "Caption\n(tokens)", "#dff2dc")
    box(2.3, 0.6, 2.1, 1.1, "Embedding +\nBi-LSTM/GRU", "#c2e3bd")
    box(4.8, 0.6, 1.9, 1.1, "Text vector\nv_txt (512)", "#a8d6a0")

    # fusion (right)
    box(7.0, 2.4, 1.4, 1.2, "Concat\n[v_img ; v_txt]", "#f7e2c0")
    box(8.5, 2.4, 1.3, 1.2, "FC + ReLU\n+ Dropout", "#f3d39b")
    box(8.5, 0.7, 1.3, 1.0, "Softmax\n(classes)", "#f0b9b9")

    arrow(1.9, 4.75, 2.3, 4.75)
    arrow(4.4, 4.75, 4.8, 4.75)
    arrow(1.9, 1.15, 2.3, 1.15)
    arrow(4.4, 1.15, 4.8, 1.15)
    arrow(6.7, 4.75, 7.7, 3.6)      # image vec -> concat
    arrow(6.7, 1.15, 7.7, 2.4)      # text vec  -> concat
    arrow(8.4, 3.0, 8.5, 3.0)       # concat -> fc
    arrow(9.15, 2.4, 9.15, 1.7)     # fc -> softmax

    path = os.path.join(ASSETS, "architecture.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    return path


def dataset_montage(meta_path, n_per_class=2):
    """Make a montage of a few demo images per class with their captions."""
    import json
    from PIL import Image as PILImage, ImageDraw, ImageFont
    with open(meta_path) as f:
        meta = json.load(f)
    classes = meta["classes"]
    root = os.path.dirname(meta_path)

    # pick the first test sample whose IMAGE is not corrupted, per class
    # (corrupt == "image" means the picture is noise, so we skip those)
    chosen = {}
    for r in meta["records"]:
        if (r["split"] == "test" and r["corrupt"] != "image"
                and r["label_name"] not in chosen):
            chosen[r["label_name"]] = r
    cols = len(classes)
    cell, pad, cap_h = 130, 10, 34
    W = cols * (cell + pad) + pad
    H = cell + cap_h + 2 * pad
    canvas = PILImage.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for i, cname in enumerate(classes):
        r = chosen.get(cname)
        if not r:
            continue
        img = PILImage.open(os.path.join(root, r["image"])).convert("RGB").resize((cell, cell))
        x = pad + i * (cell + pad)
        canvas.paste(img, (x, pad))
        draw.text((x + 4, pad + cell + 6), cname, fill="black")
    path = os.path.join(ASSETS, "dataset_montage.png")
    canvas.save(path)
    return path


if __name__ == "__main__":
    architecture_diagram()
    print("assets written to", ASSETS)
