"""
Run a trained model on a single (image, caption) pair and print the predicted
class with its probability.  Handy for the live demo during the presentation.

Run (after training the demo model):
    python -m src.predict --image data/demo/images/test/shoes_0001.png \
                          --caption "a pair of black leather shoes" \
                          --model outputs/run/model.pt
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .utils import get_device
from .data.dataset import build_transforms
from .data.text_utils import Vocabulary
from .data.dataset import build_dataloaders
from .models.multimodal import MultiModalNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--caption", required=True)
    p.add_argument("--model", default="outputs/run/model.pt")
    p.add_argument("--config", default="outputs/run/config.json")
    args = p.parse_args()

    cfg = Config.from_json(args.config) if os.path.exists(args.config) else Config()
    device = get_device()

    # We rebuild the vocabulary the same way training did, so the word ids match.
    _, info = build_dataloaders(cfg)
    vocab: Vocabulary = info["vocab"]
    classes = info["classes"]

    model = MultiModalNet(cfg, info["vocab_size"], info["pad_idx"],
                          info["num_classes"]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tf = build_transforms(cfg.image_size, train=False)
    img = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    text = torch.tensor(vocab.encode(args.caption, cfg.max_text_len),
                        dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(img, text), dim=1).squeeze(0)
    top = torch.argsort(probs, descending=True)[:3]

    print(f"\ncaption: {args.caption}")
    print("top-3 predictions:")
    for i in top:
        print(f"  {classes[i]:<10s}  {probs[i].item()*100:5.1f}%")


if __name__ == "__main__":
    main()
