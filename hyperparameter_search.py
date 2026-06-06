"""
Random-search hyper-parameter tuning.

The project asks us to "use a tuning technique (such as Grid Search or Random
Search) to optimise hyper-parameters, including learning rate, batch size and
the number of fully connected layers for fusion".

Random search is usually more efficient than grid search for the same budget
(Bergstra & Bengio, 2012), so that is what we use here.  For each sampled
configuration we train a model and record the best validation accuracy; the
configuration with the highest validation accuracy wins.

Run:
    python -m src.hyperparameter_search --trials 8 --epochs 6
"""

import argparse
import json
import random

import numpy as np

from .config import Config
from .utils import set_seed, get_device, ensure_dir
from .data.dataset import build_dataloaders
from .models.multimodal import MultiModalNet
from .engine import fit


# the search space: each entry is a list of values we sample uniformly from
SEARCH_SPACE = {
    "lr": [3e-4, 5e-4, 1e-3, 2e-3],
    "batch_size": [16, 32, 64],
    "rnn_type": ["lstm", "gru"],
    "rnn_hidden": [128, 256],
    "dropout": [0.3, 0.5, 0.6],
    "fusion_hidden": [[128], [256], [256, 128]],   # number/size of FC fusion layers
}


def sample_config(base: Config, rng: random.Random) -> Config:
    cfg = Config(**vars(base))           # shallow copy of the base config
    for key, choices in SEARCH_SPACE.items():
        setattr(cfg, key, rng.choice(choices))
    return cfg


def run_search(base: Config, trials: int, out_path: str):
    rng = random.Random(base.seed)
    device = get_device()
    results = []

    for t in range(1, trials + 1):
        cfg = sample_config(base, rng)
        set_seed(base.seed)             # same init each trial -> fair comparison
        loaders, info = build_dataloaders(cfg)
        model = MultiModalNet(cfg, info["vocab_size"], info["pad_idx"],
                              info["num_classes"]).to(device)
        _, best_val = fit(model, loaders, cfg, device, verbose=False)

        trial_info = {
            "trial": t, "val_acc": best_val,
            "lr": cfg.lr, "batch_size": cfg.batch_size, "rnn_type": cfg.rnn_type,
            "rnn_hidden": cfg.rnn_hidden, "dropout": cfg.dropout,
            "fusion_hidden": cfg.fusion_hidden,
        }
        results.append(trial_info)
        print(f"trial {t:02d}/{trials} | val_acc {best_val:.4f} | "
              f"lr={cfg.lr} bs={cfg.batch_size} rnn={cfg.rnn_type} "
              f"hid={cfg.rnn_hidden} drop={cfg.dropout} fc={cfg.fusion_hidden}")

    results.sort(key=lambda r: r["val_acc"], reverse=True)
    best = results[0]
    print("\n========== best configuration ==========")
    print(json.dumps(best, indent=2))

    with open(out_path, "w") as f:
        json.dump({"best": best, "all_trials": results}, f, indent=2)
    print(f"\nsaved search results to {out_path}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="demo")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--image_size", type=int, default=64)
    args = p.parse_args()

    base = Config(dataset=args.dataset, epochs=args.epochs, image_size=args.image_size)
    out = ensure_dir(base.out_dir)
    run_search(base, args.trials, out_path=f"{out}/hparam_search.json")


if __name__ == "__main__":
    main()
