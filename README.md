# Multi-Modal Classification of Images and Text with Deep Learning

A deep-learning pipeline that classifies fashion items by **fusing** two
modalities:

* the **image** of the item — encoded with a pre-trained **CNN** (ResNet),
* the **text** description of the item — encoded with an **RNN** (LSTM / GRU)
  on top of word embeddings,

and combines the two by **concatenation + fully-connected layers + softmax**.

This repository is the source-code deliverable for the *Deep Learning Project –
Multi-Modal Classification and Analysis of Images and Text*. It is written for
the **Fashion-Gen** dataset, and ships with a small synthetic *demo* dataset
that has the exact same structure so the whole pipeline can be run and verified
in 1–2 minutes on a normal laptop **without downloading anything**.

---

## 1. Project structure

```
Deep Learning Project/
├── README.md
├── requirements.txt
├── TASK_DISTRIBUTION.md          # who does what (3-person team)
├── report/                       # theoretical & mathematical report (.docx)
├── presentation/                 # final slides (.pptx)
├── outputs/                      # created when you train (metrics, figures, weights)
└── src/
    ├── config.py                 # all hyper-parameters in one dataclass
    ├── utils.py                  # seeding, metrics, plotting
    ├── engine.py                 # train / eval loops
    ├── train.py                  # MAIN training script
    ├── run_ablation.py           # image-only vs text-only vs fusion
    ├── cross_validation.py       # k-fold cross-validation
    ├── hyperparameter_search.py  # random search
    ├── predict.py                # single-sample inference (for the live demo)
    ├── data/
    │   ├── text_utils.py         # tokenizer + Vocabulary
    │   ├── demo_dataset.py       # synthetic Fashion-Gen-like generator
    │   ├── fashiongen.py         # loader for the real Fashion-Gen .h5 files
    │   └── dataset.py            # PyTorch Dataset + dataloaders
    └── models/
        ├── cnn_encoder.py        # image branch (ResNet/EfficientNet)
        ├── text_encoder.py       # text branch (embedding + LSTM/GRU)
        ├── fusion.py             # concatenation + FC + softmax head
        └── multimodal.py         # full model (image + text + fusion)
```

## 2. Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
```

> The CPU build of PyTorch is enough for the demo. For real Fashion-Gen
> training a GPU is strongly recommended.

## 3. Quick start (demo — no download needed)

```bash
# 1) Train the full multi-modal model on the synthetic demo set
python -m src.train --dataset demo --epochs 15 --experiment_name fusion

# 2) Reproduce the fusion ablation (image-only vs text-only vs fusion)
python -m src.run_ablation --epochs 15

# 3) Cross-validation
python -m src.cross_validation --folds 5 --epochs 8

# 4) Hyper-parameter random search
python -m src.hyperparameter_search --trials 8 --epochs 6

# 5) Predict on a single sample (after step 1)
python -m src.predict --image data/demo/images/test/shoes_0001.png \
                      --caption "a pair of black leather shoes with flat soles" \
                      --model outputs/fusion/model.pt --config outputs/fusion/config.json
```

The demo data is generated automatically the first time you run training and is
saved under `data/demo/`. Each training run writes to `outputs/<experiment_name>/`:

* `model.pt` – best weights (selected on validation accuracy)
* `training_curves.png` – loss & accuracy per epoch
* `confusion_matrix.png`
* `classification_report.txt` – per-class precision / recall / F1
* `metrics.json` – everything in machine-readable form
* `config.json` – the exact configuration used

## 4. Running on the real Fashion-Gen dataset

1. Request / download the Fashion-Gen `.h5` files and place them in
   `data/fashiongen/`:
   * `fashiongen_256_256_train.h5`
   * `fashiongen_256_256_validation.h5`
2. Train:

```bash
python -m src.train --dataset fashiongen --data_root data/fashiongen \
                    --image_size 224 --batch_size 64 --epochs 20 \
                    --experiment_name fashiongen_fusion
```

The loader (`src/data/fashiongen.py`) reads `input_image`,
`input_description` and `input_category` from the HDF5 files; the rest of the
code is identical to the demo.

## 5. What each design choice means (short version — full math is in the report)

| Component | Choice | Why |
|-----------|--------|-----|
| Image encoder | ResNet-18, ImageNet-pretrained, top layers fine-tuned | transfer learning → good features with little data |
| Text encoder | Embedding + Bi-LSTM, last hidden state | captures word order & context in the caption |
| Fusion | concatenate image & text vectors → FC layers | FC layers learn cross-modal correlations |
| Output | softmax over classes, cross-entropy loss | standard multi-class classification |
| Optimiser | Adam + StepLR | fast, adaptive, well-behaved default |
| Regularisation | dropout in the fusion layers + weight decay | reduce over-fitting |
| Validation | hold-out val set + k-fold cross-validation | check generalisation |

## 6. Reproducibility

Every run is seeded (`--seed 42` by default). On CPU the results are
deterministic up to small numerical noise.
