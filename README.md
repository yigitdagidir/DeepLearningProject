# Multi-Modal Fashion Classifier

A deep learning project that classifies fashion products using **both** their image and
their text description. Built for a university Deep Learning course.

The system has three parts — a CNN image encoder, an RNN text encoder, and a fusion head
that combines them — and the goal is to show that combining both modalities outperforms
using either one alone.

## Approach

| Component   | Choice                                                        |
|-------------|---------------------------------------------------------------|
| Image       | EfficientNetB0 (frozen ImageNet backbone) → dense projection  |
| Text        | TextVectorization → Embedding → GRU → dense projection         |
| Fusion      | Concatenate → Dense → Dropout → Softmax                        |
| Dataset     | Kaggle *Fashion Product Images (small)*                        |
| Target      | `subCategory`, top ~10 classes                                 |
| Framework   | TensorFlow / Keras, runs on Google Colab                       |

## Run it on Colab (recommended)

The project targets **Google Colab free GPU** (TensorFlow has no Python 3.14 wheels;
use Python 3.10–3.12 locally if you must). The fastest path is the turnkey notebook
[`notebooks/run_colab.ipynb`](./notebooks/run_colab.ipynb): open it on Colab with a GPU
runtime, set your repo URL in the first cell, and **Run all** — it downloads the data,
builds the splits, trains all three models, evaluates them, and writes the 3-way
comparison. EDA lives in [`notebooks/exploration.ipynb`](./notebooks/exploration.ipynb).

## Setup (CLI)

```bash
pip install -r requirements.txt
python -m src.data.download        # Kaggle small dataset -> data/ (idempotent)
python -m src.data.preprocess      # top-10 classes, stratified 70/15/15 manifests
```

## Training, evaluation & comparison

```bash
# Train each variant (saves weights + metrics.json + training curves)
python -m src.train --model image
python -m src.train --model text
python -m src.train --model fusion

# Evaluate each on the held-out test set (test metrics + confusion matrix)
python -m src.evaluate --model image
python -m src.evaluate --model text
python -m src.evaluate --model fusion

# Assemble the headline 3-way comparison (table + chart)
python -m src.compare              # -> artifacts/comparison.{csv,md,png}
```

We train all three variants on purpose: the comparison between single-modal baselines and
the fusion model is the core result. Outputs land in `artifacts/<model>/`.

## Repository layout

See [`CLAUDE.md`](./CLAUDE.md) for the full structure and design decisions, and
[`docs/PROJECT_BRIEF.md`](./docs/PROJECT_BRIEF.md) for the assignment requirements.
The phase-by-phase task plan lives in [`docs/DEVELOPMENT_PLAN.md`](./docs/DEVELOPMENT_PLAN.md).

## Team

3-person group:

- **Person A** — image / CNN branch
- **Person B** — text / RNN branch
- **Person C** — data pipeline, fusion, training & evaluation infrastructure

## Deliverables

1. **Theoretical + mathematical report** — [`docs/REPORT.md`](./docs/REPORT.md)
   (CNN conv/pool/activation, GRU gate equations, cross-entropy + Adam, fusion rationale)
2. **Documented source code** — the `src/` package + notebooks
3. **Final presentation** — [`docs/PRESENTATION.md`](./docs/PRESENTATION.md)
   (Marp slides; render with `npx @marp-team/marp-cli docs/PRESENTATION.md --pdf`)
