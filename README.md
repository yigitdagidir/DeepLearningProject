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

## Setup

```bash
pip install -r requirements.txt
python -m src.data.download
python -m src.data.preprocess
```

## Training & evaluation

```bash
# Train each variant
python -m src.train --model image
python -m src.train --model text
python -m src.train --model fusion

# Evaluate
python -m src.evaluate --model fusion
```

We train all three variants on purpose: the comparison between single-modal baselines and
the fusion model is the core result.

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

1. Theoretical + mathematical report
2. Documented source code
3. Final presentation
