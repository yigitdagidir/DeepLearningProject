# Team Task Distribution

**Project:** Multi-Modal Classification and Analysis of Images and Text using Deep Learning
**Team size:** 3 members
**Working style:** The whole team prepares everything *together* (pair/mob programming
for the hard parts, shared report writing). The table below assigns a **primary owner**
to each block so nothing falls through the cracks, but every member should understand
the whole pipeline because the final presentation is given by all three.

**Difficulty legend:** 🟢 Easy · 🟡 Medium · 🔴 Hard

---

## Member 1 — Image Branch & Data Pipeline

| # | Task | Files | Difficulty |
|---|------|-------|-----------|
| 1 | Image preprocessing: resize, normalisation (ImageNet stats), augmentation | `src/data/dataset.py` (`build_transforms`) | 🟢 Easy |
| 2 | Demo dataset generator (synthetic Fashion-Gen-like images + captions) | `src/data/demo_dataset.py` | 🟡 Medium |
| 3 | Real Fashion-Gen `.h5` loader | `src/data/fashiongen.py` | 🟡 Medium |
| 4 | CNN encoder: load pretrained ResNet, drop classifier, add projection head, freezing/fine-tuning | `src/models/cnn_encoder.py` | 🔴 Hard |
| 5 | Report sections: CNN math (convolution, pooling, ReLU, batch-norm, transfer learning) | `report/` | 🔴 Hard |

## Member 2 — Text Branch & Fusion

| # | Task | Files | Difficulty |
|---|------|-------|-----------|
| 1 | Tokeniser + Vocabulary (build from train captions only, padding/unk handling) | `src/data/text_utils.py` | 🟡 Medium |
| 2 | Text encoder: embedding layer, LSTM/GRU, packed sequences, last hidden state | `src/models/text_encoder.py` | 🔴 Hard |
| 3 | (Optional) GloVe pretrained-embedding loading | `text_encoder.load_glove` | 🟡 Medium |
| 4 | Fusion head: concatenation + FC layers + dropout + final logits | `src/models/fusion.py`, `src/models/multimodal.py` | 🔴 Hard |
| 5 | Report sections: word embeddings, LSTM/GRU gate equations, fusion math | `report/` | 🔴 Hard |

## Member 3 — Training, Optimisation, Evaluation & Deliverables

| # | Task | Files | Difficulty |
|---|------|-------|-----------|
| 1 | Config system + CLI argument parsing | `src/config.py`, `src/train.py` | 🟡 Medium |
| 2 | Training / evaluation loops, optimizer, LR scheduler, best-model selection | `src/engine.py` | 🟡 Medium |
| 3 | Metrics & plots: accuracy, confusion matrix, classification report, curves | `src/utils.py` | 🟢 Easy |
| 4 | Hyper-parameter random search | `src/hyperparameter_search.py` | 🔴 Hard |
| 5 | K-fold cross-validation | `src/cross_validation.py` | 🔴 Hard |
| 6 | Fusion ablation experiment (image vs text vs fusion) | `src/run_ablation.py` | 🟡 Medium |
| 7 | Report sections: cross-entropy loss, optimizer (Adam) math, regularisation, results | `report/` | 🟡 Medium |

---

## Shared / done together

| Task | Difficulty | Notes |
|------|-----------|-------|
| Final presentation (`presentation/`) | 🟡 Medium | Each member presents the part they owned |
| Final report integration & proofreading | 🟡 Medium | Merge the three members' sections, unify notation |
| Running experiments & collecting figures | 🟢 Easy | One person runs, all interpret the results |
| Live demo of `predict.py` during the presentation | 🟢 Easy | Member 3 drives, Members 1 & 2 explain |

---

## Suggested timeline (2 weeks)

| Day | Milestone |
|-----|-----------|
| 1–2 | Read the project, choose dataset (Fashion-Gen), set up repo & environment |
| 3–5 | Build the three branches (CNN, text, fusion) — pair on the 🔴 hard parts |
| 6–7 | First end-to-end training run on the demo set; fix bugs |
| 8–9 | Hyper-parameter search + cross-validation + ablation |
| 10–11 | Write the theoretical & mathematical report |
| 12 | Build the presentation, prepare the live demo |
| 13 | Full rehearsal, collect final figures |
| 14 | Buffer / polish |

---

## Effort balance (rough)

All three members carry **two 🔴 hard tasks** each, so the workload is balanced:

* Member 1: CNN encoder + CNN math
* Member 2: text encoder + fusion + RNN/fusion math
* Member 3: hyper-parameter search + cross-validation

The medium/easy tasks are spread so that nobody is only doing easy work and
nobody is overloaded.
