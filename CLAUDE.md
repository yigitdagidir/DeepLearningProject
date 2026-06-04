# CLAUDE.md

Context file for Claude Code. **Read this first**, then `docs/PROJECT_BRIEF.md`, then
`docs/DEVELOPMENT_PLAN.md` (the latter is the living task list — keep it updated).

---

## 1. What we are building

A **multi-modal deep learning classifier** for a university Deep Learning course.
Given a fashion product's **image** and its **text description**, the model predicts the
product category. There are three components:

- an **image encoder** (CNN),
- a **text encoder** (word embeddings + RNN),
- a **fusion head** that concatenates both encodings and classifies.

The pedagogical point is to **prove that multi-modal fusion beats either modality alone**.

## 2. Team & working style

- 3-person group. Roles map onto the architecture:
  - **Person A** — image / CNN branch
  - **Person B** — text / RNN branch
  - **Person C** — shared data pipeline, fusion, training/eval infra, report assembly
- Optimize for **"works and is explainable under time pressure,"** not maximum accuracy.
- This is a graded project: code must be **readable and well-documented** because it
  feeds a theoretical + mathematical report.

## 3. Tech stack

- Python 3.10+
- **TensorFlow / Keras** (use the **functional API** for the multi-modal model)
- `tf.data` for the input pipeline
- pandas, numpy, scikit-learn (split + metrics)
- matplotlib / seaborn (training curves, confusion matrix)
- Runs on **Google Colab free GPU** — keep memory and compute modest.

## 4. Dataset

- **Kaggle:** `paramaggarwal/fashion-product-images-small` (low-res, ~600 MB; preferred
  for Colab). Only use the full-resolution version if explicitly asked — it is ~25 GB.
- `styles.csv` columns: `id, gender, masterCategory, subCategory, articleType,
  baseColour, season, year, usage, productDisplayName`
- Images live at `images/{id}.jpg`.
- **Target label:** `subCategory`, filtered to the **top ~10 most frequent classes**
  (masterCategory is too easy → fusion contribution becomes invisible; articleType has
  140+ classes → too hard for the timeline).
- **Text input:** `productDisplayName`.
- **Split:** stratified **train/val/test = 70/15/15**. Fixed random seed.
- Note: small-version images are ~60×80; we upscale to the CNN's expected input. This is
  acceptable for a course project.

## 5. Architecture (this is the agreed design — keep it simple)

- **Image branch:** `EfficientNetB0(weights="imagenet", include_top=False)`, backbone
  **FROZEN** → `GlobalAveragePooling2D` → `Dense(256, relu)`
- **Text branch:** `TextVectorization` → `Embedding` → `GRU(128)` → `Dense(128, relu)`
- **Fusion:** `Concatenate([img, txt])` → `Dense(256, relu)` → `Dropout(0.3)` →
  `Dense(n_classes, softmax)`
- **Loss:** sparse categorical cross-entropy
- **Optimizer:** Adam
- **Training:** single stage with the backbone frozen. *Optional* stage 2: unfreeze the
  top EfficientNet blocks and fine-tune at a low learning rate — **only if time permits.**

## 6. Required baselines (DO NOT SKIP)

Train and record metrics for **three** models so the report can show fusion helps:

1. `image-only`
2. `text-only`
3. `fusion` (multi-modal)

This three-way comparison is the backbone of both the report and the presentation.

## 7. Repository structure (target)

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── PROJECT_BRIEF.md
│   └── DEVELOPMENT_PLAN.md      # living plan — Claude Code keeps this updated
├── data/                        # downloaded dataset (gitignored)
├── notebooks/
│   └── exploration.ipynb
└── src/
    ├── config.py                # paths, hyperparameters, class list, seed
    ├── data/
    │   ├── download.py          # kagglehub download
    │   ├── preprocess.py        # filter classes, build splits
    │   └── dataset.py           # tf.data pipeline yielding (image, text), label
    ├── models/
    │   ├── image_branch.py
    │   ├── text_branch.py
    │   └── fusion.py
    ├── train.py                 # --model {image,text,fusion}
    └── evaluate.py              # metrics + confusion matrix
```

## 8. Commands (target — create these as the code is built)

```bash
pip install -r requirements.txt
python -m src.data.download        # download + unpack dataset into data/
python -m src.data.preprocess      # build filtered, stratified splits
python -m src.train --model fusion # also: image, text
python -m src.evaluate --model fusion
```

## 9. Conventions

- Centralize all hyperparameters, paths, the class list, and the random seed in
  `src/config.py`. No magic numbers scattered in modules.
- Keep functions small and docstringed — the code is read for the report.
- Set a fixed seed everywhere for reproducibility.
- Each model variant must save: trained weights, a metrics JSON, and training-curve plots.

## 10. Guardrails for Claude Code

- **Do not over-engineer.** No PyTorch, no `transformers`/BERT, no heavy AutoML unless
  explicitly requested. Trainable Keras embeddings are fine — do not pull in GloVe files.
- Prefer the **small** dataset version.
- **Skip k-fold cross-validation**; use the held-out stratified split (justified by
  compute cost — document this choice in the report).
- After completing any task, **tick its checkbox in `docs/DEVELOPMENT_PLAN.md`** and note
  any deviation from the plan.
- When a design decision is ambiguous, prefer the simplest option that satisfies the
  assignment requirements in `docs/PROJECT_BRIEF.md`.

## 11. Deliverables

1. **Theoretical + mathematical report** (CNN/pooling/activations, GRU gate equations,
   cross-entropy + Adam, fusion rationale).
2. **Documented source code.**
3. **Presentation** — emphasize the impact of multi-modal fusion (the 3-way comparison).
