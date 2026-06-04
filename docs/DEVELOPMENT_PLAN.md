# Development Plan

Multi-modal deep learning classifier that predicts a fashion product's `subCategory`
from **both** its image and its `productDisplayName` text. The plan drives three
deliverables — a theoretical/mathematical report, documented source code, and a
presentation — whose central result is that **multi-modal fusion beats either modality
alone** (image-only vs text-only vs fusion).

> **Assumed timeline (explicit assumption):** ~1.5–2 weeks of part-time work by 3
> students on **Google Colab free GPU**. Phases are sized to "works and is explainable
> under time pressure," not maximum accuracy. If the real deadline differs, rescale the
> phase durations accordingly.

## Milestones / Phase overview

| Phase | Goal (one line)                                                        | Lead  |
|-------|------------------------------------------------------------------------|-------|
| 0     | Repo skeleton, dependencies, and shared `config.py` so all 3 can work  | C     |
| 1     | Reproducible stratified splits + `tf.data` pipeline `((image,text),y)` | C     |
| 2     | Train the two single-modal **baselines** (image-only, text-only)       | A & B |
| 3     | Build + train **fusion**; assemble the 3-way comparison                | C     |
| 4     | Lightweight tuning + final generalization metrics & figures            | C     |
| 5     | Report (math), documented code, presentation                           | C     |

*Roles:* **A** = image / CNN branch · **B** = text / RNN branch · **C** = data
pipeline, fusion, training/eval infra, report assembly.

## Phases

### Phase 0 — Setup

- **Goal:** Stand up the repo skeleton, dependencies, and centralized config so all
  three members can work in parallel.
- **Owner(s):** C (lead); A & B review.
- **Dependencies:** none.
- **Tasks:**
  - [ ] (C) Create `docs/` and add this `DEVELOPMENT_PLAN.md`.
  - [ ] (C) Reconcile the brief location: move `PROJECT_BRIEF.md` → `docs/PROJECT_BRIEF.md`
        (matches `CLAUDE.md` §7 and README links).
  - [ ] (C) Scaffold the `src/` package tree from `CLAUDE.md` §7 with stub modules and
        `__init__.py` files; add `data/`, `notebooks/`, `artifacts/` dirs.
  - [ ] (C) Write `requirements.txt` (tensorflow, pandas, numpy, scikit-learn,
        matplotlib, seaborn, kagglehub, pillow) — Colab-compatible, minimal pins.
  - [ ] (C) Extend `.gitignore` to ignore `data/`, `artifacts/`, and saved weights
        (`*.keras`, `*.h5`).
  - [ ] (C) Implement `src/config.py`: paths (data/artifacts), `SEED`, image size,
        batch size, vocab size, sequence length, embedding dim, GRU/Dense units,
        dropout, epochs, learning rate, `N_CLASSES`, `CLASS_LIST` placeholder, and
        the 70/15/15 ratios.
  - [ ] (C) Add a `set_seeds()` helper (python `random`, numpy, tf) used everywhere.
  - [ ] (A/B) Each member confirms they can `import src.config` and run on Colab GPU.
- **Definition of Done:**
  - `pip install -r requirements.txt` succeeds on a clean Colab runtime.
  - `python -c "import src.config"` works; `set_seeds()` runs.
  - `src/` tree matches `CLAUDE.md` §7; `data/` and `artifacts/` are gitignored.
  - This plan is committed; team can branch and work in parallel.

### Phase 1 — Data pipeline

- **Goal:** Produce reproducible stratified train/val/test splits over the top-~10
  `subCategory` classes and a `tf.data` pipeline yielding `((image, text), label)`.
- **Owner(s):** C (lead).
- **Dependencies:** Phase 0 (config + package skeleton).
- **Tasks:**
  - [ ] (C) Implement `src/data/download.py`: `kagglehub` download of
        `paramaggarwal/fashion-product-images-small` into `data/`; idempotent (skip if
        present); print resolved paths.
  - [ ] (C) EDA in `notebooks/exploration.ipynb`: load `styles.csv`, inspect the
        `subCategory` distribution, confirm the top-10 classes, check missing
        images / malformed rows.
  - [ ] (C) Implement `src/data/preprocess.py`: drop rows with missing image/text,
        select the top-10 `subCategory` by frequency, integer-encode labels, persist
        `CLASS_LIST`, do a **stratified 70/15/15** split with the fixed seed, and write
        split manifests (CSV) under `data/`.
  - [ ] (C) Join image paths (`images/{id}.jpg`), verify each referenced image exists,
        drop/justify orphans.
  - [ ] (C) Implement `src/data/dataset.py`: `tf.data` pipeline that decodes + resizes
        images to EfficientNet input, applies EfficientNet preprocessing, pairs with raw
        text, yields `((image, text), label)`; shuffle (train only), batch, prefetch,
        cache where safe; optional `~15–20k` subset cap via config while building.
  - [ ] (C) Build `TextVectorization` adapted on the **train split only**; persist its
        vocabulary so it is identical across all models (no leakage).
  - [ ] (C) Sanity check: pull one batch per split, print shapes/dtypes, visualize a few
        `(image, text, label)` triples in the notebook.
- **Definition of Done:**
  - `python -m src.data.download` populates `data/` and is idempotent.
  - `python -m src.data.preprocess` writes reproducible manifests (same classes/counts
    on re-run with the fixed seed).
  - `dataset.py` yields correctly-shaped `((image, text), label)` batches for all three
    splits; per-class proportions match across splits within tolerance.
  - TextVectorization vocab adapted on train only and persisted.

### Phase 2 — Single-modal baselines

- **Goal:** Build and train the image-only and text-only classifiers and record their
  test metrics as the two baselines for the comparison.
- **Owner(s):** A (image-only) & B (text-only); C provides the shared train/eval harness.
- **Dependencies:** Phase 1 (pipeline, splits, vectorizer).
- **Tasks:**
  - [ ] (A) Implement `src/models/image_branch.py`: functional
        `EfficientNetB0(weights="imagenet", include_top=False)` **frozen** →
        `GlobalAveragePooling2D` → `Dense(256, relu)`; expose **both** the 256-d encoder
        output (for fusion reuse) and an image-only softmax head.
  - [ ] (B) Implement `src/models/text_branch.py`: functional
        `TextVectorization → Embedding → GRU(128) → Dense(128, relu)`; expose the 128-d
        encoder output (for fusion) and a text-only softmax head.
  - [ ] (C) Implement `src/train.py` with `--model {image,text,fusion}`: build the
        requested model, compile (Adam + sparse categorical cross-entropy + accuracy),
        fit with EarlyStopping/checkpoint on val, and save weights + `metrics.json` +
        loss/accuracy curves to `artifacts/{model}/`.
  - [ ] (A) Train `--model image`; save weights, metrics JSON, training curves.
  - [ ] (B) Train `--model text`; save weights, metrics JSON, training curves.
  - [ ] (C) Implement `src/evaluate.py`: load a trained model, compute test
        accuracy / precision / recall / macro-F1 + a confusion-matrix plot; write to
        `artifacts/{model}/`.
  - [ ] (A/B) Evaluate both baselines on test; record numbers in a shared results table.
- **Definition of Done:**
  - `python -m src.train --model image` and `--model text` run end to end on Colab and
    save weights + `metrics.json` + curves.
  - `python -m src.evaluate --model {image,text}` produces test metrics + confusion
    matrices.
  - Both baselines' test metrics live in a shared comparison table (fusion row pending).
  - `image_branch` and `text_branch` expose reusable encoder outputs for Phase 3 (no
    duplication needed for fusion).

### Phase 3 — Fusion model

- **Goal:** Build and train the multi-modal fusion model by reusing the two branch
  encoders, then assemble the **non-negotiable 3-way comparison**.
- **Owner(s):** C (lead); A/B verify their branch wiring.
- **Dependencies:** Phase 2 (both encoders + train/eval harness + baseline metrics).
- **Tasks:**
  - [ ] (C) Implement `src/models/fusion.py`: functional model reusing the image (256-d)
        and text (128-d) encoders → `Concatenate` → `Dense(256, relu)` →
        `Dropout(0.3)` → `Dense(n_classes, softmax)`.
  - [ ] (C) Wire fusion into `src/train.py` under `--model fusion`; keep the
        EfficientNet backbone **frozen** for stage 1.
  - [ ] (C) Train `--model fusion`; save weights, metrics JSON, training curves.
  - [ ] (C) Evaluate fusion on test; produce a confusion matrix.
  - [ ] (C) **Assemble the 3-way comparison table** (image-only vs text-only vs fusion:
        accuracy + macro-F1 side by side) plus a grouped bar chart. **Core deliverable —
        must not be skipped.**
  - [ ] (A/B) Confirm fusion reuses the *identical* encoders + vectorizer as the
        baselines (apples-to-apples).
- **Definition of Done:**
  - `python -m src.train --model fusion` runs end to end and saves weights +
    `metrics.json` + curves.
  - `python -m src.evaluate --model fusion` produces test metrics + confusion matrix.
  - A single committed table + chart shows all three models' test metrics together.
  - Fusion verified to reuse the same pipeline/encoders as the baselines (fair test).

### Phase 4 — Tuning & evaluation

- **Goal:** Run a lightweight manual hyperparameter search and finalize generalization
  metrics + figures for the report.
- **Owner(s):** C (lead); A/B tune their own branches.
- **Dependencies:** Phase 3 (all three models train + evaluate).
- **Tasks:**
  - [ ] (C) Define a **small** manual HP grid in config: learning rate, batch size,
        number/size of FC fusion layers, dropout. A handful of runs — no AutoML.
  - [ ] (C) Run the lightweight search on fusion; log per-run val metrics; pick the best.
  - [ ] (A/B) Light tuning of image-only and text-only (e.g., GRU units, embedding dim,
        dropout) so baselines are fairly tuned too.
  - [ ] (C) Re-train all three with the chosen settings (fixed seed); refresh the
        comparison table + curves.
  - [ ] (C) Generate final figures: per-model training curves, confusion matrices, and
        the 3-way comparison chart.
  - [ ] (C) **(stretch)** Stage-2 fine-tuning: unfreeze top EfficientNet blocks, train at
        a low LR; keep only if it helps and time permits.
  - [ ] (C) Write a short results summary (numbers + 2–3 sentence interpretation: does
        fusion beat both baselines?).
- **Definition of Done:**
  - Best hyperparameters recorded; search runs logged.
  - Final metrics + figures regenerated for all three models with the chosen config.
  - The 3-way comparison is finalized and interpreted.
  - Stage-2 fine-tuning either done-and-kept (if it helped) or explicitly logged as
    skipped (stretch).

### Phase 5 — Deliverables

- **Goal:** Assemble the theoretical + mathematical report, finalize documented code, and
  build the presentation foregrounding the fusion comparison.
- **Owner(s):** C (lead, report assembly); A (CNN math), B (RNN math).
- **Dependencies:** Phase 4 (final metrics + figures).
- **Tasks:**
  - [ ] (A) Report — CNN section: convolution, pooling, activation functions (math);
        EfficientNet rationale; why the backbone is frozen.
  - [ ] (B) Report — RNN section: embeddings + GRU **update/reset** gate equations; final
        hidden state as the text encoding (math).
  - [ ] (C) Report — loss + optimization (cross-entropy + Adam) and fusion section
        (concatenation + why an FC layer captures image–text correlations); justify the
        held-out split over k-fold and the scope cuts.
  - [ ] (C) Insert final metrics, the comparison table, and figures; write
        results/discussion.
  - [ ] (All) Build the presentation; lead with the 3-way fusion-vs-single-modal result.
  - [ ] (C) Final code pass: docstrings, verify README run-commands end to end, confirm
        each variant saves weights + metrics + plots; tidy `artifacts/`.
  - [ ] (C) Tick remaining checkboxes and write final Status-log entries.
- **Definition of Done:**
  - Report covers every math item in `PROJECT_BRIEF` §1 (CNN conv/pool/activation; GRU
    gates; cross-entropy + Adam; fusion rationale) and justifies the scope cuts.
  - Presentation built and foregrounds the fusion comparison.
  - Code runs end to end from the README commands; all three variants save weights +
    metrics + plots.
  - This plan's checkboxes and Status log are up to date.

## File / module build order

Build bottom-up so each layer can be tested before the next depends on it:

1. `requirements.txt` + `.gitignore` update — environment first.
2. `src/config.py` — paths, seed, hyperparameters, `CLASS_LIST` placeholder (everything
   imports this).
3. `src/data/download.py` — get the dataset onto disk.
4. `src/data/preprocess.py` — filter top-10 classes, write `CLASS_LIST` + stratified
   split manifests.
5. `src/data/dataset.py` — `tf.data` pipeline + `TextVectorization`.
6. `src/models/image_branch.py` *(A)*.
7. `src/models/text_branch.py` *(B)*.
8. `src/models/fusion.py` *(C, reuses 6 & 7)*.
9. `src/train.py` — `--model {image,text,fusion}` (started in Phase 2 with image/text,
   extended in Phase 3 for fusion).
10. `src/evaluate.py` — metrics + confusion matrix.

`notebooks/exploration.ipynb` is built alongside Phase 1.

## Risks & mitigations

| Risk                                       | Mitigation                                                                                  |
|--------------------------------------------|---------------------------------------------------------------------------------------------|
| Colab timeouts / GPU disconnects           | Modest epochs; checkpoint to `artifacts/` (and Drive); train on ~15–20k subset first.       |
| Class imbalance (top-10 uneven)            | Stratified split; report **macro-F1** + confusion matrix, not just accuracy; optional class weights. |
| Low-res images (~60×80 upscaled)           | Accept per brief; document as a limitation; use EfficientNet preprocess; don't expect SOTA. |
| Fusion does **not** beat baselines         | Reuse the exact shared pipeline/encoders; tune fusion FC/dropout; if still flat, report honestly with analysis (a valid result). |
| Data leakage (vectorizer/norm on all data) | Adapt `TextVectorization` on **train only**; fixed seed; persist vocab.                      |
| Scope creep / time overrun                 | Hold to committed decisions; anything beyond minimum is **(stretch)**; prioritize the comparison over accuracy. |
| Kaggle download / auth friction on Colab   | Use `kagglehub`; document the credentials step; keep the download idempotent.               |
| 3-person merge conflicts                   | Centralized `config.py`; clear ownership (A=image, B=text, C=infra/fusion); branch per task.|

## Status log

*(Reverse-chronological. Append a dated one-line entry whenever a task completes or a
decision changes.)*

- **2026-06-04** — Created `docs/DEVELOPMENT_PLAN.md`. Phase structure approved: baselines
  in Phase 2, fusion in Phase 3, with the 3-way comparison locked as a Phase 3 DoD. Repo
  confirmed greenfield (only `CLAUDE.md`, `README.md`, `PROJECT_BRIEF.md`, `.gitignore`).
  Noted: `PROJECT_BRIEF.md` sits at the repo root but is referenced as
  `docs/PROJECT_BRIEF.md` — reconciliation added as a Phase 0 task.
