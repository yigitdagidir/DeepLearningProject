# Project Brief

This file captures **what the assignment requires** and **the decisions we have already
committed to**. It is the source of truth for *why* the project is shaped the way it is.
Claude Code should treat the requirements here as fixed constraints and the decisions as
defaults that should only change with an explicit instruction.

---

## 1. Assignment requirements (from the course handout)

Build a multi-modal pipeline that processes image and text data with deep learning:

1. **Classify images** using a Convolutional Neural Network (CNN).
2. **Analyze text** descriptions using word embeddings and an RNN.
3. **Fuse** image and text information for a multi-modal model.
4. **Describe the optimal model theoretically and mathematically**, explaining each
   architectural choice.

### Mandatory pipeline steps

- **Image:** resize + normalize; pre-trained CNN (ResNet / VGG / EfficientNet) for feature
  extraction; fine-tune upper layers for the dataset classes.
- **Text:** tokenize; convert words to embedding vectors (Word2Vec / GloVe / BERT, or a
  trainable embedding); pass through an RNN (LSTM or GRU); use the final RNN state as the
  text encoding.
- **Fusion:** concatenate the CNN visual features and the RNN text encoding; pass through
  fully connected layers; final softmax over classes.
- **Tuning:** hyperparameter search (learning rate, batch size, number of FC fusion
  layers); dropout regularization; evaluate for generalization.

### Required deliverables

1. **Theoretical + mathematical report** — full description of the chosen model with the
   math of each component.
2. **Source code** — well-documented, trains and evaluates the model.
3. **Presentation** — results, discussion of model choices, optimization, and the impact
   of multi-modal fusion.

### Report must mathematically cover

- CNN: convolution, pooling, activation functions.
- RNN: LSTM/GRU activation equations and the gates (input / forget / output for LSTM;
  update / reset for GRU).
- Loss + optimization: cross-entropy loss and the optimizer's role in minimizing it.
- Fusion: the concatenation process and why a fully connected layer captures
  image–text correlations.

---

## 2. Committed decisions

These were chosen to minimize friction under a tight timeline. Treat them as defaults.

- **Dataset:** Kaggle *Fashion Product Images (small)* — chosen because it ships with
  ready-made class labels **and** text descriptions, unlike COCO/Flickr (captioning
  datasets that would require deriving labels). Filed under the handout's
  "Student-Proposed Dataset" option.
- **Classification target:** `subCategory`, restricted to the **top ~10 classes**.
- **Text field:** `productDisplayName`.
- **Image model:** EfficientNetB0, ImageNet weights, backbone frozen initially.
- **Text model:** trainable Embedding → GRU (lighter than BERT; descriptions are short).
- **Fusion:** concatenate → Dense → Dropout(0.3) → Softmax.
- **Framework:** TensorFlow / Keras, functional API.
- **Evaluation:** stratified held-out train/val/test split **instead of** k-fold CV
  (full k-fold on deep nets is too expensive for the timeline — to be justified in the
  report).
- **Three models trained** for comparison: image-only, text-only, fusion.

## 3. Scope cuts (to protect the timeline)

- Use the **small** (low-res) dataset, not the 25 GB full-resolution version.
- Work on a **subset (~15–20k images)** while building the pipeline; scale up only if time
  allows.
- Treat **stage-2 fine-tuning** (unfreezing EfficientNet blocks) as optional, attempted
  only after the full pipeline works end to end.
- Keep the hyperparameter search **small and manual / lightweight** — no expensive AutoML.

## 4. Definition of success

- All three models train end to end and produce saved metrics + plots.
- The fusion model's metrics are reported **alongside** the two single-modal baselines.
- The report covers every mathematical item in section 1.
- The presentation foregrounds the **fusion-vs-single-modal comparison.**
