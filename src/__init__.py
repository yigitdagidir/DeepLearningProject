"""Multi-modal fashion product classifier (image + text -> subCategory).

Package layout:
    config            - paths, hyperparameters, class list, seed (import this everywhere)
    data.download     - kagglehub dataset download
    data.preprocess   - top-N class filter + stratified 70/15/15 split manifests
    data.dataset      - tf.data pipeline yielding ((image, text), label) + vectorizer
    models.image_branch / text_branch / fusion - the three encoders + heads
    train             - `python -m src.train --model {image,text,fusion}`
    evaluate          - test metrics + confusion matrix
    compare           - assembles the 3-way comparison table + chart
"""
