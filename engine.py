"""
The actual training / evaluation loops, kept separate from the command-line
script so that cross_validation.py and hyperparameter_search.py can reuse them.
"""

import torch
import torch.nn as nn

from .utils import AverageMeter, accuracy


def build_optimizer(cfg, model):
    params = model.trainable_parameters()
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer '{cfg.optimizer}'")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, text, labels in loader:
        images, text, labels = images.to(device), text.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images, text)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy(logits, labels), labels.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds: bool = False):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    all_preds, all_labels = [], []
    for images, text, labels in loader:
        images, text, labels = images.to(device), text.to(device), labels.to(device)
        logits = model(images, text)
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy(logits, labels), labels.size(0))
        if return_preds:
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if return_preds:
        return loss_meter.avg, acc_meter.avg, all_labels, all_preds
    return loss_meter.avg, acc_meter.avg


def fit(model, loaders, cfg, device, verbose: bool = True):
    """
    Train for cfg.epochs, keeping the weights that gave the best validation
    accuracy (a simple form of early stopping / model selection).
    Returns (history dict, best_val_acc).
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step,
                                                gamma=cfg.lr_gamma)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_state = -1.0, None

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion,
                                          optimizer, device)
        va_loss, va_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"epoch {epoch:02d}/{cfg.epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)        # restore best weights
    return history, best_val
