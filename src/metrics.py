"""Evaluation metrics on the canonical 3-class label space.

All metrics accept sequences of label strings or ints mapping onto ``LABELS``.
macro-F1 is the headline number because classes may be imbalanced (plan §2).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .labels import LABELS, LABEL2ID


def _to_ids(y: Sequence, labels: Sequence[str] = LABELS) -> np.ndarray:
    out = np.empty(len(y), dtype=np.int64)
    for i, v in enumerate(y):
        if isinstance(v, (int, np.integer)):
            out[i] = int(v)
        else:
            out[i] = LABEL2ID[str(v)]
    return out


def accuracy(y_true, y_pred) -> float:
    yt = _to_ids(y_true)
    yp = _to_ids(y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


def per_class_f1(y_true, y_pred, labels: Sequence[str] = LABELS) -> dict[str, float]:
    yt = _to_ids(y_true)
    yp = _to_ids(y_pred)
    out: dict[str, float] = {}
    for c in range(len(labels)):
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[labels[c]] = (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return out


def macro_f1(y_true, y_pred, labels: Sequence[str] = LABELS) -> float:
    """Macro-averaged F1 (equal weight per class, even if unseen)."""
    per = per_class_f1(y_true, y_pred, labels)
    return float(np.mean(list(per.values())))


def classification_summary(y_true, y_pred, labels: Sequence[str] = LABELS) -> dict:
    """Return ``{accuracy, macro_f1, per_class_f1, support}``."""
    yt = _to_ids(y_true)
    support = {labels[c]: int(np.sum(yt == c)) for c in range(len(labels))}
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class_f1": per_class_f1(y_true, y_pred, labels),
        "support": support,
    }


def cohen_kappa(y1, y2, labels: Sequence[str] = LABELS) -> float:
    """Cohen's kappa for teacher–human agreement (plan §2)."""
    a = _to_ids(y1)
    b = _to_ids(y2)
    n = len(a)
    if n == 0:
        return 0.0
    po = float(np.mean(a == b))
    pe = 0.0
    for c in range(len(labels)):
        pe += float(np.mean(a == c)) * float(np.mean(b == c))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def confusion_matrix(y_true, y_pred, labels: Sequence[str] = LABELS) -> list[list[int]]:
    """Return a ``len(labels) x len(labels)`` integer matrix (rows = true)."""
    yt = _to_ids(y_true)
    yp = _to_ids(y_pred)
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(yt, yp):
        cm[int(t)][int(p)] += 1
    return cm
