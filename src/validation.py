"""Validation helpers for the public two-class labels.

The project target is three-class sentiment, but the source corpora only carry
binary labels. These helpers deliberately treat those source labels as an
external, noisy check rather than human gold. ``neutral`` predictions are
reported as abstentions, so coverage and covered accuracy remain visible.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .labels import coerce_label


def source_label_to_binary(raw) -> Optional[int]:
    """Map a source-corpus label to ``0`` (negative) or ``1`` (positive).

    ``None`` is preserved. Unsupported numeric values raise ``ValueError``
    instead of being silently mapped to a class.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        if raw not in (0, 1):
            raise ValueError(f"unsupported binary source label: {raw!r}")
        return raw
    if isinstance(raw, str) and raw.strip() in ("0", "1"):
        return int(raw.strip())

    label = coerce_label(raw)
    if label == "negative":
        return 0
    if label == "positive":
        return 1
    return None


def prediction_to_binary(raw) -> Optional[int]:
    """Map a three-class prediction to binary; ``neutral`` becomes ``None``."""
    if raw is None:
        return None
    label = coerce_label(raw)
    if label == "negative":
        return 0
    if label == "positive":
        return 1
    return None


def fallback_abstentions(
    primary: Sequence,
    fallback: Sequence,
) -> list[str]:
    """Fill primary ``neutral`` predictions from a binary fallback predictor.

    A neutral fallback prediction remains neutral because there is no binary
    information to add. This is a deployment diagnostic, not a new gold label.
    """
    if len(primary) != len(fallback):
        raise ValueError(
            f"primary length {len(primary)} != fallback length {len(fallback)}"
        )

    out: list[str] = []
    for pred, fb in zip(primary, fallback):
        pred_label = coerce_label(pred) if pred is not None else "neutral"
        fb_label = coerce_label(fb) if fb is not None else "neutral"
        if pred_label == "neutral" and fb_label in ("negative", "positive"):
            out.append(fb_label)
        else:
            out.append(pred_label)
    return out


def validate_binary(y_true: Sequence, y_pred: Sequence) -> dict:
    """Summarize binary predictions while counting neutral as uncovered.

    ``accuracy_neutral_as_error`` treats every neutral prediction as incorrect
    over the full set. ``covered_accuracy`` scores only non-neutral predictions.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length {len(y_true)} != y_pred length {len(y_pred)}"
        )
    if not y_true:
        raise ValueError("cannot validate an empty sequence")

    targets = [source_label_to_binary(v) for v in y_true]
    for i, target in enumerate(targets):
        if target is None:
            raise ValueError(f"source target at index {i} is not binary")
    binary_preds = [prediction_to_binary(v) for v in y_pred]

    n = len(targets)
    true_distribution = {
        "negative": sum(1 for value in targets if value == 0),
        "positive": sum(1 for value in targets if value == 1),
    }
    prediction_distribution = {"negative": 0, "neutral": 0, "positive": 0}
    for pred in binary_preds:
        if pred is None:
            prediction_distribution["neutral"] += 1
        elif pred == 0:
            prediction_distribution["negative"] += 1
        else:
            prediction_distribution["positive"] += 1

    correct = sum(
        target == pred
        for target, pred in zip(targets, binary_preds)
        if pred is not None
    )
    covered = sum(1 for pred in binary_preds if pred is not None)
    recalls = {}
    for index, label in ((0, "negative"), (1, "positive")):
        denominator = true_distribution[label]
        recalls[label] = (
            sum(
                target == index and pred == index
                for target, pred in zip(targets, binary_preds)
            )
            / denominator
            if denominator
            else 0.0
        )

    confusion = {}
    for index, label in ((0, "negative"), (1, "positive")):
        matching = [
            pred for target, pred in zip(targets, binary_preds) if target == index
        ]
        confusion[label] = {
            "negative": sum(1 for pred in matching if pred == 0),
            "positive": sum(1 for pred in matching if pred == 1),
            "abstain": sum(1 for pred in matching if pred is None),
        }

    return {
        "n": n,
        "coverage_count": covered,
        "coverage": covered / n,
        "covered_accuracy": correct / covered if covered else 0.0,
        "accuracy_neutral_as_error": correct / n,
        "balanced_recall": (recalls["negative"] + recalls["positive"]) / 2,
        "recall": recalls,
        "true_distribution": true_distribution,
        "prediction_distribution": prediction_distribution,
        "confusion": confusion,
    }
