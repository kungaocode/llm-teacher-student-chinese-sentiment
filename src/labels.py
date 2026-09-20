"""Canonical label vocabulary and normalization.

The project uses a fixed 3-class coarse sentiment scheme (plan §3):

    negative / neutral / positive

Every producer — teacher free-text, dataset original labels, baseline
predictions — is normalized onto this space so all evaluations share one
label universe.
"""
from __future__ import annotations

import json
import re
from typing import Optional

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABELS)}

# Common raw tokens -> canonical label (Chinese + English + numeric).
_TOKEN_MAP: dict[str, str] = {
    # negative
    "负": "negative", "消极": "negative", "负面": "negative", "差": "negative",
    "negative": "negative", "neg": "negative", "0": "negative",
    # neutral
    "中": "neutral", "中性": "neutral", "一般": "neutral", "普通": "neutral",
    "neutral": "neutral", "neu": "neutral", "1": "neutral",
    # positive
    "正": "positive", "积极": "positive", "正面": "positive", "好": "positive",
    "positive": "positive", "pos": "positive", "2": "positive",
}

_JSON_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)
_LABEL_RE = re.compile(r"[\"']?label[\"']?\s*[:：]\s*[\"']?([^\"',}，\s]+)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason[\"']?\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)


def normalize_label(raw) -> str:
    """Normalize an arbitrary label token to one of the canonical 3 labels.

    Raises ``ValueError`` when the token cannot be mapped — callers should
    decide whether to drop or back off rather than silently guess.
    """
    if raw is None:
        raise ValueError("label is None")
    if isinstance(raw, (int, bool)) and not isinstance(raw, bool):
        raw = str(int(raw))
    key = str(raw).strip().lower()
    if key in LABELS:
        return key
    if key in _TOKEN_MAP:
        return _TOKEN_MAP[key]
    # substring fallback for prefixed/suffixed tokens like "很负面" / "positive!".
    for tok, lbl in _TOKEN_MAP.items():
        if tok in key:
            return lbl
    raise ValueError(f"cannot normalize label: {raw!r}")


def parse_teacher_json(text: str) -> tuple[str, str]:
    """Parse a teacher response into ``(label, reason)``.

    Accepts strict JSON ``{"label": "...", "reason": "..."}`` and falls back to
    a loose regex extraction over the same keys. Label is normalized to the
    canonical 3-class space.
    """
    label, reason = _try_parse_json(text)
    if label is not None:
        return normalize_label(label), (reason or "").strip()
    label, reason = _regex_fallback(text)
    if label is not None:
        return normalize_label(label), (reason or "").strip()
    raise ValueError(f"could not parse label/reason from teacher output: {text[:200]!r}")


def _try_parse_json(text: str) -> tuple[Optional[str], Optional[str]]:
    m = _JSON_BRACE_RE.search(text)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict):
        return None, None
    return obj.get("label"), obj.get("reason")


def _regex_fallback(text: str) -> tuple[Optional[str], Optional[str]]:
    label = None
    lm = _LABEL_RE.search(text)
    if lm:
        label = lm.group(1)
    reason = None
    rm = _REASON_RE.search(text)
    if rm:
        reason = rm.group(1).strip().strip('"').strip("'").rstrip(",}")
    return label, reason


def map_dataset_label(raw, scheme: str = "binary") -> str:
    """Map a dataset's native label to the canonical space.

    ``scheme="binary"`` (ChnSentiCorp): ``0 -> negative``, ``1 -> positive``.
    """
    if scheme == "binary":
        return "positive" if int(raw) == 1 else "negative"
    return normalize_label(raw)


def coerce_label(raw) -> str:
    """Best-effort map of an unknown label token to the canonical space.

    Handles binary dataset labels (``0``/``1``) via :func:`map_dataset_label`,
    and free-text / 3-class ids via :func:`normalize_label`. Raises
    ``ValueError`` if unmappable.
    """
    if isinstance(raw, bool):
        raw = int(raw)
    if isinstance(raw, int) or (isinstance(raw, str) and raw.strip() in ("0", "1")):
        return map_dataset_label(raw, "binary")
    return normalize_label(raw)
