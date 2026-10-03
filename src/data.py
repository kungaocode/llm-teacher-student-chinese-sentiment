"""Data loading, cleaning, and deterministic split into train-pool / dev / gold.

Leak-prevention is the hard rule (plan §4.2):
    * the gold split is protected and becomes the ONLY evaluation truth after
      human review;
    * it must never enter student training (not even as unlabeled text);
    * the train pool is unlabeled and gets its labels from the teacher.

Splits are produced by a fixed seed and are deduplicated across sets so a
review appearing in gold cannot also appear in the train pool.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[^一-鿿]")


def is_repetitive(text: str, min_len: int = 6, ratio: float = 0.5) -> bool:
    """True if ``text`` is one short phrase repeated (spammy low-quality review).

    Measured as the distinct-bigram ratio of the CJK content: a review made of a
    repeated word/phrase (``一般一般一般``, ``还行吧还行吧``) has very few distinct
    bigrams relative to its length. Short texts (few CJK chars) are never flagged.
    """
    t = _CJK_RE.sub("", text)
    if len(t) < min_len:
        return False
    bigrams = [t[i:i + 2] for i in range(len(t) - 1)]
    return len(set(bigrams)) / len(bigrams) < ratio


def clean_text(text: str) -> str:
    """Normalize whitespace, strip BOM/surrounding noise. Returns "" if empty."""
    if text is None:
        return ""
    return _WS_RE.sub(" ", str(text).replace("﻿", "")).strip()


def _normalized_key(text: str) -> str:
    return clean_text(text).lower()


def load_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> tuple[list[str], Optional[list[str]]]:
    """Load ``text`` (+ optional ``label``) columns from a CSV."""
    texts: list[str] = []
    labels: list[str] = []
    has_label = label_col is not None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = clean_text(row.get(text_col, ""))
            if not t:
                continue
            texts.append(t)
            if has_label and label_col in row:
                labels.append(row[label_col])
            elif has_label:
                labels = []  # label column missing -> treat as unlabeled
                has_label = False
    return texts, (labels if has_label else None)


def load_jsonl(
    path: str | Path,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> tuple[list[str], Optional[list[str]]]:
    """Load ``text`` (+ optional ``label``) fields from a JSONL file."""
    texts: list[str] = []
    labels: list[str] = []
    has_label = label_col is not None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = clean_text(obj.get(text_col, ""))
            if not t:
                continue
            texts.append(t)
            if has_label and label_col in obj:
                labels.append(obj[label_col])
            elif has_label:
                has_label = False
    return texts, (labels if has_label else None)


def load_raw(
    path: str | Path,
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> tuple[list[str], Optional[list[str]]]:
    """Dispatch to ``load_csv`` / ``load_jsonl`` by file extension."""
    p = Path(path)
    if p.suffix.lower() in (".csv", ".tsv"):
        return load_csv(p, text_col, label_col)
    if p.suffix.lower() in (".jsonl", ".ndjson", ".json"):
        return load_jsonl(p, text_col, label_col)
    raise ValueError(f"unsupported raw data format: {p.suffix!r} (use .csv/.jsonl)")


def find_raw_files(raw_dir: str | Path) -> list[Path]:
    """Return supported raw data files under ``raw_dir``."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return []
    exts = {".csv", ".tsv", ".jsonl", ".ndjson"}
    return sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def dedupe_indices(texts: list[str]) -> list[int]:
    """Indices of first occurrence of each unique text (keeps original order)."""
    seen: set[str] = set()
    out: list[int] = []
    for i, t in enumerate(texts):
        k = _normalized_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(i)
    return out


def make_splits(
    texts: list[str],
    n_gold: int,
    n_dev: int,
    n_pool: Optional[int] = None,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split *deduplicated* indices into ``gold`` / ``dev`` / ``train_pool``.

    The rng shuffle uses ``seed`` so the assignment is reproducible. ``gold`` is
    drawn first (it is the protected truth), then ``dev``, then the remainder
    (capped at ``n_pool``) becomes the train pool.

    Returns ``{"gold": [...], "dev": [...], "train_pool": [...]}`` as index lists.
    """
    idx = np.asarray(dedupe_indices(texts), dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idx))
    idx = idx[perm]

    need = n_gold + n_dev
    if len(idx) < need:
        raise ValueError(
            f"not enough unique texts: need {need} (gold {n_gold} + dev {n_dev}), "
            f"got {len(idx)}"
        )
    gold = idx[:n_gold].tolist()
    dev = idx[n_gold:n_gold + n_dev].tolist()
    rest = idx[n_gold + n_dev:]
    if n_pool is not None:
        rest = rest[:n_pool]
    return {"gold": gold, "dev": dev, "train_pool": rest.tolist()}


def make_splits_stratified(
    texts: list[str],
    neutral_scores: list[int],
    n_gold: int,
    n_dev: int,
    n_pool: Optional[int] = None,
    n_gold_neutral: int = 0,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Like :func:`make_splits` but reserves ``n_gold_neutral`` neutral-candidate
    slots in gold (plan §4.2: the neutral class has no public ground truth, so
    gold must be seeded with plausible-neutral reviews for human confirmation).

    ``neutral_scores`` is aligned with ``texts``; ``>0`` marks a candidate and the
    magnitude ranks it (higher = more likely neutral, see ``src.neutral``). Gold
    takes the top ``n_gold_neutral`` candidates by score plus random non-candidates
    to fill ``n_gold``. Leftover candidates rejoin the train pool so nothing is
    dropped. All sets stay disjoint and deduplicated.
    """
    idx = np.asarray(dedupe_indices(texts), dtype=np.int64)
    scores = np.asarray(neutral_scores, dtype=np.int64)
    if len(scores) != len(texts):
        raise ValueError(
            f"neutral_scores length {len(scores)} != texts length {len(texts)}"
        )
    scores = scores[idx]

    cand_mask = scores > 0
    cand_idx = idx[cand_mask]
    # Stable sort by score desc -> strongest neutral signals first.
    cand_idx = cand_idx[np.argsort(-scores[cand_mask], kind="stable")]
    rest_idx = idx[~cand_mask]

    rng = np.random.default_rng(seed)
    rng.shuffle(rest_idx)

    n_gold_neutral = min(n_gold_neutral, n_gold, len(cand_idx))
    gold_neutral = cand_idx[:n_gold_neutral].tolist()
    gold_rest = rest_idx[: n_gold - n_gold_neutral].tolist()
    gold = gold_neutral + gold_rest
    rng.shuffle(gold)  # cosmetic: don't clump neutral at the top

    rest_remaining = rest_idx[n_gold - n_gold_neutral:]
    cand_remaining = cand_idx[n_gold_neutral:]

    if len(rest_remaining) < n_dev:
        raise ValueError(
            f"not enough non-neutral texts for dev: need {n_dev}, got {len(rest_remaining)}"
        )
    dev = rest_remaining[:n_dev].tolist()

    train = np.concatenate([rest_remaining[n_dev:], cand_remaining])
    rng.shuffle(train)
    if n_pool is not None:
        train = train[:n_pool]

    return {"gold": gold, "dev": dev, "train_pool": train.tolist()}


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write a list of dicts as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_jsonl_dicts(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
