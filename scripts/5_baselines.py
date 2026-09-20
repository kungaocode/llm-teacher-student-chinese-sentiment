#!/usr/bin/env python3
"""Step 5: traditional baseline — TF-IDF (char/word n-grams) + LinearSVC.

Trains on a labeled set (default order: teacher-labeled train pool, else
``dev.jsonl``) and evaluates on the gold set. Writes predictions and a usage
record (for the cost step) to ``results/``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.cost import UsageRecord, usage_to_dict
from src.data import load_jsonl_dicts, write_jsonl
from src.labels import coerce_label
from src.metrics import classification_summary

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
except ImportError:
    raise SystemExit(
        "[5_baselines] scikit-learn is not installed. Run: pip install scikit-learn"
    )


def _load_labeled(path: Path) -> tuple[list[int], list[str], list[str]]:
    rows = [r for r in load_jsonl_dicts(path) if r.get("label") is not None]
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    labels = [coerce_label(r["label"]) for r in rows]
    return ids, texts, labels


def main() -> int:
    cfg = load_config()
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    processed_dir = ROOT / cfg.get("paths.processed_dir", "data/processed")
    results_dir = ROOT / cfg.get("paths.results_dir", "results")
    usage_dir = results_dir / "usage"

    # Pick a labeled training set.
    candidates = [
        processed_dir / "teacher_labels.jsonl",
        splits_dir / "dev.jsonl",
    ]
    train_path = next((p for p in candidates if p.exists()), None)
    gold_path = splits_dir / "gold.jsonl"
    if train_path is None:
        print("[5_baselines] no labeled training set (teacher_labels.jsonl / dev.jsonl); "
              "run 1_prepare.py (+ 2_teacher_label.py) first.")
        return 1
    if not gold_path.exists():
        print(f"[5_baselines] missing {gold_path}; run 1_prepare.py first.")
        return 1

    train_ids, train_texts, train_labels = _load_labeled(train_path)
    # Predict on ALL gold rows even before human labels exist; summary only when
    # the full gold set is labeled (score against results/test_composition.jsonl
    # or later human gold via scripts/score_test.py).
    gold = load_jsonl_dicts(gold_path)
    gold_texts = [g["text"] for g in gold]
    gold_ids = [g["id"] for g in gold]
    gold_labels = [coerce_label(g["label"]) for g in gold if g.get("label") is not None]

    max_feat = int(cfg.get("baseline.tfidf_max_features", 20000))
    ngram = tuple(int(x) for x in cfg.get("baseline.tfidf_ngram_range", [2, 4]))
    analyzer = cfg.get("baseline.tfidf_analyzer", "char_wb")
    C = float(cfg.get("baseline.svc_C", 1.0))

    print(f"[5_baselines] train on {train_path.name} (n={len(train_texts)}), "
          f"eval on gold (n={len(gold_texts)})")

    vectorizer = TfidfVectorizer(
        analyzer=analyzer, max_features=max_feat, ngram_range=ngram,
    )
    t0 = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    clf = LinearSVC(C=C, random_state=int(cfg.get("project.seed", 42)), max_iter=20000)
    clf.fit(X_train, train_labels)
    X_gold = vectorizer.transform(gold_texts)
    preds = clf.predict(X_gold)
    elapsed = time.time() - t0

    if len(gold_labels) == len(gold):
        summary = classification_summary(gold_labels, preds)
        print(f"[5_baselines] accuracy={summary['accuracy']:.3f}  "
              f"macro_f1={summary['macro_f1']:.3f}  per_class={summary['per_class_f1']}")
    else:
        print(f"[5_baselines] gold has no human labels yet ({len(gold_labels)}/{len(gold)}); "
              "wrote predictions for scoring via scripts/score_test.py")

    results_dir.mkdir(parents=True, exist_ok=True)
    usage_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        results_dir / "baseline_predictions.jsonl",
        [{"id": i, "text": t, "label": p} for i, t, p in zip(gold_ids, gold_texts, preds)],
    )
    rec = UsageRecord(model="tfidf-svc", n_predictions=len(gold_texts), seconds=elapsed)
    write_jsonl(usage_dir / "baseline_usage.jsonl", [usage_to_dict(rec)])
    print(f"[5_baselines] -> results/baseline_predictions.jsonl, usage/ (fit+infer {elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
