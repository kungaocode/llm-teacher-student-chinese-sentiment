#!/usr/bin/env python3
"""Score cloud-model predictions on the exported test set (finetune_test.jsonl).

Usage:
    python3 scripts/score_test.py results/predictions/my_model.jsonl

Each prediction line: {"id": <int or str>, "label": "<positive|neutral|negative>"}
(or {"id": ..., "prediction": ...}). Prints overall accuracy, macro-F1, per-class
F1 and per-hard-case-category accuracy, so you can judge where the model fails
before you ship it.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_jsonl_dicts
from src.labels import coerce_label
from src.metrics import classification_summary

_CLASSES = ("positive", "neutral", "negative")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", help="JSONL: {id, label} per line")
    ap.add_argument("--test", default=None,
                    help="override test file (default data/processed/finetune_test.jsonl)")
    ap.add_argument("--meta", default=None,
                    help="override composition file (default results/test_composition.jsonl)")
    args = ap.parse_args()

    test_path = Path(args.test or ROOT / "data/processed/finetune_test.jsonl")
    meta_path = Path(args.meta or ROOT / "results/test_composition.jsonl")

    # Test JSONL holds only messages; its row ORDER matches the composition file
    # (both exported from gold in the same order), so we align by position and get
    # the id back from the composition metadata.
    test_rows = load_jsonl_dicts(test_path)
    meta = load_jsonl_dicts(meta_path) if meta_path.exists() else [{}] * len(test_rows)
    if len(test_rows) != len(meta):
        print(f"[score_test] WARNING: test ({len(test_rows)}) vs meta ({len(meta)}) length mismatch; "
              "per-category scores may be misaligned.")

    preds = {str(r["id"]): coerce_label(r.get("label") or r.get("prediction"))
             for r in load_jsonl_dicts(args.predictions)}

    y_true, y_pred, rows = [], [], []
    missing = 0
    for test_row, m in zip(test_rows, meta):
        label = coerce_label(test_row["messages"][2]["content"])  # truth = assistant content
        gid = str(m.get("id", len(y_true)))
        p = preds.get(gid)
        if p is None:
            missing += 1
            continue
        y_true.append(label)
        y_pred.append(p)
        rows.append((m, label, p))
    if missing:
        print(f"[score_test] {missing} test rows unmatched by id -> skipped.")

    if not rows:
        print("[score_test] no matched predictions; check id format in", args.predictions)
        return 1

    s = classification_summary(y_true, y_pred)
    print(f"[score_test] n={len(rows)}  accuracy={s['accuracy']:.4f}  macro-F1={s['macro_f1']:.4f}")
    print("  per-class F1:", {k: round(v, 4) for k, v in s["per_class_f1"].items()})
    print("  support:", s["support"])

    cat = defaultdict(lambda: [0, 0])
    for m, t, p in rows:
        for c in m.get("categories", ["plain"]):
            cat[c][1] += 1
            if t == p:
                cat[c][0] += 1
    print("  per-category accuracy (on this model's predictions):")
    for c, (ok, n) in sorted(cat.items(), key=lambda kv: -kv[1][1]):
        print(f"    {c:20s} {ok:4d}/{n:<4d} = {ok / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
