#!/usr/bin/env python3
"""Step 6: evaluate model predictions against the human gold set.

Reads the gold truth plus prediction files under ``results/predictions/`` and
top-level ``results/*_predictions.jsonl``. Existing teacher proposals are also
scored as a prediction source, without calling the teacher API again. Writes
``results/metrics.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_jsonl_dicts
from src.labels import coerce_label
from src.metrics import accuracy, classification_summary, cohen_kappa


def _load_gold(path: Path) -> tuple[list[int], list[str]]:
    rows = [g for g in load_jsonl_dicts(path) if g.get("label") is not None]
    ids = [g["id"] for g in rows]
    labels = [coerce_label(g["label"]) for g in rows]
    return ids, labels


def main() -> int:
    cfg = load_config()
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    results_dir = ROOT / cfg.get("paths.results_dir", "results")
    preds_dir = results_dir / "predictions"

    gold_path = splits_dir / "gold.jsonl"
    if not gold_path.exists():
        print(f"[6_evaluate] missing {gold_path}; run 1_prepare.py first.")
        return 1

    gold_ids, gold_labels = _load_gold(gold_path)
    gold_map = dict(zip(gold_ids, gold_labels))

    pred_files = sorted(preds_dir.glob("*.jsonl")) if preds_dir.exists() else []
    pred_files.extend(sorted(results_dir.glob("*_predictions.jsonl")))

    if not pred_files:
        print("[6_evaluate] no prediction files found under results/; "
              "run 5_baselines.py / 2_teacher_label.py / 4_train_student.py first.")
        return 1

    metrics = {}
    proposal_path = results_dir / "gold_teacher_proposals.jsonl"
    if proposal_path.exists():
        proposal_rows = [
            {"id": row["id"], "label": row.get("teacher_label")}
            for row in load_jsonl_dicts(proposal_path)
            if row.get("teacher_label")
        ]
        teacher_model = str(cfg.get("teacher.model", "teacher")).replace("/", "_")
        pred_files.append((teacher_model, proposal_rows))

    for pf in pred_files:
        if isinstance(pf, tuple):
            model, rows = pf
            file_name = proposal_path.name
        else:
            rows = load_jsonl_dicts(pf)
            model = pf.stem.replace("_predictions", "").replace("predictions_", "")
            file_name = pf.name
        pred_map = {r["id"]: coerce_label(r["label"]) for r in rows if r.get("label") is not None}
        # Evaluate on gold ids that this model predicted (ordered by gold).
        common = [i for i in gold_ids if i in pred_map]
        if not common:
            print(f"[6_evaluate] {file_name}: no overlapping ids with gold; skipped.")
            continue
        y_true = [gold_map[i] for i in common]
        y_pred = [pred_map[i] for i in common]
        s = classification_summary(y_true, y_pred)
        metrics[model] = {
            "file": file_name,
            "n": len(common),
            "accuracy": round(s["accuracy"], 4),
            "macro_f1": round(s["macro_f1"], 4),
            "cohen_kappa": round(cohen_kappa(y_true, y_pred), 4),
            "per_class_f1": {k: round(v, 4) for k, v in s["per_class_f1"].items()},
        }
        print(f"[6_evaluate] {model:12s}  n={len(common):4d}  acc={s['accuracy']:.3f}  "
              f"macro_f1={s['macro_f1']:.3f}  kappa={cohen_kappa(y_true, y_pred):.3f}")

    out = results_dir / "metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[6_evaluate] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
