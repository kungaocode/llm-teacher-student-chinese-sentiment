#!/usr/bin/env python3
"""Step 3: teacher–human gold loop (plan §2, §5.1).

Two modes, chosen automatically by whether ``gold.jsonl`` already carries a
3-class ``label`` (filled by human annotation; double-review is recommended):

* **Propose mode** (gold still unlabeled): the teacher proposes a 3-class label
  + reason for every gold sample and writes ``results/gold_teacher_proposals.jsonl``
  for human review. This is how the ``neutral`` class gets its ground truth —
  the public corpora are 2-class, so neutral exists only where a human confirms
  the teacher's call.

* **Agreement mode** (gold has human labels): compare teacher vs human on the
  labeled subset and gate on accuracy / Cohen's kappa before full-scale labeling.
  This mode calls the teacher API again; ``6_evaluate.py`` can score existing
  teacher proposals offline without another API call.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_jsonl_dicts, write_jsonl
from src.metrics import accuracy, classification_summary, cohen_kappa
from src.teacher import make_teacher


def _propose(teacher, gold: list[dict], results_dir: Path) -> int:
    proposals = []
    for g in gold:
        label, reason, _rec = teacher.label(g["text"])
        proposals.append({**g, "teacher_label": label, "reason": reason})
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "gold_teacher_proposals.jsonl"
    write_jsonl(out, proposals)
    print(f"[3_check_gold] PROPOSE MODE: teacher={teacher.model} proposed 3-class labels "
          f"on {len(proposals)} gold samples")
    print(f"               -> {out}")
    print("               NEXT: human-review these, collect the labels into gold.jsonl,")
    print("               then score existing proposals offline with 6_evaluate.py.")
    return 0


def _agreement(teacher, labeled: list[dict], results_dir: Path, cfg) -> int:
    human = [g["label"] for g in labeled]
    teacher_pred = [teacher.label(g["text"])[0] for g in labeled]

    acc = accuracy(human, teacher_pred)
    kappa = cohen_kappa(human, teacher_pred)
    summary = classification_summary(human, teacher_pred)

    min_acc = float(cfg.get("agreement.min_accuracy", 0.85))
    min_kappa = float(cfg.get("agreement.min_kappa", 0.6))
    report = {
        "n": len(labeled),
        "teacher_model": teacher.model,
        "accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
        "per_class_f1": {k: round(v, 4) for k, v in summary["per_class_f1"].items()},
        "thresholds": {"min_accuracy": min_acc, "min_kappa": min_kappa},
        "passed": acc >= min_acc and kappa >= min_kappa,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "teacher_agreement.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[3_check_gold] AGREEMENT: n={len(labeled)}  teacher={teacher.model}")
    print(f"               accuracy={acc:.3f}  kappa={kappa:.3f}  "
          f"per_class_f1={summary['per_class_f1']}")
    if report["passed"]:
        print("               -> gate PASSED (teacher is trusted for labeling)")
    else:
        print(f"               -> gate FAILED (need acc>={min_acc} / kappa>={min_kappa}); "
              "fix prompt/examples before full labeling")
    print(f"               -> {out}")
    return 0 if report["passed"] else 2


def main() -> int:
    cfg = load_config()
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    results_dir = ROOT / cfg.get("paths.results_dir", "results")

    gold_path = splits_dir / "gold.jsonl"
    if not gold_path.exists():
        print(f"[3_check_gold] missing {gold_path}; run 1_prepare.py first.")
        return 1

    gold = load_jsonl_dicts(gold_path)
    teacher = make_teacher(cfg)
    labeled = [g for g in gold if g.get("label")]

    if not labeled:
        return _propose(teacher, gold, results_dir)
    if len(labeled) < len(gold):
        print(f"[3_check_gold] {len(labeled)}/{len(gold)} gold samples have human labels; "
              "checking agreement on the labeled subset.")
    return _agreement(teacher, labeled, results_dir, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
