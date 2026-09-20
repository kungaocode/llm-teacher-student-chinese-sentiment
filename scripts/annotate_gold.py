#!/usr/bin/env python3
"""Human gold-annotation workflow (plan §4.2): export / collect.

The gold set is the ONLY evaluation truth and must be human-labeled with double
review. This script is the human-in-the-loop glue around ``data/splits/gold.jsonl``:

  --export   gold.jsonl -> data/annotations/gold_annotation.csv
             columns: id, text, source, domain, source_label, teacher_suggest,
                      annotator_a, annotator_b, final_label, note
             (opens cleanly in Excel/Sheets; two annotators fill A/B independently)

  --collect  read the filled CSV -> write the verified 3-class ``label`` back into
             gold.jsonl; compute inter-annotator Cohen's kappa; list disagreements
             that still need adjudication.

Label vocabulary (src.labels.LABELS): positive / neutral / negative.
Annotators may also write 正面/负面/中性 or 好/差/一般 etc.; numeric 0/1/2 is
REJECTED (ambiguous vs. the source binary 0/1) — use words.

Anchor note: ``teacher_suggest`` is the teacher's proposal from step 3 (propose
mode). Annotators should read the TEXT first and form their own call, then
optionally compare — never let the suggestion become the "answer".
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_jsonl_dicts, write_jsonl
from src.labels import LABELS, normalize_label
from src.metrics import cohen_kappa

_FIELDNAMES = [
    "id", "text", "source", "domain", "source_label",
    "teacher_suggest", "annotator_a", "annotator_b", "final_label", "note",
]
_LEGEND = (
    "标注词汇表: positive(正面) / neutral(中性) / negative(负面) —— "
    "不要写 0/1/2（与源二分类混淆）。两人独立标注 annotator_a / annotator_b，"
    "意见不一致时在 final_label 填讨论后的定论。"
)


def _export(cfg) -> int:
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    results_dir = ROOT / cfg.get("paths.results_dir", "results")
    out_dir = ROOT / "data" / "annotations"

    gold_path = splits_dir / "gold.jsonl"
    if not gold_path.exists():
        print(f"[annotate_gold] missing {gold_path}; run 1_prepare.py first.")
        return 1

    gold = load_jsonl_dicts(gold_path)

    # Join teacher proposals if step 3 (propose mode) has run.
    suggest: dict[str, str] = {}
    prop_path = results_dir / "gold_teacher_proposals.jsonl"
    if prop_path.exists():
        for p in load_jsonl_dicts(prop_path):
            if p.get("teacher_label"):
                suggest[str(p["id"])] = f"{p['teacher_label']} — {p.get('reason', '')}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gold_annotation.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        w.writeheader()
        for g in gold:
            w.writerow({
                "id": g["id"],
                "text": g["text"],
                "source": g.get("source", ""),
                "domain": g.get("domain", ""),
                "source_label": g.get("source_label", ""),
                "teacher_suggest": suggest.get(str(g["id"]), ""),
                "annotator_a": "",
                "annotator_b": "",
                "final_label": "",
                "note": "",
            })

    print(f"[annotate_gold] EXPORT {len(gold)} gold rows -> {out}")
    print(f"               {_LEGEND}")
    return 0


def _clean(v: str) -> str:
    return (v or "").strip()


def _norm(v: str) -> str:
    """Normalize an annotator token, rejecting bare 0/1/2."""
    v = _clean(v)
    if not v:
        return ""
    if v in ("0", "1", "2"):
        raise ValueError(
            f"numeric label '{v}' is ambiguous (source 0/1 != 3-class 0/1/2); "
            "write positive/neutral/negative or 正面/中性/负面"
        )
    return normalize_label(v)


def _collect(cfg) -> int:
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    out_dir = ROOT / "data" / "annotations"
    csv_path = out_dir / "gold_annotation.csv"
    gold_path = splits_dir / "gold.jsonl"

    if not csv_path.exists():
        print(f"[annotate_gold] missing {csv_path}; run --export first.")
        return 1
    if not gold_path.exists():
        print(f"[annotate_gold] missing {gold_path}.")
        return 1

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    gold = load_jsonl_dicts(gold_path)
    by_id = {str(g["id"]): g for g in gold}

    a_lab, b_lab = [], []
    disagreements = []
    n_final = 0
    n_skip = 0

    for row in rows:
        try:
            a = _norm(row.get("annotator_a", ""))
            b = _norm(row.get("annotator_b", ""))
            final = _norm(row.get("final_label", ""))
        except ValueError as exc:
            print(f"[annotate_gold] row id={row.get('id')}: {exc}")
            return 1

        if a and b:
            a_lab.append(a)
            b_lab.append(b)

        if not final:
            if a and b and a == b:
                final = a
            elif a and b:  # genuine disagreement, needs adjudication
                disagreements.append((row["id"], row["text"][:40], a, b))
                continue
            elif a:
                final = a
            elif b:
                final = b

        if final:
            rid = str(row["id"]).strip()
            if rid in by_id:
                by_id[rid]["label"] = final
                n_final += 1
        else:
            n_skip += 1

    kappa = cohen_kappa(a_lab, b_lab) if a_lab else float("nan")
    n_both = len(a_lab)

    write_jsonl(gold_path, gold)

    print(f"[annotate_gold] COLLECT: {n_final}/{len(gold)} gold rows got a verified 3-class label")
    print(f"               double-annotated rows: {n_both}  Cohen's kappa = {kappa:.3f}")
    if disagreements:
        print(f"               UNRESOLVED disagreements ({len(disagreements)}):")
        for did, txt, x, y in disagreements:
            print(f"                 id={did}  A={x}  B={y}  | {txt}")
    if n_skip:
        print(f"               skipped (no label yet): {n_skip}")
    print(f"               -> wrote labels into {gold_path}")
    print("               (rerun 3_check_gold.py now to run the teacher–human agreement gate)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Human gold-annotation workflow")
    parser.add_argument("--export", action="store_true", help="gold.jsonl -> annotation CSV")
    parser.add_argument("--collect", action="store_true", help="CSV -> write labels back + kappa")
    args = parser.parse_args()

    cfg = load_config()
    if args.export:
        return _export(cfg)
    if args.collect:
        return _collect(cfg)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
