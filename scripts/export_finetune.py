#!/usr/bin/env python3
"""Export teacher-labeled data as OpenAI-style chat JSONL for cloud fine-tuning.

Three files, one line per example:

    {"messages": [{"role": "system", "content": "You are a helpful assistant"},
                  {"role": "user", "content": "<task prompt + review text>"},
                  {"role": "assistant", "content": "<label>"}]}

    finetune_train.jsonl  <- teacher-labeled train pool (2,000)
    finetune_dev.jsonl    <- teacher-labeled dev set     (200)
    finetune_test.jsonl   <- gold set (200), ground truth =
                             human gold label if present, else teacher proposal

The test set is the held-out gold set (disjoint from train/dev) plus a sidecar
``results/test_composition.jsonl`` tagging every test row with hard-case
categories (implicit negation / sarcasm-ish / mixed pos+neg / neutral-candidate)
so per-category accuracy can be computed after cloud inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_jsonl_dicts, write_jsonl
from src.labels import coerce_label
from src.neutral import hint_counts

PROMPT = "请判断下面评论的情感：\n{text}\n\n情感标签（positive/neutral/negative）："
SYSTEM = "You are a helpful assistant"

# Hard-case category markers (heuristic tags for the report, never the label).
_IMPLICIT_NEG = ("不太", "不怎么", "没那么", "谈不上", "说不上", "一般般", "勉强", "算不上", "还行", "还可以")
_SARCASM = ("呵呵", "真是", "好家伙", "服了", "牛啊", "真棒", "太好了", "厉害", "哈哈", "惊喜", "感谢")
_COLLOQUIAL = ("555", "emmm", "yyds", "绝绝子", "无语", "坑爹", "gg", "靠", "吐槽", "666")


def _categories(text: str) -> list[str]:
    n, p, neg = hint_counts(text)
    tags = []
    if any(w in text for w in _IMPLICIT_NEG):
        tags.append("implicit_negation")
    if any(w in text for w in _SARCASM):
        tags.append("sarcasm_candidate")
    if any(w in text for w in _COLLOQUIAL):
        tags.append("colloquial")
    if p > 0 and neg > 0:
        tags.append("mixed_pos_neg")
    if n > 0:
        tags.append("neutral_candidate")
    if not tags:
        tags.append("plain")
    return tags


def _messages(text: str, label: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(text=text)},
        {"role": "assistant", "content": label},
    ]


def _load_rows(path: Path) -> list[dict]:
    return [r for r in load_jsonl_dicts(path) if r.get("text")]


def _proposal_labels(results_dir: Path) -> dict[str, str]:
    prop = results_dir / "gold_teacher_proposals.jsonl"
    if not prop.exists():
        return {}
    return {str(r["id"]): coerce_label(r["teacher_label"]) for r in _load_rows(prop)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None, help="overwrite output dir (default data/processed)")
    args = parser.parse_args()

    cfg = load_config()
    processed_dir = ROOT / cfg.get("paths.processed_dir", "data/processed")
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    results_dir = ROOT / cfg.get("paths.results_dir", "results")
    if args.out_dir:
        processed_dir = Path(args.out_dir)

    # --- train -----------------------------------------------------------------
    train = _load_rows(processed_dir / "teacher_labels.jsonl")
    train = [r for r in train if r.get("label") is not None]
    # --- dev -------------------------------------------------------------------
    dev = _load_rows(splits_dir / "dev.jsonl")
    dev = [r for r in dev if r.get("label") is not None]
    # --- test (gold): prefer human label, fall back to teacher proposal ---------
    gold = _load_rows(splits_dir / "gold.jsonl")
    prop = _proposal_labels(results_dir)

    test_rows = []
    test_meta = []
    for g in gold:
        gid = str(g["id"])
        label = coerce_label(g["label"]) if g.get("label") else prop.get(gid)
        if label is None:
            print(f"[export_finetune] WARNING: no label for gold id={gid}; run 3_check_gold.py first.")
            continue
        test_rows.append({"id": g["id"], "text": g["text"], "label": label})
        test_meta.append({
            "id": g["id"], "label": label,
            "source": g.get("source", ""), "domain": g.get("domain", ""),
            "source_label": g.get("source_label"),
            "categories": _categories(g["text"]),
            "truth": "human" if g.get("label") else "teacher",
        })

    def write(rows: list[dict], name: str) -> Path:
        out = processed_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": _messages(r["text"], coerce_label(r["label"]))},
                                   ensure_ascii=False) + "\n")
        print(f"[export_finetune] {name:24s} n={len(rows):5d} -> {out}")
        return out

    write(train, "finetune_train.jsonl")
    write(dev, "finetune_dev.jsonl")
    write(test_rows, "finetune_test.jsonl")

    # --- test composition report ------------------------------------------------
    meta_out = results_dir / "test_composition.jsonl"
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(meta_out, test_meta)
    print(f"[export_finetune] test metadata                    -> {meta_out}")

    label_dist = Counter(m["label"] for m in test_meta)
    cat_dist = Counter(c for m in test_meta for c in m["categories"])
    truth_dist = Counter(m["truth"] for m in test_meta)
    print(f"\n[export_finetune] TEST composition (n={len(test_meta)}):")
    print("  label:", dict(label_dist))
    print("  truth-source:", dict(truth_dist))
    print("  categories:", dict(cat_dist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
