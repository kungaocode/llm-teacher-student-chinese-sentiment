#!/usr/bin/env python3
"""Step 2: label the train pool AND the dev set with the teacher LLM.

The train-pool labels (-> ``data/processed/teacher_labels.jsonl``) are the
student's training signal; the dev labels (-> written back into
``data/splits/dev.jsonl``) give step 4 its early-stopping signal. Both are
teacher-labeled — only gold is human-labeled (plan §4.2).

Provider comes from config ``teacher.provider`` (dashscope | deepseek | moonshot
| zhipu | openai | mock). Usage is recorded per request to
``results/usage/teacher_usage.jsonl`` for the cost step (step 7).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_jsonl_dicts, write_jsonl
from src.teacher import make_teacher


def _label_one(teacher, row: dict) -> tuple[dict, object]:
    """Label a single row. Return (out_dict, usage_record)."""
    label, reason, rec = teacher.label(row["text"])
    out = {"id": row["id"], "text": row["text"], "label": label, "reason": reason}
    for k in ("source", "source_label", "domain"):
        if k in row and row[k] is not None:
            out[k] = row[k]
    return out, rec


def _label_batch(teacher, rows: list[dict], max_workers: int) -> tuple[list[dict], list]:
    """Label ``rows`` concurrently, preserving input order.

    The OpenAI client is thread-safe, so one teacher instance is shared across
    workers. A single failing row is skipped (logged) rather than aborting the run.
    """
    labeled, usage = [], []
    total = len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_label_one, teacher, row): row for row in rows}
        for i, fut in enumerate(futures, 1):
            row = futures[fut]
            try:
                out, rec = fut.result()
                labeled.append(out)
                usage.append(rec)
            except Exception as exc:  # noqa: BLE001 - skip bad rows, keep going
                print(f"  [warn] label failed id={row['id']}: {exc}", file=sys.stderr)
            if i % 100 == 0 or i == total:
                print(f"  [progress] {i}/{total}", file=sys.stderr, flush=True)
    return labeled, usage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Label train_pool + dev with the teacher LLM.")
    parser.add_argument("--workers", type=int, default=None,
                        help="concurrent requests (default: teacher.max_workers)")
    parser.add_argument("--limit", type=int, default=None,
                        help="label only the first N train_pool rows (dev is always full)")
    args = parser.parse_args(argv)

    cfg = load_config()
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    processed_dir = ROOT / cfg.get("paths.processed_dir", "data/processed")
    usage_dir = ROOT / cfg.get("paths.results_dir", "results") / "usage"
    max_workers = args.workers or int(cfg.get("teacher.max_workers", 3))

    pool_path = splits_dir / "train_pool.jsonl"
    if not pool_path.exists():
        print(f"[2_teacher_label] missing {pool_path}; run 1_prepare.py first.")
        return 1

    teacher = make_teacher(cfg)

    # 1) Train pool -> student training set.
    pool = load_jsonl_dicts(pool_path)
    if args.limit:
        pool = pool[: args.limit]
    labeled, usage = _label_batch(teacher, pool, max_workers)
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed_dir / "teacher_labels.jsonl", labeled)

    # 2) Dev set -> early-stopping labels (written back into dev.jsonl).
    n_dev = 0
    dev_path = splits_dir / "dev.jsonl"
    if dev_path.exists():
        dev = load_jsonl_dicts(dev_path)
        dev_labeled, dev_usage = _label_batch(teacher, dev, max_workers)
        by_id = {str(r["id"]): r["label"] for r in dev_labeled}
        for r in dev:
            if str(r["id"]) in by_id:
                r["label"] = by_id[str(r["id"])]
        write_jsonl(dev_path, dev)
        usage += dev_usage
        n_dev = len(dev_labeled)

    usage_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(usage_dir / "teacher_usage.jsonl", [r.__dict__ for r in usage])

    print(f"[2_teacher_label] teacher={teacher.model}  provider={cfg.get('teacher.provider')}  "
          f"workers={max_workers}")
    print(f"                  train_pool labeled: {len(labeled)} -> "
          f"{processed_dir / 'teacher_labels.jsonl'}")
    if n_dev:
        print(f"                  dev labeled:        {n_dev} -> {dev_path}")
    print(f"                  usage -> {usage_dir / 'teacher_usage.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
