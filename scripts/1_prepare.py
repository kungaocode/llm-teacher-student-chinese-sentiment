#!/usr/bin/env python3
"""Step 1: raw corpora -> clean, deterministic splits (train_pool / dev / gold).

Reads the JSONL files under ``data/raw`` (written by ``0_download.py``), cleans
+ dedupes, and writes three splits to ``data/splits``:

    train_pool.jsonl  -- {id, text, source_label, source, domain}; labels come
                         from the teacher (step 2)
    dev.jsonl         -- {id, text, source_label, source, domain, label: null}
    gold.jsonl        -- {id, text, source_label, source, domain, label: null}

``source_label`` is the ORIGINAL 2-class integer (0/1). The canonical 3-class
``label`` is intentionally left null: it is filled by the teacher (train pool)
or by teacher + human double-review (dev / gold). This keeps the 2-class public
data honest against the 3-class project target — the neutral class has no
public ground truth and must come from human review.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import (
    clean_text, find_raw_files, is_repetitive, load_jsonl_dicts,
    make_splits, make_splits_stratified, write_jsonl,
)
from src.neutral import neutral_score


def main() -> int:
    cfg = load_config()
    seed = int(cfg.get("project.seed", 42))
    raw_dir = ROOT / cfg.get("paths.raw_dir", "data/raw")
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    n_gold = int(cfg.get("data.gold_size", 200))
    n_dev = int(cfg.get("data.dev_size", 200))
    n_pool = int(cfg.get("data.train_pool_size", 2000))
    n_gold_neutral = int(cfg.get("data.gold_neutral_size", 0))

    raw_files = find_raw_files(raw_dir)
    if not raw_files:
        print(f"[1_prepare] no raw data under {raw_dir}; run 0_download.py first.")
        return 0

    # Merge all raw files into one cleaned, deduplicated pool of dicts.
    pool: list[dict] = []
    n_dropped_spam = 0
    for f in raw_files:
        for row in load_jsonl_dicts(f):
            text = clean_text(row.get("text", ""))
            if not text:
                continue
            if is_repetitive(text):
                n_dropped_spam += 1
                continue
            pool.append(
                {
                    "text": text,
                    "source_label": row.get("source_label"),
                    "source": row.get("source", f.stem),
                    "domain": row.get("domain"),
                }
            )

    print(f"[1_prepare] loaded {len(pool)} rows from {len(raw_files)} file(s) "
          f"(dropped {n_dropped_spam} repetitive/spam reviews)")

    texts = [p["text"] for p in pool]
    try:
        if n_gold_neutral > 0:
            scores = [neutral_score(t) for t in texts]
            splits = make_splits_stratified(
                texts, scores, n_gold=n_gold, n_dev=n_dev, n_pool=n_pool,
                n_gold_neutral=n_gold_neutral, seed=seed,
            )
            n_cand_gold = sum(1 for i in splits["gold"] if scores[i] > 0)
            print(f"[1_prepare] neutral-aware split: {n_cand_gold} candidate-neutral "
                  f"reviews seeded into gold (target {n_gold_neutral})")
        else:
            splits = make_splits(texts, n_gold=n_gold, n_dev=n_dev, n_pool=n_pool, seed=seed)
    except ValueError as exc:
        print(f"[1_prepare] ERROR: {exc}")
        return 1

    splits_dir.mkdir(parents=True, exist_ok=True)

    def _rows(indices: list[int], with_label_slot: bool) -> list[dict]:
        rows = []
        for i in indices:
            p = pool[i]
            row = {"id": i, "text": p["text"], "source": p["source"]}
            if p.get("source_label") is not None:
                row["source_label"] = p["source_label"]
            if p.get("domain") is not None:
                row["domain"] = p["domain"]
            if with_label_slot:
                row["label"] = None  # 3-class slot, filled by teacher+human later
            rows.append(row)
        return rows

    write_jsonl(splits_dir / "train_pool.jsonl", _rows(splits["train_pool"], with_label_slot=False))
    write_jsonl(splits_dir / "dev.jsonl", _rows(splits["dev"], with_label_slot=True))
    write_jsonl(splits_dir / "gold.jsonl", _rows(splits["gold"], with_label_slot=True))

    print("[1_prepare] wrote splits ->", splits_dir)
    print(f"            train_pool={len(splits['train_pool'])}  dev={len(splits['dev'])}  "
          f"gold={len(splits['gold'])}  (seed={seed})")
    print("            NOTE: 'label' (3-class) is null until teacher+human fill it;")
    print("                  'source_label' keeps the original 2-class value for reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
