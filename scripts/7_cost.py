#!/usr/bin/env python3
"""Step 7: aggregate usage logs -> cost table (CNY & seconds per 1,000 preds).

Reads every ``results/usage/*.jsonl`` (one ``UsageRecord`` per line), sums them
per model, and prints + writes a unified cost table (plan §5.4).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.cost import UsageRecord, format_table, summarize, usage_from_dict
from src.data import load_jsonl_dicts


def main() -> int:
    cfg = load_config()
    results_dir = ROOT / cfg.get("paths.results_dir", "results")
    usage_dir = results_dir / "usage"

    if not usage_dir.exists() or not list(usage_dir.glob("*.jsonl")):
        print("[7_cost] no usage logs under results/usage/; run labeling/inference first.")
        return 1

    price_in = float(cfg.get("teacher.price_cny_per_1k_input", 0.0))
    price_out = float(cfg.get("teacher.price_cny_per_1k_output", 0.0))

    # Aggregate per model.
    by_model: dict[str, UsageRecord] = {}
    for f in sorted(usage_dir.glob("*.jsonl")):
        for row in load_jsonl_dicts(f):
            rec = usage_from_dict(row)
            by_model.setdefault(rec.model, UsageRecord(model=rec.model)).add(rec)

    summary = summarize(by_model.values(), price_in_per_1k=price_in, price_out_per_1k=price_out)
    table = format_table(summary)

    out = results_dir / "cost.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[7_cost] prices (CNY/1k tokens): input={price_in}, output={price_out}")
    print(table)
    print(f"[7_cost] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
