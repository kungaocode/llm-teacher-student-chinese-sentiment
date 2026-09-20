#!/usr/bin/env python3
"""Step 0: download the public review corpora into ``data/raw/`` as JSONL.

Sources (both 2-class; the 3-class labels are produced later by teacher + human):
    * online_shopping_10_cats  (ttxy/online_shopping_10_cats)    — 62,773 reviews, 10 categories
    * ChnSentiCorp             (lansinuote/ChnSentiCorp)         — 12,000 reviews (hotel/laptop/book)

Normalized schema per line: ``{text, source_label, source, domain}`` where
``source_label`` is the ORIGINAL 2-class integer (0/1). These corpora carry no
explicit license (research use); record source + license in ``reports/`` before
publishing (立项书 §9).

Proxy note: this environment sets a ``socks://`` ALL_PROXY that breaks httpx, so
we strip it here and keep ``http_proxy``/``https_proxy``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Strip the socks proxy BEFORE importing datasets/httpx.
for _k in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import clean_text, write_jsonl  # noqa: E402

SHOP = "XiangPan/online_shopping_10_cats_62k"  # NOTE: ttxy/... mislabels `label` as category id (1-10)
CHN = "lansinuote/ChnSentiCorp"


def _normalize(ds, source: str, text_key: str, label_key: str, domain_key=None) -> list[dict]:
    rows = []
    for ex in ds:
        text = clean_text(ex[text_key])
        if not text:
            continue
        row = {"text": text, "source_label": int(ex[label_key]), "source": source}
        if domain_key and domain_key in ex:
            row["domain"] = ex[domain_key]
        rows.append(row)
    return rows


def main() -> int:
    from datasets import load_dataset

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) online_shopping_10_cats
    shop = load_dataset(SHOP, split="train")
    shop_rows = _normalize(shop, "online_shopping_10_cats", "text", "label", "cat")
    write_jsonl(raw_dir / "online_shopping_10_cats.jsonl", shop_rows)

    # 2) ChnSentiCorp (train + validation + test)
    chn_rows = []
    for split in ("train", "validation", "test"):
        chn_rows += _normalize(
            load_dataset(CHN, split=split), "chnsenticorp", "text", "label", None
        )
    write_jsonl(raw_dir / "chnsenticorp.jsonl", chn_rows)

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [
            {
                "name": "online_shopping_10_cats",
                "hub": SHOP,
                "n_rows": len(shop_rows),
                "label": "binary 0/1",
                "domains": sorted({r.get("domain") for r in shop_rows if r.get("domain")}),
                "license": "unstated (research use)",
            },
            {
                "name": "chnsenticorp",
                "hub": CHN,
                "n_rows": len(chn_rows),
                "label": "binary 0/1",
                "domains": ["hotel", "laptop", "book"],
                "license": "unstated (research use; Tan Songbo @ ICT CAS)",
            },
        ],
    }
    (raw_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for d in manifest["datasets"]:
        print(f"[0_download] {d['name']:24s} {d['n_rows']:6d} rows  <- {d['hub']}")
    print(f"[0_download] manifest -> {raw_dir / 'data_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
