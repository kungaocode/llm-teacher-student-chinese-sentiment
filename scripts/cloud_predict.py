#!/usr/bin/env python3
"""Call the user's cloud-deployed (Bailian/DashScope OpenAI-compatible) model and
score it against the exported test set.

    python scripts/cloud_predict.py --list                 # discover which api.txt key works + model names
    python scripts/cloud_predict.py --model <NAME> --out results/predictions/<name>.jsonl
    python scripts/cloud_predict.py --model <NAME>         # predict + run score_test.py

Prompt is taken verbatim from the exported finetune_*.jsonl rows (system+user),
so inference matches training exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os

# httpx cannot parse socks:// scheme proxies; match src/teacher.py's guard.
for _proxy_key in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_proxy_key, None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI

from src.data import load_jsonl_dicts, write_jsonl
from src.labels import LABELS, normalize_label

BASES = [
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
]
DEFAULT_TEST = ROOT / "data/processed/finetune_test.jsonl"


def _keys() -> list[str]:
    p = ROOT / "api.txt"
    if not p.exists():
        p = ROOT / ".env"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and not line.startswith("sk-"):
            line = line.split("=", 1)[1].strip()
        if line.startswith("sk-"):
            out.append(line)
    return out


def list_models(verbose: bool = True) -> None:
    for base in BASES:
        for key in _keys():
            client = OpenAI(base_url=base, api_key=key, timeout=30)
            try:
                models = client.models.list()
                names = sorted(m.id for m in models.data)
                print(f"[list] OK  base={base}  key=...{key[-4:]}  n={len(names)}")
                for n in names:
                    print("        ", n)
            except Exception as exc:  # noqa: BLE001
                print(f"[list] FAIL base={base} key=...{key[-4:]}  {type(exc).__name__}: {str(exc)[:200]}")


def _parse_label(content: str) -> str:
    s = (content or "").strip()
    for lbl in LABELS:
        if lbl in s.lower():
            return lbl
    try:
        return normalize_label(s)
    except ValueError as exc:
        raise ValueError(f"cannot parse label from: {content[:80]!r}") from exc


def _working_key() -> str:
    """Return the first api.txt key that authenticates against the default base."""
    for key in _keys():
        try:
            client = OpenAI(base_url=BASES[0], api_key=key, timeout=15)
            client.models.list()
            return key
        except Exception:  # noqa: BLE001  (AuthenticationError etc.) -> try next
            continue
    raise SystemExit("[cloud_predict] no working api.txt key for " + BASES[0])


def predict(model: str, workers: int) -> tuple[Path, int]:
    test = load_jsonl_dicts(DEFAULT_TEST)
    meta = load_jsonl_dicts(ROOT / "results" / "test_composition.jsonl") or [{}] * len(test)
    if len(meta) != len(test):
        print("[cloud_predict] WARNING: test_composition length mismatch; ids may be wrong.")
    key = _working_key()
    client = OpenAI(base_url=BASES[0], api_key=key, timeout=60)

    def one(row: dict, i: int) -> dict:
        msgs = [m for m in row["messages"] if m["role"] != "assistant"]
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=0, max_tokens=16,
            extra_body={"enable_thinking": False},  # qwen3 deployment: explicit false for non-streaming
        )
        label = _parse_label(resp.choices[0].message.content or "")
        usage = resp.usage
        return {
            "id": meta[i].get("id", i) if i < len(meta) else i,
            "label": label,
            "usage": {
                "model": model,
                "n_predictions": 1,
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "seconds": time.time() - t0,
            },
        }

    rows, usage = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, row, i): i for i, row in enumerate(test)}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            usage.append(r.pop("usage"))
    rows.sort(key=lambda r: r["id"])
    usage.sort(key=lambda u: u["n_predictions"])

    out = ROOT / "results" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / f"{model.replace('/', '_')}_predictions.jsonl"
    write_jsonl(pred_path, rows)

    usage_dir = ROOT / "results" / "usage"
    usage_dir.mkdir(parents=True, exist_ok=True)
    usage_path = usage_dir / f"cloud_{model.replace('/', '_')}.jsonl"
    write_jsonl(usage_path, usage)

    print(f"[cloud_predict] {len(rows)} predictions -> {pred_path}")
    print(f"[cloud_predict] usage            -> {usage_path}")
    return pred_path, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list accessible models per key")
    ap.add_argument("--model", help="deployment / fine-tuned model id to call")
    ap.add_argument("--out", default=None, help="output predictions path")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-score", action="store_true", help="predict only, no scoring")
    args = ap.parse_args()

    if args.list or not args.model:
        list_models()
        return 0
    pred_path, n = predict(args.model, args.workers)
    if not args.no_score:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts/score_test.py"), str(pred_path)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
