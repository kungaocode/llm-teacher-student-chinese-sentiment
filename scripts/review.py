#!/usr/bin/env python3
"""Interactive human review of the gold set (plan §4.2 double review).

Label gold samples one at a time in the terminal:

    p / 正   -> positive       n / 负   -> negative
    m / 中   -> neutral        s        -> skip (leave for later)
    q        -> save & quit    (each label is saved immediately)

Modes:
  (default)        write ``label`` straight back into data/splits/gold.jsonl.
                   Resumable: already-labeled rows are skipped on the next run.
  --annotator BOB  write to data/annotations/review_BOB.jsonl instead, so two
                   people label independently; then --merge A B computes Cohen's
                   kappa + lists disagreements.
  --only neutral   review only the neutral-candidate rows (spot-check the
                   heuristic / seed the neutral class).
  --limit N        process at most N rows this session.
  --merge A B      no loop: merge two annotator files -> kappa + disagreements.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_jsonl_dicts, write_jsonl
from src.labels import normalize_label
from src.metrics import cohen_kappa
from src.neutral import matched_hints, neutral_score

_KEYMAP = {
    "p": "positive", "pos": "positive", "positive": "positive",
    "n": "negative", "neg": "negative", "negative": "negative",
    "m": "neutral", "mid": "neutral", "neu": "neutral", "neutral": "neutral",
}
_SKIP = {"s", "skip", "x", ""}
_QUIT = {"q", "quit", "exit"}


def _parse_input(s: str) -> str:
    """Map a typed token -> canonical label, or 'skip' / 'quit' / '' (unknown)."""
    s = s.strip().lower()
    if s in _QUIT:
        return "quit"
    if s in _SKIP:
        return "skip"
    if s in _KEYMAP:
        return _KEYMAP[s]
    try:
        return normalize_label(s)  # accepts 正面/负面/中性/好/差/一般 ...
    except ValueError:
        return ""


def _hint_line(text: str) -> str:
    h = matched_hints(text)
    parts = []
    if h["neutral"]:
        parts.append(f"中:{'/'.join(h['neutral'][:6])}")
    if h["positive"]:
        parts.append(f"正:{'/'.join(h['positive'][:6])}")
    if h["negative"]:
        parts.append(f"负:{'/'.join(h['negative'][:6])}")
    return "  命中词  " + "  ".join(parts) if parts else ""


def _show(row: dict, i: int, n: int, tally: dict, width: int) -> None:
    print("\n" + "─" * min(width, 78))
    print(f"[{i}/{n}]  id={row['id']}  src_label={row.get('source_label')}  "
          f"domain={row.get('domain') or '-'}")
    print()
    for ln in textwrap.wrap(row["text"], width=max(40, width - 4)) or [row["text"]]:
        print("  " + ln)
    print()
    if row.get("teacher_suggest"):
        print(f"  教师建议: {row['teacher_suggest']}")
    hl = _hint_line(row["text"])
    if hl:
        print(hl)
    t = "  ".join(f"{k}={v}" for k, v in tally.items())
    print(f"\n  已标: {t or '(空)'}")


def _merge(cfg, name_a: str, name_b: str) -> int:
    from collections import Counter

    out_dir = ROOT / "data" / "annotations"
    pa = out_dir / f"review_{name_a}.jsonl"
    pb = out_dir / f"review_{name_b}.jsonl"
    if not pa.exists() or not pb.exists():
        print(f"[review] need both {pa.name} and {pb.name} to merge.")
        return 1

    a = {str(r["id"]): r["label"] for r in load_jsonl_dicts(pa)}
    b = {str(r["id"]): r["label"] for r in load_jsonl_dicts(pb)}
    common = sorted(set(a) & set(b), key=int)
    ya = [a[i] for i in common]
    yb = [b[i] for i in common]
    print(f"[review] merge {name_a}({len(a)}) x {name_b}({len(b)}) on {len(common)} shared rows")
    if common:
        print(f"         Cohen's kappa = {cohen_kappa(ya, yb):.3f}")
        print(f"         distribution {name_a}: {dict(Counter(ya))}")
        print(f"         distribution {name_b}: {dict(Counter(yb))}")
        disagree = [i for i in common if a[i] != b[i]]
        print(f"         disagreements: {len(disagree)}")
        for i in disagree[:20]:
            print(f"           id={i}  {name_a}={a[i]}  {name_b}={b[i]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive gold review")
    parser.add_argument("--annotator", metavar="NAME", help="write to review_NAME.jsonl instead of gold.jsonl")
    parser.add_argument("--only", choices=["neutral"], help="review only neutral candidates")
    parser.add_argument("--limit", type=int, default=0, help="max rows this session")
    parser.add_argument("--merge", nargs=2, metavar=("A", "B"), help="merge two annotators, no loop")
    args = parser.parse_args()

    cfg = load_config()
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")
    gold_path = splits_dir / "gold.jsonl"

    if args.merge:
        return _merge(cfg, args.merge[0], args.merge[1])

    if not gold_path.exists():
        print(f"[review] missing {gold_path}; run 1_prepare.py first.")
        return 1

    gold = load_jsonl_dicts(gold_path)

    # Teacher proposals (if step 3 propose mode ran) -> show as suggestion.
    suggest: dict[str, str] = {}
    prop_path = ROOT / cfg.get("paths.results_dir", "results") / "gold_teacher_proposals.jsonl"
    if prop_path.exists():
        for p in load_jsonl_dicts(prop_path):
            if p.get("teacher_label"):
                suggest[str(p["id"])] = f"{p['teacher_label']} — {p.get('reason', '')}"
    for g in gold:
        g["teacher_suggest"] = suggest.get(str(g["id"]), "")

    # Per-annotator output (for double review) vs. writing gold.jsonl directly.
    out_path = None
    done: set[str] = set()
    if args.annotator:
        out_dir = ROOT / "data" / "annotations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"review_{args.annotator}.jsonl"
        if out_path.exists():
            done = {str(r["id"]) for r in load_jsonl_dicts(out_path)}

    # Build the review queue.
    queue = []
    for g in gold:
        if args.only == "neutral" and neutral_score(g["text"]) <= 0:
            continue
        if out_path is not None and str(g["id"]) in done:
            continue
        if out_path is None and g.get("label") is not None:
            continue
        queue.append(g)

    if args.limit > 0:
        queue = queue[: args.limit]

    if not queue:
        print("[review] nothing left to review (all done, or --only filter empty).")
        return 0

    width = min(78, _term_width())
    tally: dict[str, int] = {}
    n_done = 0

    print(f"[review] {len(queue)} samples queued. 输入 p/n/m（正/负/中），s 跳过，q 保存退出。")
    for i, g in enumerate(queue, 1):
        _show(g, i, len(queue), tally, width)
        while True:
            try:
                raw = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                raw = "q"
            lab = _parse_input(raw)
            if lab == "quit":
                _save(gold, out_path, done, tally)
                print(f"\n[review] 已保存，本次新增 {n_done} 条。")
                return 0
            if lab == "skip":
                break
            if lab:
                g["label"] = lab
                tally[lab] = tally.get(lab, 0) + 1
                n_done += 1
                if out_path is not None:
                    _append(out_path, {"id": g["id"], "label": lab, "annotator": args.annotator})
                    done.add(str(g["id"]))
                else:
                    write_jsonl(gold_path, gold)  # save after each label
                break
            print("  无法识别，请输入 p/n/m（或 正/中/负）、s 跳过、q 退出。")

    _save(gold, out_path, done, tally)
    print(f"\n[review] 全部完成，本次新增 {n_done} 条。最终分布: {tally or '(空)'}")
    return 0


def _term_width() -> int:
    try:
        import shutil
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def _append(path: Path, row: dict) -> None:
    import json
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save(gold, out_path, done, tally) -> None:
    # gold.jsonl is already saved incrementally in direct mode; annotator files
    # are appended incrementally. Nothing extra needed here.
    return


if __name__ == "__main__":
    raise SystemExit(main())
