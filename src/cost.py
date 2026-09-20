"""Cost accounting (plan §5.4).

Unified, comparable units for every model:
    * CNY per 1,000 predictions
    * seconds per 1,000 predictions

API models: token usage × unit price (from config). Local models: measured
latency / throughput, with no API cost (optionally add a compute cost you fill in).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass
class UsageRecord:
    model: str
    n_predictions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def add(self, other: "UsageRecord") -> None:
        """Accumulate another record's usage into this one."""
        self.n_predictions += other.n_predictions
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.seconds += other.seconds


def cost_per_1k(
    rec: UsageRecord,
    price_in_per_1k: float = 0.0,
    price_out_per_1k: float = 0.0,
) -> float:
    """CNY per 1,000 predictions from token usage and per-1k-token prices."""
    if rec.n_predictions == 0:
        return 0.0
    total_cny = (
        rec.input_tokens * price_in_per_1k + rec.output_tokens * price_out_per_1k
    ) / 1000.0
    return total_cny * 1000.0 / rec.n_predictions


def seconds_per_1k(rec: UsageRecord) -> float:
    """Wall-clock seconds per 1,000 predictions."""
    if rec.n_predictions == 0:
        return 0.0
    return rec.seconds * 1000.0 / rec.n_predictions


def summarize(
    records: Iterable[UsageRecord],
    price_in_per_1k: float = 0.0,
    price_out_per_1k: float = 0.0,
) -> dict:
    """Aggregate records into a report-friendly dict (plan §5.4 table rows)."""
    rows = []
    totals = UsageRecord(model="TOTAL")
    for r in records:
        totals.add(r)
        rows.append(
            {
                **asdict(r),
                "cny_per_1k": round(cost_per_1k(r, price_in_per_1k, price_out_per_1k), 4),
                "sec_per_1k": round(seconds_per_1k(r), 2),
            }
        )
    return {
        "records": rows,
        "total": {
            **asdict(totals),
            "cny_per_1k": round(cost_per_1k(totals, price_in_per_1k, price_out_per_1k), 4),
            "sec_per_1k": round(seconds_per_1k(totals), 2),
        },
    }


def usage_to_dict(rec: UsageRecord) -> dict:
    return asdict(rec)


def usage_from_dict(d: dict) -> UsageRecord:
    """Rebuild a ``UsageRecord`` from a JSONL row (missing fields default to 0)."""
    return UsageRecord(
        model=d.get("model", "unknown"),
        n_predictions=int(d.get("n_predictions", 0) or 0),
        input_tokens=int(d.get("input_tokens", 0) or 0),
        output_tokens=int(d.get("output_tokens", 0) or 0),
        seconds=float(d.get("seconds", 0.0) or 0.0),
    )


def format_table(summary: dict) -> str:
    """Render a ``summarize`` result as a fixed-width text table."""
    header = ["model", "n", "in_tok", "out_tok", "sec", "cny/1k", "sec/1k"]
    rows = []
    for r in summary["records"]:
        rows.append(
            [
                r["model"], str(r["n_predictions"]), str(r["input_tokens"]),
                str(r["output_tokens"]), f"{r['seconds']:.1f}",
                f"{r['cny_per_1k']:.4f}", f"{r['sec_per_1k']:.1f}",
            ]
        )
    t = summary["total"]
    rows.append(
        ["TOTAL", str(t["n_predictions"]), str(t["input_tokens"]),
         str(t["output_tokens"]), f"{t['seconds']:.1f}",
         f"{t['cny_per_1k']:.4f}", f"{t['sec_per_1k']:.1f}"],
    )

    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join(
        "  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows
    )
    return f"{line}\n{sep}\n{body}"
