"""File-backed annotation storage for the local review web app.

Each confirmed label is written to one of three label directories. The
per-label JSON files are the source of truth; the JSONL index and annotator
files are rebuilt after every mutation so interrupted sessions remain
recoverable and compatible with the existing review tooling.
"""
from __future__ import annotations

import csv
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.data import load_jsonl_dicts
from src.labels import LABELS, normalize_label


class AnnotationStoreError(RuntimeError):
    """Raised when annotation state is invalid or cannot be persisted."""


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_ANNOTATOR_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_CSV_FIELDS = [
    "id",
    "text",
    "source",
    "domain",
    "source_label",
    "teacher_suggest",
    "annotator_a",
    "annotator_b",
    "final_label",
    "note",
]


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding=encoding)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sort_key(value: str) -> tuple[int, Any]:
    text = str(value)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


class AnnotationStore:
    """Read gold rows and persist one independent annotation per row."""

    labels: tuple[str, ...] = LABELS

    def __init__(
        self,
        root: Path | str,
        annotator: str = "a",
        *,
        gold_path: Optional[Path | str] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.annotator = str(annotator).strip()
        if not _SAFE_ANNOTATOR_RE.fullmatch(self.annotator):
            raise ValueError(
                "annotator must contain only letters, numbers, '_' or '-' "
                "and be at most 32 characters"
            )

        self.gold_path = (
            Path(gold_path).resolve()
            if gold_path is not None
            else self.root / "data" / "splits" / "gold.jsonl"
        )
        self.annotation_root = self.root / "data" / "annotations"
        self.run_dir = self.annotation_root / "web" / self.annotator
        self.labels_dir = self.run_dir / "labels"
        self.index_path = self.run_dir / "annotations.jsonl"
        self.review_path = self.annotation_root / f"review_{self.annotator}.jsonl"
        self.csv_path = self.annotation_root / "gold_annotation.csv"
        self._lock = threading.RLock()

    def session(self) -> dict[str, Any]:
        """Return the gold rows joined with the current annotator's labels."""
        with self._lock:
            gold = self._load_gold()
            annotations = self._scan_annotations_locked()
            items = [
                self._public_item(row, annotations.get(str(row["id"])))
                for row in gold
            ]
            return {
                "annotator": self.annotator,
                "total": len(items),
                "annotated": sum(1 for item in items if item["label"]),
                "counts": self._counts(items),
                "items": items,
            }

    def annotate(self, item_id: str | int, label: str) -> dict[str, Any]:
        """Persist a label, moving the row when its label changes."""
        row_id = self._safe_id(item_id)
        try:
            canonical = normalize_label(label)
        except ValueError as exc:
            raise AnnotationStoreError(str(exc)) from exc
        if canonical not in self.labels:
            raise AnnotationStoreError(f"unsupported label: {label!r}")

        with self._lock:
            gold_by_id = {str(row["id"]): row for row in self._load_gold()}
            if row_id not in gold_by_id:
                raise AnnotationStoreError(f"unknown gold id: {row_id}")

            annotations = self._scan_annotations_locked()
            previous = annotations.get(row_id)
            previous_label = previous.get("label") if previous else None
            if previous_label and previous_label != canonical:
                old_path = self._label_path(previous_label, row_id)
                old_path.unlink(missing_ok=True)

            gold = gold_by_id[row_id]
            now = _utc_now()
            record = {
                "id": gold["id"],
                "text": gold.get("text", ""),
                "source": gold.get("source", ""),
                "source_label": gold.get("source_label"),
                "domain": gold.get("domain", ""),
                "annotator": self.annotator,
                "label": canonical,
                "annotated_at": (
                    previous.get("annotated_at")
                    if previous and previous_label == canonical
                    else now
                ),
                "updated_at": now,
            }
            _atomic_write_json(self._label_path(canonical, row_id), record)

            annotations[row_id] = record
            self._rebuild_outputs_locked(annotations)
            return self._public_item(gold, record)

    def clear(self, item_id: str | int) -> dict[str, Any]:
        """Remove an existing annotation and return the now-unlabeled item."""
        row_id = self._safe_id(item_id)
        with self._lock:
            gold_by_id = {str(row["id"]): row for row in self._load_gold()}
            if row_id not in gold_by_id:
                raise AnnotationStoreError(f"unknown gold id: {row_id}")

            annotations = self._scan_annotations_locked()
            previous = annotations.pop(row_id, None)
            if previous:
                self._label_path(previous["label"], row_id).unlink(missing_ok=True)
                self._rebuild_outputs_locked(annotations)
            return self._public_item(gold_by_id[row_id], None)

    def export_csv(self) -> dict[str, Any]:
        """Write current labels into the existing annotator A/B CSV workflow."""
        with self._lock:
            gold = self._load_gold()
            annotations = self._scan_annotations_locked()
            field = self._csv_field()
            fieldnames, existing = self._read_csv_locked(field)
            existing_by_id = {str(row.get("id", "")).strip(): row for row in existing}
            suggestions = self._teacher_suggestions()
            managed_by_web = self.run_dir.exists()
            rows: list[dict[str, Any]] = []

            for gold_row in gold:
                row_id = str(gold_row["id"])
                row = dict(existing_by_id.get(row_id, {}))
                row.setdefault("id", gold_row["id"])
                row.setdefault("text", gold_row.get("text", ""))
                row.setdefault("source", gold_row.get("source", ""))
                row.setdefault("domain", gold_row.get("domain", ""))
                row.setdefault("source_label", gold_row.get("source_label", ""))
                row.setdefault("teacher_suggest", suggestions.get(row_id, ""))
                for name in fieldnames:
                    row.setdefault(name, "")
                annotation = annotations.get(row_id)
                if managed_by_web:
                    row[field] = annotation["label"] if annotation else ""
                elif annotation:
                    row[field] = annotation["label"]
                rows.append(row)

            desired = list(fieldnames)
            for row in rows:
                for key in row:
                    if key not in desired:
                        desired.append(key)

            temp = self.csv_path.with_name(f".{self.csv_path.name}.tmp")
            temp.parent.mkdir(parents=True, exist_ok=True)
            with temp.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=desired)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp, self.csv_path)
            return {
                "path": str(self.csv_path.relative_to(self.root)),
                "field": field,
                "rows": len(rows),
                "annotated": len(annotations),
            }

    def _load_gold(self) -> list[dict[str, Any]]:
        if not self.gold_path.exists():
            raise AnnotationStoreError(
                f"missing {self.gold_path}; run scripts/1_prepare.py first"
            )
        rows = load_jsonl_dicts(self.gold_path)
        if not rows:
            raise AnnotationStoreError(f"gold file is empty: {self.gold_path}")
        seen: set[str] = set()
        for row in rows:
            row_id = self._safe_id(row.get("id"))
            if row_id in seen:
                raise AnnotationStoreError(f"duplicate gold id: {row_id}")
            seen.add(row_id)
        return rows

    def _scan_annotations_locked(self) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for label in self.labels:
            directory = self.labels_dir / label
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise AnnotationStoreError(f"cannot read annotation {path}: {exc}") from exc
                row_id = self._safe_id(record.get("id", path.stem))
                if str(record.get("label", "")) != label:
                    raise AnnotationStoreError(
                        f"annotation {path} is in {label}/ but records label="
                        f"{record.get('label')!r}"
                    )
                if row_id in found:
                    raise AnnotationStoreError(
                        f"id {row_id} has annotations in multiple label directories"
                    )
                found[row_id] = record
        return found

    def _rebuild_outputs_locked(self, annotations: dict[str, dict[str, Any]]) -> None:
        ordered = [
            annotations[row_id]
            for row_id in sorted(annotations, key=_sort_key)
        ]
        index_text = "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in ordered
        )
        _atomic_write_text(self.index_path, index_text)

        review_rows = [
            {
                "id": record["id"],
                "label": record["label"],
                "annotator": self.annotator,
            }
            for record in ordered
        ]
        review_text = "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in review_rows
        )
        _atomic_write_text(self.review_path, review_text)

    def _read_csv_locked(self, field: str) -> tuple[list[str], list[dict[str, str]]]:
        if not self.csv_path.exists():
            return list(_CSV_FIELDS), []
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or _CSV_FIELDS)
            rows = list(reader)
        if field not in fieldnames:
            fieldnames.append(field)
        return fieldnames, rows

    def _teacher_suggestions(self) -> dict[str, str]:
        proposal_path = self.root / "results" / "gold_teacher_proposals.jsonl"
        if not proposal_path.exists():
            return {}
        suggestions: dict[str, str] = {}
        for row in load_jsonl_dicts(proposal_path):
            label = row.get("teacher_label")
            if label:
                suggestions[str(row["id"])] = f"{label} - {row.get('reason', '')}"
        return suggestions

    def _public_item(
        self,
        gold: dict[str, Any],
        annotation: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "id": str(gold["id"]),
            "text": gold.get("text", ""),
            "source": gold.get("source", ""),
            "domain": gold.get("domain", ""),
            "label": annotation.get("label") if annotation else None,
            "annotated_at": annotation.get("annotated_at") if annotation else None,
            "updated_at": annotation.get("updated_at") if annotation else None,
        }

    def _label_path(self, label: str, row_id: str) -> Path:
        if label not in self.labels:
            raise AnnotationStoreError(f"unsupported label directory: {label!r}")
        return self.labels_dir / label / f"{self._safe_id(row_id)}.json"

    def _csv_field(self) -> str:
        lowered = self.annotator.lower()
        if lowered in ("a", "b"):
            return f"annotator_{lowered}"
        return f"annotator_{self.annotator}"

    @staticmethod
    def _safe_id(value: Any) -> str:
        row_id = str(value).strip()
        if not row_id or not _SAFE_ID_RE.fullmatch(row_id):
            raise AnnotationStoreError(f"unsafe or empty id: {value!r}")
        return row_id

    @staticmethod
    def _counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counts = {label: 0 for label in LABELS}
        for item in items:
            label = item.get("label")
            if label in counts:
                counts[label] += 1
        return counts
