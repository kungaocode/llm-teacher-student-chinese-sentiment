import csv
import json

import pytest

from annotator.store import AnnotationStore, AnnotationStoreError
from src.data import write_jsonl


def _make_store(tmp_path, annotator="a"):
    gold = tmp_path / "data" / "splits" / "gold.jsonl"
    write_jsonl(
        gold,
        [
            {
                "id": 1,
                "text": "很好，下次还买",
                "source": "shop",
                "source_label": 1,
                "domain": "衣服",
                "label": None,
            },
            {
                "id": 2,
                "text": "一般般，没有特别感觉",
                "source": "hotel",
                "source_label": 0,
                "domain": "酒店",
                "label": None,
            },
        ],
    )
    return AnnotationStore(tmp_path, annotator=annotator)


def test_session_starts_with_all_rows_unlabeled(tmp_path):
    store = _make_store(tmp_path)

    session = store.session()

    assert session["annotator"] == "a"
    assert session["total"] == 2
    assert session["annotated"] == 0
    assert session["counts"] == {"negative": 0, "neutral": 0, "positive": 0}
    assert [item["label"] for item in session["items"]] == [None, None]


def test_annotate_persists_and_moves_between_label_directories(tmp_path):
    store = _make_store(tmp_path)

    first = store.annotate(1, "positive")
    positive_path = store.labels_dir / "positive" / "1.json"
    assert first["label"] == "positive"
    assert positive_path.exists()

    changed = store.annotate(1, "negative")
    negative_path = store.labels_dir / "negative" / "1.json"
    assert changed["label"] == "negative"
    assert negative_path.exists()
    assert not positive_path.exists()

    reloaded = AnnotationStore(tmp_path, annotator="a")
    item = next(row for row in reloaded.session()["items"] if row["id"] == "1")
    assert item["label"] == "negative"


def test_clear_removes_label_and_rebuilds_review_file(tmp_path):
    store = _make_store(tmp_path)
    store.annotate(1, "neutral")
    store.annotate(2, "positive")

    cleared = store.clear(1)

    assert cleared["label"] is None
    assert not (store.labels_dir / "neutral" / "1.json").exists()
    review_rows = [
        json.loads(line)
        for line in store.review_path.read_text(encoding="utf-8").splitlines()
    ]
    assert review_rows == [{"id": 2, "label": "positive", "annotator": "a"}]


def test_export_csv_updates_only_the_current_annotator_field(tmp_path):
    store = _make_store(tmp_path)
    store.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with store.csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": 1,
                "text": "很好，下次还买",
                "source": "shop",
                "domain": "衣服",
                "source_label": 1,
                "annotator_b": "neutral",
            }
        )
        writer.writerow(
            {
                "id": 2,
                "text": "一般般，没有特别感觉",
                "source": "hotel",
                "domain": "酒店",
                "source_label": 0,
                "annotator_b": "negative",
            }
        )
    store.annotate(1, "positive")

    result = store.export_csv()

    assert result["field"] == "annotator_a"
    with store.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["annotator_a"] == "positive"
    assert rows[0]["annotator_b"] == "neutral"
    assert rows[1]["annotator_a"] == ""
    assert rows[1]["annotator_b"] == "negative"

    store.clear(1)
    store.export_csv()
    with store.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        cleared_rows = list(csv.DictReader(handle))
    assert cleared_rows[0]["annotator_a"] == ""
    assert cleared_rows[0]["annotator_b"] == "neutral"


def test_rejects_unsafe_id_and_unknown_gold_row(tmp_path):
    store = _make_store(tmp_path)

    with pytest.raises(AnnotationStoreError, match="unsafe"):
        store.annotate("../1", "positive")
    with pytest.raises(AnnotationStoreError, match="unknown gold id"):
        store.annotate(99, "positive")
