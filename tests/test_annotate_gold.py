import csv

import scripts.annotate_gold as annotate_gold
from src.data import load_jsonl_dicts, write_jsonl


def _setup_project(tmp_path, monkeypatch, labels=("positive", "negative")):
    gold_path = tmp_path / "data" / "splits" / "gold.jsonl"
    write_jsonl(
        gold_path,
        [
            {"id": 1, "text": "很好", "label": None},
            {"id": 2, "text": "很差", "label": None},
        ],
    )
    csv_path = tmp_path / "data" / "annotations" / "gold_annotation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
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
        for row_id, text, label in zip((1, 2), ("很好", "很差"), labels):
            writer.writerow(
                {
                    "id": row_id,
                    "text": text,
                    "annotator_a": "",
                    "annotator_b": label,
                }
            )
    monkeypatch.setattr(annotate_gold, "ROOT", tmp_path)
    return gold_path, {
        "paths.splits_dir": "data/splits",
        "paths.results_dir": "results",
    }


def test_single_annotator_collection_requires_explicit_flag(
    tmp_path,
    monkeypatch,
    capsys,
):
    gold_path, cfg = _setup_project(tmp_path, monkeypatch)

    result = annotate_gold._collect(cfg)

    assert result == 1
    assert all(row["label"] is None for row in load_jsonl_dicts(gold_path))
    assert "--allow-single-annotator" in capsys.readouterr().out


def test_single_annotator_collection_writes_human_labels(
    tmp_path,
    monkeypatch,
    capsys,
):
    gold_path, cfg = _setup_project(tmp_path, monkeypatch)

    result = annotate_gold._collect(cfg, allow_single_annotator=True)

    assert result == 0
    rows = load_jsonl_dicts(gold_path)
    assert [row["label"] for row in rows] == ["positive", "negative"]
    output = capsys.readouterr().out
    assert "no inter-annotator kappa" in output
    assert "2/2 gold rows" in output
