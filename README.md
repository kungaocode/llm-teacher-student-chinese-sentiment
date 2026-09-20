# Teacher–Student Sentiment Analysis of Chinese Reviews

Knowledge-distillation pipeline that labels ~2,000 unlabeled Chinese reviews with a
strong teacher LLM, fine-tunes a small student model (LoRA), and measures the
**macro-F1 × cost** trade-off of each model on a human-labeled gold set.

> Status: **data + teacher labeling done; fine-tune JSONL (train/dev/test) exported;
> gold human review pending; student fine-tuning runs on a cloud GPU (user-chosen Qwen 4B or smaller).**

## Why (one paragraph)

Strong LLMs are accurate but expensive per prediction. This project quantifies
*how much* accuracy a small distilled student gives up and *how much* cost it saves —
the "task–model right-sizing" curve: teacher (API) vs. student (LoRA) vs. a
traditional TF-IDF + LinearSVC baseline, all scored on the **same** human gold set.

## Pipeline

```
data/raw ──▶ 1_prepare.py ──▶ data/splits/{train_pool, dev, gold}
                                      │
  2_teacher_label.py (teacher LLM) ──▶ train_pool.jsonl + labels
  3_check_gold.py    (teacher–human agreement gate)
  4_train_student.py (LoRA fine-tune) ──▶ checkpoints/
  5_baselines.py     (TF-IDF + LinearSVC)
  6_evaluate.py      (macro-F1 of teacher / student / baseline on gold)
  7_cost.py          (CNY & seconds per 1,000 predictions)
```

## Labels

Fixed 3-class coarse scheme: `negative / neutral / positive`.
(ChnSentiCorp is 2-class — `0→negative, 1→positive` — which is fine for macro-F1.)

## Layout

```
src/            bottom-level library: config, data, labels, metrics, cost
scripts/        1..7 numbered pipeline (run in order)
configs/        default.yaml — model names, sizes, seed (pinned for reproducibility)
data/           raw / processed / splits (large files not committed)
results/        metrics.json, cost table, error samples
reports/        findings (incl. the Chinese-hard-cases subsection)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill DASHSCOPE_API_KEY for the teacher
```

## Reproduce

```bash
python scripts/1_prepare.py      # data/raw -> splits (needs the corpus first)
python scripts/2_teacher_label.py
python scripts/3_check_gold.py     # propose gold labels for human review
python scripts/export_finetune.py  # train/dev/test -> cloud fine-tune JSONL (messages format)
python scripts/score_test.py results/predictions/<model>.jsonl  # score cloud-model predictions
python scripts/5_baselines.py
python scripts/4_train_student.py
python scripts/6_evaluate.py
python scripts/7_cost.py
```

## Data & license

ChnSentiCorp (open Chinese hotel/shopping reviews). Download is deferred; place the
corpus under `data/raw/` as `*.csv` / `*.jsonl` (columns `text`, optional `label`)
or the HF `.arrow` files, then run `1_prepare.py`. Record source + license in
`reports/` before publication.
