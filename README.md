# Teacher–Student Sentiment Analysis of Chinese Reviews

Knowledge-distillation pipeline that labels ~2,000 unlabeled Chinese reviews with a
strong teacher LLM, fine-tunes a small student model (LoRA), and measures the
**macro-F1 × cost** trade-off of each model on a human-labeled gold set.

> Status: **pipeline complete for a demo run — teacher (kimi-k3) labeled 2,000 reviews,
> student (qwen3-4b-instruct-2507, fine-tuned on Bailian) deployed; final test on 200 held-out
> reviews: student acc 0.910 / macro-F1 0.874 vs baseline 0.805 / 0.618 (truth = teacher proposals;
> human-gold review is the optional next step for publication-grade numbers).**

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
  3_check_gold.py    (teacher proposes gold labels for human review)
  export_finetune.py (train/dev/test -> cloud fine-tune JSONL, messages format)
  cloud_predict.py   (call the deployed model + score_test per class/category)
  5_baselines.py     (TF-IDF + LinearSVC)
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
cp .env.example .env      # fill MOONSHOT_API_KEY (teacher kimi-k3)
# append the Bailian deployment API key to api.txt (used by cloud_predict.py)
```

## Reproduce

```bash
python scripts/1_prepare.py      # data/raw -> splits (needs the corpus first)
python scripts/2_teacher_label.py
python scripts/3_check_gold.py     # propose gold labels for human review
python scripts/export_finetune.py  # train/dev/test -> cloud fine-tune JSONL (messages format)
python scripts/cloud_predict.py --model <deployment-id>   # cloud predict + score
python scripts/score_test.py results/predictions/<model>_predictions.jsonl
python scripts/5_baselines.py      # TF-IDF + LinearSVC baseline (local, zero cost)
python scripts/7_cost.py           # cost table from results/usage/*.jsonl
# human-gold review (optional, publication-grade truth):
python scripts/review.py           # single-annotator interactive gold labeling
python scripts/export_finetune.py  # rerun -> test truth switches to human labels
```

## Results (demo run · 2026-09)

Setup: teacher = `kimi-k3` (Moonshot API), student = `qwen3-4b-instruct-2507`
fine-tuned on 2,000 teacher labels via Bailian (LoRA) and served from a dedicated
deployment, baseline = TF-IDF (char 2–4-grams) + LinearSVC. Test = 200 held-out
reviews (disjoint from train/dev). **Truth = teacher proposals** (human gold
labels are the optional next step), so the teacher row is 1.0 by construction.

| model | accuracy | macro-F1 | neutral F1 | est. cost / 1k preds |
|---|---|---|---|---|
| Teacher kimi-k3 (API) | 1.000 | 1.000 | 1.000 | ¥1.15 (list-price estimate) |
| **Student qwen3-4B (deployed)** | **0.910** | **0.874** | **0.742** | ¥0.19 est. · 452 s/1k |
| Baseline TF-IDF + LinearSVC | 0.805 | 0.618 | **0.129** | ¥0.00 · 3 s/1k |

Takeaways:

- The distilled student keeps ~91% of the teacher's label agreement while costing
  ~1/6 per prediction at list prices — the task–model right-sizing curve.
- Neutral is the hard class for everyone; the traditional baseline collapses on it
  (F1 0.129), which is the strongest argument for distillation on this 3-class task.
- Per-category (accuracy): plain 0.939 · mixed pos/neg 0.854 · sarcasm-candidate
  0.857 · implicit negation 0.840 · neutral-candidate 0.850 · colloquial 0.750 —
  typical failures are *overall-positive reviews with negative details → neutral*
  and *“一般般/凑合” → negative*.

## Data & license

ChnSentiCorp (open Chinese hotel/shopping reviews). Download is deferred; place the
corpus under `data/raw/` as `*.csv` / `*.jsonl` (columns `text`, optional `label`)
or the HF `.arrow` files, then run `1_prepare.py`. Record source + license in
`reports/` before publication.
