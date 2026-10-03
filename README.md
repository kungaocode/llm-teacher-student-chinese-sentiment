# Teacher–Student Sentiment Analysis of Chinese Reviews

Knowledge-distillation pipeline that labels ~2,000 unlabeled Chinese reviews with a
strong teacher LLM, fine-tunes a small student model (LoRA), and measures the
**macro-F1 × cost** trade-off of each model on a protected held-out set with an
explicit truth-source boundary.

> Status: **pipeline complete for a demo run — teacher (kimi-k3) labeled 2,000 reviews,
> student (qwen3-4b-instruct-2507, fine-tuned on Bailian) deployed; final test on 200 held-out
> reviews: student acc 0.910 / macro-F1 0.874 vs baseline 0.805 / 0.618 (truth = teacher proposals;
> human-gold review is the optional next step for publication-grade numbers).**

See [`FINAL_REPORT.md`](FINAL_REPORT.md) for the close-out report and
[`DATA_SOURCES.md`](DATA_SOURCES.md) for provenance and licence caveats.

## Why (one paragraph)

Strong LLMs are accurate but expensive per prediction. This project quantifies
*how much* accuracy a small distilled student gives up and *how much* cost it saves —
the "task–model right-sizing" curve: teacher (API) vs. student (LoRA) vs. a
traditional TF-IDF + LinearSVC baseline, all scored on the **same** held-out rows.
The current demo run uses teacher proposals as the three-class reference; it does
not yet have an independent human gold set.

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
  8_validate_binary.py (external weak check against source two-class labels)
  9_annotate_web.py  (local browser bench -> per-label human annotation files)
```

## Labels

Fixed 3-class coarse scheme: `negative / neutral / positive`.
(ChnSentiCorp is 2-class — `0→negative, 1→positive` — which is fine for macro-F1.)

## Layout

```
src/            bottom-level library: config, data, labels, metrics, cost, validation
annotator/      zero-dependency local HTTP server, storage, and static web UI
scripts/        0..9 numbered pipeline (run in order)
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
python3 scripts/8_validate_binary.py  # weak binary check + reports/binary_validation.{json,md}
# human-gold review (optional, publication-grade truth):
# run one server per annotator, in separate terminals
python3 scripts/9_annotate_web.py --annotator a --open
python3 scripts/9_annotate_web.py --annotator b --port 8766 --open
# label independently as A and B, then collect the two columns:
python3 scripts/annotate_gold.py --collect
python scripts/export_finetune.py  # rerun -> test truth switches to human labels
python3 scripts/8_validate_binary.py  # refresh diagnostics
```

## Human annotation bench

`scripts/9_annotate_web.py` starts a local-only web app at
`http://127.0.0.1:8765` by default. It shows one review at a time, provides
neutral / positive / negative buttons, asks for confirmation, and saves each
confirmed label immediately. Keyboard shortcuts are `1` neutral, `2` positive,
and `3` negative.

Use separate annotator identities for independent double review:

```bash
# terminal 1
python3 scripts/9_annotate_web.py --annotator a --open
# terminal 2
python3 scripts/9_annotate_web.py --annotator b --port 8766 --open
```

The per-label JSON files are the source of truth. The app also rebuilds the
JSONL indexes used by the existing workflow:

```text
data/annotations/web/<annotator>/labels/neutral/<id>.json
data/annotations/web/<annotator>/labels/positive/<id>.json
data/annotations/web/<annotator>/labels/negative/<id>.json
data/annotations/web/<annotator>/annotations.jsonl
data/annotations/review_<annotator>.jsonl
```

The UI's **导出标注表** action writes the current annotator column into the
existing `data/annotations/gold_annotation.csv`. It does not write human labels
directly into `data/splits/gold.jsonl`; use
`python3 scripts/annotate_gold.py --collect` after A/B review so that matching
labels are collected and disagreements remain available for adjudication.

## Results (demo run · 2026-09)

Setup: teacher = `kimi-k3` (Moonshot API), student = `qwen3-4b-instruct-2507`
fine-tuned on 2,000 teacher labels via Bailian (LoRA) and served from a dedicated
deployment, baseline = TF-IDF (char 2–4-grams) + LinearSVC. Test = 200 held-out
reviews (disjoint from train/dev). **Truth = teacher proposals** (human gold
labels are the optional next step), so the teacher row is 1.0 by construction.
`configs/default.yaml` is the local 0.6B reference configuration; the committed
demo artifacts use the cloud-fine-tuned 4B deployment named above.

| model | accuracy | macro-F1 | neutral F1 | est. cost / 1k preds |
|---|---|---|---|---|
| Teacher kimi-k3 (API) | 1.000 | 1.000 | 1.000 | ¥1.15 (list-price estimate) |
| **Student qwen3-4B (deployed)** | **0.910** | **0.874** | **0.742** | ¥0.19 est. · 452 s/1k |
| Baseline TF-IDF + LinearSVC | 0.805 | 0.618 | **0.129** | ¥0.00 · 3 s/1k |

This table is a teacher-consistency comparison, not an independent estimate of
three-class quality. The public source corpora provide only two-class labels, so
the project also runs a deliberately weaker external check:

| model | neutral abstentions | coverage | covered accuracy | neutral-as-error accuracy |
|---|---:|---:|---:|---:|
| Teacher kimi-k3 | 28 | 86.00% | 94.77% | 81.50% |
| Student qwen3-4B | 34 | 83.00% | 96.39% | 80.00% |
| Baseline TF-IDF + LinearSVC | 3 | 98.50% | 83.25% | 82.00% |
| Student, neutral falls back to baseline | 2 | 99.00% | 87.88% | 87.00% |

The fallback row is a deployment diagnostic only. It improves the noisy binary
score by giving up neutral predictions, so it is not evidence of better
three-class performance. The source labels are noisy, cover no neutral class,
and cannot replace human annotation. Full tables and pairwise agreement are in
[`reports/binary_validation.json`](reports/binary_validation.json).

Takeaways:

- The distilled student keeps ~91% of the teacher's label agreement while costing
  ~1/6 per prediction at list prices — the task–model right-sizing curve.
- On the external two-class check, all models have lower positive recall than
  negative recall; the student abstains on more positives than the teacher or
  baseline, which explains why covered accuracy is high but coverage is lower.
- Neutral is the hard class for everyone; the traditional baseline collapses on it
  (F1 0.129), which is the strongest argument for distillation on this 3-class task.
- Per-category (accuracy): plain 0.939 · mixed pos/neg 0.854 · sarcasm-candidate
  0.857 · implicit negation 0.840 · neutral-candidate 0.850 · colloquial 0.750 —
  typical failures are *overall-positive reviews with negative details → neutral*
  and *“一般般/凑合” → negative*.

## Data & license

The run uses `online_shopping_10_cats` and ChnSentiCorp. Both distributions
carried no explicit licence when downloaded; "research use" is an operational
note, not a legal licence. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for hubs,
counts, credit, and publication caveats. Raw and processed corpora stay
gitignored; rerun `python scripts/0_download.py` before `1_prepare.py`.
