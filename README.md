# Teacher–Student Sentiment Analysis of Chinese Reviews

Knowledge-distillation pipeline that labels ~2,000 unlabeled Chinese reviews with a
strong teacher LLM, fine-tunes a small student model (LoRA), and measures the
**macro-F1 × cost** trade-off of each model on a protected held-out set with an
explicit truth-source boundary.

> Status: **close-out complete on a 200-row human gold set — teacher (kimi-k3) labeled 2,000
> reviews, student (qwen3-4b-instruct-2507, fine-tuned on Bailian) was deployed, and the held-out
> set uses the B column from the A/B annotation workflow as final gold: student acc 0.835 /
> macro-F1 0.771, teacher 0.835 / 0.764, baseline 0.760 / 0.565. Because the final labels come
> from one column, it has no inter-annotator kappa and is not presented as double-reviewed
> publication-grade gold.**

See [`FINAL_REPORT.md`](FINAL_REPORT.md) for the close-out report and
[`ANNOTATION_WEB.md`](ANNOTATION_WEB.md) for the web annotation workflow.
See [`DATA_SOURCES.md`](DATA_SOURCES.md) for provenance and licence caveats.

## Why (one paragraph)

Strong LLMs are accurate but expensive per prediction. This project quantifies
*how much* accuracy a small distilled student gives up and *how much* cost it saves —
the "task–model right-sizing" curve: teacher (API) vs. student (LoRA) vs. a
traditional TF-IDF + LinearSVC baseline, all scored on the **same** held-out rows.
The final run uses the B column from the A/B human annotation workflow as the
three-class reference; teacher proposals remain available as an independent model
prediction source.

## Pipeline

```
data/raw ──▶ 1_prepare.py ──▶ data/splits/{train_pool, dev, gold}
                                      │
  2_teacher_label.py (teacher LLM) ──▶ train_pool.jsonl + labels
  3_check_gold.py    (teacher proposes gold labels for human review)
  9_annotate_web.py  (local human annotation bench)
  annotate_gold.py --collect (human labels -> gold.jsonl)
  export_finetune.py (train/dev/test -> cloud fine-tune JSONL, messages format)
  cloud_predict.py   (call the deployed model + score_test per class/category)
  5_baselines.py     (TF-IDF + LinearSVC)
  6_evaluate.py      (all predictions vs. human gold; no teacher API call)
  7_cost.py          (CNY & seconds per 1,000 predictions)
  8_validate_binary.py (external weak check against source two-class labels)
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

# Human gold used by this close-out: annotator B only.
python3 scripts/9_annotate_web.py --annotator b --open
# Export the CSV from the UI, then collect the human labels:
python3 scripts/annotate_gold.py --collect --allow-single-annotator
python3 scripts/export_finetune.py       # refresh test truth = human labels
python3 scripts/6_evaluate.py            # score all models offline on human gold
python3 scripts/8_validate_binary.py     # refresh the external weak check
```

For stronger evaluation, run A and B independently, then collect without the
single-annotator flag. Matching labels are accepted automatically; disagreements
remain uncollected until `final_label` contains the adjudicated label.

## Human annotation bench

`scripts/9_annotate_web.py` starts a local-only web app at
`http://127.0.0.1:8765` by default. It shows one review at a time, provides
neutral / positive / negative buttons, asks for confirmation, and saves each
confirmed label immediately. Keyboard shortcuts are `1` neutral, `2` positive,
and `3` negative.

For the final close-out, the B column is sufficient as the selected gold, but the
report must state that A was not collected:

```bash
python3 scripts/9_annotate_web.py --annotator b --open
```

For independent double review, use separate annotator identities:

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
labels are collected and disagreements remain available for adjudication. If
only one column was intentionally filled, add `--allow-single-annotator`; the
command then reports that no inter-annotator kappa is available.

## Results (final run · 2026-10)

Setup: teacher = `kimi-k3` (Moonshot API), student = `qwen3-4b-instruct-2507`
fine-tuned on 2,000 teacher labels via Bailian (LoRA) and served from a dedicated
deployment, baseline = TF-IDF (char 2–4-grams) + LinearSVC. Test = 200 held-out
reviews (disjoint from train/dev). **Truth = human labels from annotator B**
(single-annotator gold; positive 91, negative 84, neutral 25).
`configs/default.yaml` is the local 0.6B reference configuration; the committed
demo artifacts use the cloud-fine-tuned 4B deployment named above.

| model | accuracy | macro-F1 | neutral F1 | kappa vs. human | est. cost / 1k preds |
|---|---:|---:|---:|---:|---|
| Teacher kimi-k3 (API) | 0.835 | 0.764 | 0.528 | 0.728 | ¥1.15 (list-price estimate) |
| **Student qwen3-4B (deployed)** | **0.835** | **0.771** | **0.542** | **0.733** | ¥0.19 est. · 452 s/1k |
| Baseline TF-IDF + LinearSVC | 0.760 | 0.565 | 0.071 | 0.579 | ¥0.00 · 3 s/1k |

Student and teacher tie on accuracy; the student is 0.007 higher on macro-F1 on
this 200-row test. The result should be read as a close single-model comparison,
not as proof that the student matches the teacher in general. The student matches
the teacher on 91.0% of held-out rows (kappa 0.854), showing strong distillation
consistency; that pairwise number is not human accuracy. The public source
corpora provide only two-class labels, so the project also runs a deliberately
weaker external check:

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
  ~1/6 per prediction at list prices; on human gold it reaches 0.835 accuracy /
  0.771 macro-F1, effectively tying the teacher on this sample.
- On the external two-class check, all models have lower positive recall than
  negative recall; the student abstains on more positives than the teacher or
  baseline, which explains why covered accuracy is high but coverage is lower.
- Neutral is the hard class for everyone; the traditional baseline collapses on it
  (F1 0.071), while the student reaches 0.542.
- Per-category (student accuracy): plain 0.886 · sarcasm-candidate 0.857 ·
  colloquial 0.750 · neutral-candidate 0.700 · implicit negation 0.700 ·
  mixed pos/neg 0.688 —
  typical failures are *overall-positive reviews with negative details → neutral*
  and *“一般般/凑合” → negative*.

## Evaluation limits

The platform supports independent A/B labels, but the final gold uses the B column
only. The report can therefore state human-gold accuracy, but cannot report
inter-annotator agreement or adjudication reliability. The neutral class has only
25 human examples, which makes its F1 and per-category estimates sensitive to a
few decisions. Adding an independent second annotation is the highest-value way
to strengthen the evaluation; the student does not need to be retrained for that
step.

## Data & license

The run uses `online_shopping_10_cats` and ChnSentiCorp. Both distributions
carried no explicit licence when downloaded; "research use" is an operational
note, not a legal licence. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for hubs,
counts, credit, and publication caveats. Raw and processed corpora stay
gitignored; rerun `python scripts/0_download.py` before `1_prepare.py`.
