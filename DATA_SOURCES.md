# Data Sources and Usage Boundary

## Corpora

| corpus | Hugging Face hub used by `0_download.py` | rows | labels | domains |
|---|---|---:|---|---|
| `online_shopping_10_cats` | `XiangPan/online_shopping_10_cats_62k` | 62,773 | original binary `0/1` | 10 product categories |
| ChnSentiCorp | `lansinuote/ChnSentiCorp` | 12,000 | original binary `0/1` | hotel, laptop, book |

`scripts/0_download.py` normalizes both into:

```json
{"text": "...", "source": "...", "source_label": 0, "domain": "..."}
```

The raw files, processed training files, and split files are intentionally
gitignored.

## Provenance

- `online_shopping_10_cats` was downloaded from the hub repository
  `XiangPan/online_shopping_10_cats_62k`. The original project is associated
  with the online shopping review corpus; verify the upstream repository terms
  before redistribution or commercial use.
- ChnSentiCorp credit belongs to Tan Songbo at the Institute of Computing
  Technology, Chinese Academy of Sciences. The Hugging Face mirror
  `lansinuote/ChnSentiCorp` was used for this run.
- `data/raw/data_manifest.json` records the exact hub identifiers and row
  counts used by this workspace.

## Licence Status

Neither downloaded distribution included an explicit licence file or licence
field at collection time. The manifest therefore says `unstated (research use)`;
that is an internal risk note, not a licence grant.

Before publishing, redistributing derived data, or making commercial claims:

1. confirm the terms with the upstream dataset owners;
2. keep raw corpora out of the repository unless permission is explicit;
3. state that the current numerical results are research/demo results with
   unresolved data-licence status.

## Label Validity

The source `0/1` labels are useful as an external weakness check but are not
human gold for this project:

- they contain only two classes and cannot validate `neutral`;
- spot checks found polarity disagreements in the source labels;
- the current 200-row `data/splits/gold.jsonl` has empty `label` fields, so the
  teacher-proposal comparison is not independent evaluation;
- `scripts/8_validate_binary.py` therefore reports coverage, covered accuracy,
  and neutral-as-error accuracy separately.

Publication-grade claims require independent human labels for the 200 held-out
rows, preferably with two annotators and adjudication for disagreements.
