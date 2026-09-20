#!/usr/bin/env python3
"""Step 4: fine-tune the student model (LoRA) on teacher-labeled data.

Trains a small Qwen3 student to reproduce the teacher's labels on the train
pool, with early stopping on a held-out dev set. Hyperparameters come from
config ``student`` (LoRA rank 16 / alpha 32, 3 epochs, lr 2e-4, seq len 256).

Class imbalance: the teacher labels skew pos/neg (neutral ~6%). When
``student.class_weights = balanced``, each sample's loss is reweighted inversely
to its class frequency (sklearn-style ``n / (k * n_class)``) so the minority
``neutral`` class contributes proportionally. Prompt tokens are masked
(``-100``) and carry no loss.

Design choice: the student is prompted to emit the label *word* directly
(positive/neutral/negative) rather than the teacher's ``{label, reason}`` JSON —
a 0.6B model emits the token more reliably than valid JSON, and the label is
all evaluation needs. Teacher output still keeps ``reason`` for error analysis.

Requires a GPU + ``peft``/``accelerate``. Use ``--dry-run`` to validate the data
pipeline and print the class balance + weights without touching torch.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_jsonl_dicts
from src.labels import coerce_label

_PROMPT = "请判断下面评论的情感：\n{text}\n\n情感标签（positive/neutral/negative）："


def _load_train_dev(cfg) -> tuple[list[dict], list[dict]]:
    processed_dir = ROOT / cfg.get("paths.processed_dir", "data/processed")
    splits_dir = ROOT / cfg.get("paths.splits_dir", "data/splits")

    train_path = processed_dir / "teacher_labels.jsonl"
    dev_path = splits_dir / "dev.jsonl"
    if not train_path.exists():
        raise SystemExit(f"[4_train_student] missing {train_path}; run 2_teacher_label.py first.")
    train = [r for r in load_jsonl_dicts(train_path) if r.get("label") is not None]
    dev = [r for r in load_jsonl_dicts(dev_path) if r.get("label") is not None] if dev_path.exists() else []
    return train, dev


def _compute_class_weights(train: list[dict]) -> dict[str, float]:
    """sklearn-style balanced weights: ``n / (n_classes * count)``."""
    counts = Counter(coerce_label(r["label"]) for r in train)
    n = sum(counts.values())
    return {k: n / (len(counts) * v) for k, v in counts.items()}


def build_pairs(rows: list[dict], weights: dict[str, float] | None = None) -> list[dict]:
    """Return ``[{prompt, completion, weight}]`` with canonical labels."""
    pairs = []
    for r in rows:
        label = coerce_label(r["label"])
        pair = {"prompt": _PROMPT.format(text=r["text"]), "completion": label}
        if weights is not None:
            pair["weight"] = weights[label]
        pairs.append(pair)
    return pairs


def _tokenize(examples, tokenizer, max_len: int) -> list[dict]:
    """Tokenize prompt+completion; mask prompt positions; keep per-sample weight.

    Returns a list of per-example dicts (what HF's data collator expects), not a
    dict-of-lists.
    """
    out = []
    for ex in examples:
        p = tokenizer(ex["prompt"], add_special_tokens=True)["input_ids"]
        c = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = (p + c)[:max_len]
        lab = [-100] * min(len(p), max_len) + c[: max_len - len(p)]
        lab = (lab + [-100] * max_len)[:max_len]
        rec = {"input_ids": ids, "labels": lab}
        if "weight" in ex:
            rec["weight"] = ex["weight"]
        out.append(rec)
    return out


def dry_run(cfg) -> int:
    train, dev = _load_train_dev(cfg)
    mode = cfg.get("student.class_weights", "balanced")
    weights = _compute_class_weights(train) if mode == "balanced" else None
    pairs = build_pairs(train, weights)

    print(f"[4_train_student] DRY-RUN: train={len(train)} dev={len(dev)}")
    print(f"                 student={cfg.get('student.base_model')}  "
          f"lora_r={cfg.get('student.lora_r')}  lora_alpha={cfg.get('student.lora_alpha')}  "
          f"epochs={cfg.get('student.epochs')}  lr={cfg.get('student.lr')}")

    counts = Counter(coerce_label(r["label"]) for r in train)
    n = sum(counts.values())
    print("                 class distribution (train):")
    for k in ("positive", "neutral", "negative"):
        c = counts.get(k, 0)
        w = weights.get(k, 1.0) if weights else 1.0
        print(f"                   {k:8s}: {c:5d}  ({c / n * 100:5.1f}%)  weight={w:.3f}")

    if pairs:
        print("                 sample prompt:")
        print("                 " + pairs[0]["prompt"].replace("\n", " ")[:120])
        print(f"                 sample completion: {pairs[0]['completion']}")
        if "weight" in pairs[0]:
            print(f"                 sample weight: {pairs[0]['weight']:.3f}")
    return 0


def train(cfg) -> int:
    try:
        import torch
        import torch.nn as nn
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "[4_train_student] missing dependency (need torch/transformers/peft/accelerate).\n"
            f"                 {exc}\n"
            "                 install: pip install torch peft accelerate"
        )

    if not torch.cuda.is_available():
        print("[4_train_student] WARNING: no CUDA detected; training will be very slow on CPU.")

    base_model = cfg.get("student.base_model", "Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train, dev = _load_train_dev(cfg)
    mode = cfg.get("student.class_weights", "balanced")
    weights = _compute_class_weights(train) if mode == "balanced" else None
    if weights:
        print("[4_train_student] class weights:", {k: round(v, 3) for k, v in weights.items()})

    train_pairs = build_pairs(train, weights)
    dev_pairs = build_pairs(dev)  # dev is never weighted

    max_len = int(cfg.get("student.seq_len", 256))
    train_ds = _tokenize(train_pairs, tokenizer, max_len)
    dev_ds = _tokenize(dev_pairs, tokenizer, max_len)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    lora = LoraConfig(
        r=int(cfg.get("student.lora_r", 16)),
        lora_alpha=int(cfg.get("student.lora_alpha", 32)),
        lora_dropout=float(cfg.get("student.lora_dropout", 0.05)),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    class Collator:
        """Pad input_ids/labels; stack per-sample ``weight`` when present."""

        def __call__(self, examples: list[dict]) -> dict:
            batch = tokenizer.pad(
                {"input_ids": [e["input_ids"] for e in examples]},
                padding=True, return_tensors="pt",
            )
            labels = [e["labels"] for e in examples]
            m = max(len(l) for l in labels)
            batch["labels"] = torch.tensor(
                [l + [-100] * (m - len(l)) for l in labels], dtype=torch.long,
            )
            if "weight" in examples[0]:
                batch["weight"] = torch.tensor(
                    [e["weight"] for e in examples], dtype=torch.float,
                )
            return batch

    class WeightedTrainer(Trainer):
        """Trainer that multiplies each sample's token losses by its class weight."""

        def compute_loss(self, model, inputs, return_outputs=False):
            weights = inputs.pop("weight", None)
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            mask = (shift_labels != -100).float()
            if weights is not None:
                loss = loss * weights.to(loss.device).unsqueeze(1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            return (loss, outputs) if return_outputs else loss

    collator = Collator()

    args = TrainingArguments(
        output_dir=str(ROOT / "checkpoints" / base_model.split("/")[-1]),
        num_train_epochs=int(cfg.get("student.epochs", 3)),
        per_device_train_batch_size=int(cfg.get("student.batch_size", 16)),
        gradient_accumulation_steps=int(cfg.get("student.grad_accum", 1)),
        learning_rate=float(cfg.get("student.lr", 2e-4)),
        warmup_ratio=float(cfg.get("student.warmup_ratio", 0.1)),
        weight_decay=float(cfg.get("student.weight_decay", 0.01)),
        logging_steps=10,
        evaluation_strategy="epoch" if dev else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if dev else False,
        seed=int(cfg.get("project.seed", 42)),
        bf16=True,
    )
    trainer_cls = WeightedTrainer if weights else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds or None,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[4_train_student] adapter saved -> {args.output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate data + print plan only")
    args = parser.parse_args()
    cfg = load_config()
    if args.dry_run:
        return dry_run(cfg)
    return train(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
