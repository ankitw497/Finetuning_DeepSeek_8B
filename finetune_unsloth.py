"""
finetune_unsloth.py — QLoRA fine-tuning via Unsloth

Model : DeepSeek-R1-0528-Qwen3-8B (4-bit NF4 via Unsloth FastLanguageModel)
Target: C coding instruction following
GPU   : Tesla T4 16 GB

Purpose:
    Compare training speed (tokens/sec) and peak GPU memory usage against
    finetune_transformers.py which uses the standard HuggingFace PEFT path.

Usage:
    python finetune_unsloth.py

Install Unsloth for T4/CUDA 11.8 (do this BEFORE running):
    pip install "unsloth[cu118-torch230] @ git+https://github.com/unslothai/unsloth.git"

Note: Unsloth does NOT support DeepSpeed — it achieves memory efficiency through its
own kernel-level optimisations (fused attention, gradient checkpointing rewrites).
For DeepSpeed comparison use finetune_transformers.py --use_deepspeed.
"""

import argparse
import json
import os
import time

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# Constants — same as finetune_transformers.py for fair comparison
# ---------------------------------------------------------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

PROMPT_TEMPLATE = (
    "Below is a C programming task. Write a complete, compilable C solution.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def format_sample(sample: dict) -> dict:
    return {"text": PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample.get("input", ""),
        output=sample["output"],
    )}


def load_train_dataset(dataset_path: str):
    from datasets import Dataset
    with open(dataset_path) as f:
        data = json.load(f)
    samples = [format_sample(s) for s in data["train"]]
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Unsloth model loading
# ---------------------------------------------------------------------------

def load_model_with_unsloth(model_name: str, max_seq_length: int = MAX_SEQ_LENGTH):
    """
    Load model via Unsloth FastLanguageModel.
    Unsloth handles 4-bit NF4 quantisation internally and patches attention
    kernels for ~2x faster training with ~50% less VRAM vs vanilla HF.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,                  # auto-detect: fp16 on T4
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Attach LoRA via Unsloth's optimised get_peft_model
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-efficient GC
        random_state=42,
        use_rslora=False,
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training args — identical to finetune_transformers.py for fair comparison
# ---------------------------------------------------------------------------

def get_training_args(output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="none",
        optim="adamw_8bit",              # Unsloth recommends adamw_8bit (not paged)
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        group_by_length=True,
    )


# ---------------------------------------------------------------------------
# Token throughput helper
# ---------------------------------------------------------------------------

class ThroughputCallback:
    """Lightweight callback-compatible object to track tokens/sec."""
    def __init__(self):
        self.total_tokens = 0
        self.start_time = None

    def on_train_begin(self):
        self.start_time = time.time()

    def on_train_end(self, total_tokens: int):
        elapsed = time.time() - self.start_time
        self.tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        return self.tokens_per_sec


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    print(f"[finetune_unsloth] Loading model via Unsloth: {args.model_name}")
    print(f"[finetune_unsloth] Output dir: {args.output_dir}")
    print(f"[finetune_unsloth] NOTE: Unsloth does not use DeepSpeed (uses its own optimisations)")

    # Reset peak memory counter
    torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model_with_unsloth(args.model_name, args.max_seq_length)
    print(f"[finetune_unsloth] Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_dataset = load_train_dataset(args.dataset_path)
    print(f"[finetune_unsloth] Train samples: {len(train_dataset)}")

    training_args = get_training_args(args.output_dir)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=False,
    )

    # Estimate total tokens for throughput logging
    total_tokens_estimate = (
        len(train_dataset)
        * args.max_seq_length
        * training_args.num_train_epochs
    )

    throughput_tracker = ThroughputCallback()
    throughput_tracker.on_train_begin()

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    tokens_per_sec = throughput_tracker.on_train_end(total_tokens_estimate)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    print(f"\n[finetune_unsloth] === Performance Summary ===")
    print(f"  Training time        : {elapsed:.1f}s")
    print(f"  Tokens/sec (est.)    : {tokens_per_sec:.1f}")
    print(f"  Peak GPU memory (MB) : {peak_mem_mb:.1f}")

    # Save adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save run metadata for comparison with finetune_transformers.py
    meta = {
        "backend": "unsloth",
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_gpu_mem_mb": round(peak_mem_mb, 1),
        "num_train_epochs": int(training_args.num_train_epochs),
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "max_seq_length": args.max_seq_length,
    }
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[finetune_unsloth] Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning via Unsloth FastLanguageModel"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/c_coding_dataset.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuned-unsloth",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
