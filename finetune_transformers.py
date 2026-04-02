"""
finetune_transformers.py — QLoRA fine-tuning via HuggingFace Transformers + PEFT

Model : DeepSeek-R1-0528-Qwen3-8B (4-bit NF4 QLoRA)
Target: C coding instruction following
GPU   : Tesla T4 16 GB

Usage (no DeepSpeed):
    python finetune_transformers.py

Usage (with DeepSpeed ZeRO-2):
    deepspeed --num_gpus=1 finetune_transformers.py --use_deepspeed

All key hyperparameters are exposed as CLI flags.
"""

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Config factories (exposed so tests can import them without running training)
# ---------------------------------------------------------------------------

def get_qlora_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config — safe for T4 16GB."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    """LoRA adapter config: r=8, alpha=16, targets all attention + MLP projections."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Below is a C programming task. Write a complete, compilable C solution.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def format_sample(sample: dict) -> dict:
    return {"text": PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample.get("input", ""),
        output=sample["output"],
    )}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_train_dataset(dataset_path: str) -> Dataset:
    with open(dataset_path) as f:
        data = json.load(f)
    samples = [format_sample(s) for s in data["train"]]
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Training arguments factory
# ---------------------------------------------------------------------------

def get_training_args(output_dir: str, use_deepspeed: bool, ds_config_path: str) -> TrainingArguments:
    deepspeed_cfg = ds_config_path if use_deepspeed else None

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
        deepspeed=deepspeed_cfg,
        optim="paged_adamw_8bit",        # memory-efficient optimizer for T4
        max_grad_norm=1.0,
        dataloader_num_workers=0,        # avoid multiprocessing issues on T4
        group_by_length=True,            # reduces padding overhead
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    print(f"[finetune_transformers] Loading model: {args.model_name}")
    print(f"[finetune_transformers] DeepSpeed: {args.use_deepspeed}")
    print(f"[finetune_transformers] Output dir: {args.output_dir}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Model — 4-bit QLoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=get_qlora_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False                   # required for gradient checkpointing
    model.config.pretraining_tp = 1

    # 3. Prepare for k-bit training (cast LayerNorms to fp32, etc.)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 4. Attach LoRA adapters
    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    # 5. Dataset
    train_dataset = load_train_dataset(args.dataset_path)
    print(f"[finetune_transformers] Train samples: {len(train_dataset)}")

    # 6. Training args (DeepSpeed toggled via flag)
    training_args = get_training_args(
        output_dir=args.output_dir,
        use_deepspeed=args.use_deepspeed,
        ds_config_path=args.ds_config,
    )

    # 7. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=False,
    )

    # 8. Train + log peak memory
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"\n[finetune_transformers] Training complete in {elapsed:.1f}s")
    print(f"[finetune_transformers] Peak GPU memory: {peak_mem_mb:.1f} MB")

    # 9. Save adapter weights
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 10. Save run metadata
    meta = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "use_deepspeed": args.use_deepspeed,
        "elapsed_seconds": round(elapsed, 1),
        "peak_gpu_mem_mb": round(peak_mem_mb, 1),
        "num_train_epochs": training_args.num_train_epochs,
        "lora_r": 8,
        "lora_alpha": 16,
        "max_seq_length": args.max_seq_length,
    }
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[finetune_transformers] Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning via HuggingFace Transformers + PEFT"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/c_coding_dataset.json",
        help="Path to instruction dataset JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetuned-transformers",
        help="Directory to save adapter weights",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum token length per sample",
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        default=False,
        help="Enable DeepSpeed ZeRO-2 (launch with: deepspeed --num_gpus=1 finetune_transformers.py --use_deepspeed)",
    )
    parser.add_argument(
        "--ds_config",
        type=str,
        default="configs/ds_config.json",
        help="Path to DeepSpeed config JSON (only used when --use_deepspeed is set)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
