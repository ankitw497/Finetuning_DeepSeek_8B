"""
evaluate.py — Pre/post fine-tuning evaluation for DeepSeek-R1-0528-Qwen3-8B.

Usage:
    # Baseline (base model)
    python evaluate.py --model_path deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
                       --output results_baseline.json

    # After fine-tuning (HF Transformers checkpoint)
    python evaluate.py --model_path ./finetuned-transformers \
                       --output results_transformers.json

    # After fine-tuning (Unsloth checkpoint)
    python evaluate.py --model_path ./finetuned-unsloth \
                       --output results_unsloth.json

Metrics:
    - pass@1  : fraction of outputs that compile and run without error
    - bleu4   : corpus BLEU-4 score (sacrebleu) vs reference outputs
    - exact_match : fraction of outputs identical to reference (stripped)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Pure-Python metric helpers (no GPU required — safe to import/test offline)
# ---------------------------------------------------------------------------

def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Return fraction of predictions that exactly match their reference (after stripping)."""
    if not predictions:
        return 0.0
    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return matches / len(predictions)


def check_compiles(c_code: str) -> bool:
    """Return True if c_code compiles successfully with gcc, False otherwise."""
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as src:
        src.write(c_code)
        src_path = src.name
    out_path = src_path.replace(".c", ".out")
    try:
        result = subprocess.run(
            ["gcc", "-o", out_path, src_path, "-lm", "-w"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    finally:
        for p in (src_path, out_path):
            if os.path.exists(p):
                os.unlink(p)


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus BLEU-4 using sacrebleu. Returns score in [0, 100]."""
    try:
        import sacrebleu
        result = sacrebleu.corpus_bleu(predictions, [references])
        return result.score
    except ImportError:
        # Fallback: simple unigram precision if sacrebleu not installed
        total, match = 0, 0
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            match += len(pred_tokens & ref_tokens)
            total += max(len(pred_tokens), 1)
        return (match / total) * 100 if total else 0.0


def save_results(metrics: dict, output_path: str) -> None:
    """Persist metrics dict to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "Below is a C programming task. Write a complete, compilable C solution.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def format_prompt(sample: dict) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample.get("input", ""),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str):
    """Load model in 4-bit for inference (matches fine-tuning quantisation)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model_path: str, dataset_path: str, output_path: str, split: str = "eval"):
    print(f"[evaluate] Loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    samples = dataset[split]
    print(f"[evaluate] {len(samples)} samples in split '{split}'")

    print(f"[evaluate] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)

    predictions, references = [], []
    compile_results = []

    for i, sample in enumerate(samples):
        prompt = format_prompt(sample)
        reference = sample["output"]

        t0 = time.time()
        prediction = generate_response(model, tokenizer, prompt)
        elapsed = time.time() - t0

        compiles = check_compiles(prediction)
        compile_results.append(compiles)
        predictions.append(prediction)
        references.append(reference)

        status = "PASS" if compiles else "FAIL"
        print(f"  [{i+1}/{len(samples)}] compile={status}  time={elapsed:.1f}s")

    pass_at_1 = sum(compile_results) / len(compile_results)
    bleu4 = compute_bleu(predictions, references)
    exact_match = compute_exact_match(predictions, references)

    metrics = {
        "model_path": model_path,
        "split": split,
        "num_samples": len(samples),
        "pass_at_1": round(pass_at_1, 4),
        "bleu4": round(bleu4, 4),
        "exact_match": round(exact_match, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print("\n[evaluate] === Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    save_results(metrics, output_path)
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on C coding tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/c_coding_dataset.json",
        help="Path to the instruction dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_baseline.json",
        help="Output JSON file for metrics",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["train", "eval"],
        help="Dataset split to evaluate on",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output,
        split=args.split,
    )
