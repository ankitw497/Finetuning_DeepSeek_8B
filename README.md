# DeepSeek-R1-0528-Qwen3-8B — C Coding Fine-Tuning Pipeline

End-to-end QLoRA fine-tuning pipeline for **DeepSeek-R1-0528-Qwen3-8B** on a synthetic C coding instruction dataset.
Compares HuggingFace Transformers vs Unsloth backends, with and without DeepSpeed ZeRO-2, on a Tesla T4 16 GB GPU.

---

## Project Structure

```
.
├── data/
│   └── c_coding_dataset.json      # 50 train + 10 eval synthetic C coding samples
├── configs/
│   └── ds_config.json             # DeepSpeed ZeRO-2 config (T4-safe)
├── evaluate.py                    # Pre/post evaluation: Pass@1, BLEU-4, Exact Match
├── finetune_transformers.py       # QLoRA via HF Transformers + PEFT (±DeepSpeed)
├── finetune_unsloth.py            # QLoRA via Unsloth FastLanguageModel
├── experiment.ipynb               # Master notebook — full experiment top-to-bottom
├── requirements.txt               # Pinned dependencies for T4/CUDA 11.8
└── tests/                         # pytest test suite (44 tests)
```

---

## Hardware Requirements

| Item | Requirement |
|---|---|
| GPU | NVIDIA Tesla T4 (or any GPU with ≥ 16 GB VRAM) |
| CUDA | 11.8+ |
| Python | 3.10+ |
| RAM | 16 GB system RAM recommended |

---

## Environment Setup

### 1. Install PyTorch (CUDA 11.8 wheel)
```bash
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install remaining dependencies
```bash
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Unsloth (T4 / CUDA 11.8 specific)
```bash
pip install "unsloth[cu118-torch230] @ git+https://github.com/unslothai/unsloth.git"
```

### 4. (Optional) Flash Attention for faster inference
```bash
pip install flash-attn --no-build-isolation
```

### 5. HuggingFace login (model may require access)
```bash
huggingface-cli login
```

---

## Dataset

`data/c_coding_dataset.json` — 60 synthetic C coding instruction-output pairs:

| Split | Count | Domains covered |
|---|---|---|
| train | 50 | Pointers, Memory management, Algorithms, Structs, File I/O |
| eval | 10 | Same domains (held-out) |

Schema:
```json
{
  "instruction": "Write a C function that...",
  "input": "",
  "output": "#include <stdio.h>\n..."
}
```

---

## Running the Experiment

### Option A — Jupyter Notebook (recommended)
```bash
jupyter notebook experiment.ipynb
```
Run all cells top-to-bottom. Each section is self-contained with subprocess calls to the `.py` scripts.

---

### Option B — Individual scripts

#### Step 1: Baseline evaluation (base model, no fine-tuning)
```bash
python evaluate.py \
    --model_path deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --output results_baseline.json
```

#### Step 2a: Fine-tune with HF Transformers (no DeepSpeed)
```bash
python finetune_transformers.py \
    --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --output_dir finetuned-transformers
```

#### Step 2b: Fine-tune with HF Transformers + DeepSpeed ZeRO-2
```bash
deepspeed --num_gpus=1 finetune_transformers.py \
    --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --output_dir finetuned-transformers-ds \
    --use_deepspeed \
    --ds_config configs/ds_config.json
```

#### Step 2c: Fine-tune with Unsloth
```bash
python finetune_unsloth.py \
    --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --output_dir finetuned-unsloth
```

#### Step 3: Post fine-tuning evaluation
```bash
# Evaluate each checkpoint
python evaluate.py --model_path finetuned-transformers    --output results_ft_transformers.json
python evaluate.py --model_path finetuned-transformers-ds --output results_ft_ds.json
python evaluate.py --model_path finetuned-unsloth         --output results_ft_unsloth.json
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Pass@1** | Fraction of generated C programs that compile successfully with `gcc` |
| **BLEU-4** | Corpus BLEU-4 score (sacrebleu) vs reference solutions, range [0, 100] |
| **Exact Match** | Fraction of outputs identical to reference after whitespace stripping |

Results are written to JSON files (`results_*.json`) and displayed as a comparison table + bar charts in the notebook.

---

## Training Parameters

All scripts use the same QLoRA hyperparameters for a fair comparison:

| Parameter | Value |
|---|---|
| Quantisation | 4-bit NF4, double quantisation |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Max sequence length | 512 tokens |
| Micro batch size | 1 |
| Gradient accumulation | 4 steps (effective batch = 4) |
| Learning rate | 2e-4, cosine scheduler |
| Epochs | 3 |
| Optimizer | paged_adamw_8bit (Transformers) / adamw_8bit (Unsloth) |

---

## DeepSpeed Config (`configs/ds_config.json`)

- **ZeRO Stage 2** — partitions optimizer states and gradients across GPUs (useful even on 1 GPU for memory savings)
- **fp16 enabled** — T4 has excellent fp16 throughput
- **No CPU offload** — T4 16 GB + 4-bit model fits entirely in VRAM
- Bucket sizes tuned for T4's 320 GB/s memory bandwidth

---

## Unsloth vs Transformers — What to expect on T4

| Metric | HF Transformers | Unsloth |
|---|---|---|
| Training speed | Baseline | ~1.5–2× faster |
| Peak VRAM | Baseline | ~30–50% lower |
| Convergence | Same | Same (identical LoRA config) |
| DeepSpeed compat | Yes (`--use_deepspeed`) | No (uses its own optimisations) |

---

## Running Tests
```bash
python -m pytest tests/ -v
```
Expected: **44 passed** (all offline, no GPU required for tests).

---

## Expected Outputs

After running all steps:

```
results_baseline.json          ← baseline Pass@1, BLEU-4, Exact Match
results_ft_transformers.json   ← post fine-tuning (HF, no DS)
results_ft_ds.json             ← post fine-tuning (HF + DeepSpeed)
results_ft_unsloth.json        ← post fine-tuning (Unsloth)
finetuned-transformers/        ← LoRA adapter weights
finetuned-transformers-ds/     ← LoRA adapter weights (DS run)
finetuned-unsloth/             ← LoRA adapter weights
metrics_comparison.png         ← evaluation bar chart
efficiency_comparison.png      ← training efficiency bar chart
```
