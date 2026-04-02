# Sprint v1 — PRD: DeepSeek-R1-0528-Qwen-8B Fine-Tuning Pipeline for C Coding

## Overview
Build an end-to-end QLoRA fine-tuning pipeline for DeepSeek-R1-0528-Qwen-8B on a synthetic C coding instruction dataset. The pipeline evaluates model performance before and after fine-tuning, supports both HuggingFace Transformers and Unsloth backends, and handles parallelization options (with/without DeepSpeed) — all targeting a Tesla T4 16GB GPU.

## Goals
- Generate a small, high-quality synthetic instruction dataset for C coding tasks
- Evaluate baseline model performance (pre-fine-tuning) on C coding benchmarks
- Fine-tune using QLoRA via HuggingFace Transformers (`finetune_transformers.py`)
- Fine-tune using QLoRA via Unsloth (`finetune_unsloth.py`) for speed/memory comparison
- Produce a single Jupyter notebook (`experiment.ipynb`) orchestrating all steps with metrics comparison

## User Stories
- As an ML engineer, I want a ready-made synthetic C coding dataset, so I can test fine-tuning without curating real data
- As a researcher, I want pre/post fine-tuning evaluation metrics, so I can quantify what fine-tuning achieved
- As an engineer, I want two QLoRA backends (Transformers vs Unsloth), so I can compare training speed and memory usage
- As a practitioner with a T4 GPU, I want DeepSpeed and non-DeepSpeed configs, so I can choose based on my setup
- As a user, I want a single notebook that runs the full experiment, so I can reproduce everything end-to-end

## Technical Architecture

### Tech Stack
- **Model**: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (via HuggingFace Hub)
- **Fine-tuning**: QLoRA (4-bit NormalFloat quantization via bitsandbytes)
- **Backends**: HuggingFace Transformers + PEFT, and Unsloth
- **Parallelization**: DeepSpeed ZeRO-2 config (ds_config.json) and single-GPU baseline
- **Evaluation**: Pass@1 (functional correctness), BLEU-4, exact match on held-out C tasks
- **Hardware**: Tesla T4, 16GB VRAM
- **Notebook**: Jupyter (`experiment.ipynb`)

### Component Diagram
```
project/
├── data/
│   └── c_coding_dataset.json        # Synthetic instruction dataset (train/eval split)
├── configs/
│   └── ds_config.json               # DeepSpeed ZeRO-2 config for T4
├── finetune_transformers.py          # QLoRA via HuggingFace Transformers + PEFT
├── finetune_unsloth.py              # QLoRA via Unsloth
├── evaluate.py                       # Pre/post evaluation with metrics
├── experiment.ipynb                  # Master notebook tying all together
└── sprints/v1/
    ├── PRD.md
    └── TASKS.md
```

### Data Flow
```
Synthetic Dataset (c_coding_dataset.json)
        │
        ├──► evaluate.py ──► Baseline metrics (pre-finetune)
        │
        ├──► finetune_transformers.py ──► finetuned-transformers/ (checkpoint)
        │         └── [with/without DeepSpeed flag]
        │
        ├──► finetune_unsloth.py ──► finetuned-unsloth/ (checkpoint)
        │
        └──► evaluate.py ──► Post-finetune metrics (both checkpoints)
                    │
                    └──► experiment.ipynb (metrics comparison table)
```

### Dataset Schema
```json
{
  "instruction": "Write a C function that...",
  "input": "",
  "output": "#include <stdio.h>\n..."
}
```
- ~50 train samples, ~10 eval samples (kept minimal for pipeline demonstration)
- Covers: pointer manipulation, memory management, algorithms, structs, file I/O

### QLoRA Config (both backends)
- **Quantization**: 4-bit NF4, double quantization enabled
- **LoRA rank**: r=8, alpha=16, dropout=0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Max seq length**: 512 tokens
- **Batch size**: 1 (gradient accumulation steps=4, effective batch=4)
- **Learning rate**: 2e-4, cosine scheduler, 3 epochs

### DeepSpeed Config (ds_config.json)
- ZeRO Stage 2 (optimizer + gradient partitioning, no parameter sharding)
- fp16 enabled, T4-compatible
- Offload disabled (T4 VRAM sufficient for 8B at 4-bit)

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| Pass@1 | % of generated C code that compiles and produces correct output |
| BLEU-4 | n-gram overlap vs reference solutions |
| Exact Match | % of outputs identical to reference |
| Training Speed | tokens/sec (Transformers vs Unsloth) |
| Peak GPU Memory | MB during training (logged via `nvidia-smi`) |

## Out of Scope (v2+)
- Multi-GPU / multi-node training
- Real benchmark datasets (HumanEval-C, MBPP-C)
- RLHF / DPO alignment
- Model merging / full fine-tuning
- Serving / inference API
- Automated hyperparameter search

## Dependencies
- HuggingFace account + model access (DeepSeek-R1-0528-Qwen3-8B must be accessible)
- CUDA 11.8+ driver on Tesla T4 machine
- Python 3.10+
- Key packages: `transformers`, `peft`, `bitsandbytes`, `trl`, `unsloth`, `deepspeed`, `evaluate`, `sacrebleu`, `accelerate`
