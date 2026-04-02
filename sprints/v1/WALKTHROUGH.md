# Sprint v1 — Walkthrough

## Summary

This sprint built a complete, end-to-end QLoRA fine-tuning pipeline for **DeepSeek-R1-0528-Qwen3-8B** targeting C coding tasks on a Tesla T4 16 GB GPU. The pipeline includes a hand-crafted synthetic instruction dataset (60 samples across 5 C domains), two separate fine-tuning scripts (HuggingFace Transformers + PEFT and Unsloth), a standalone evaluation harness measuring Pass@1 / BLEU-4 / Exact Match, a DeepSpeed ZeRO-2 config for memory-efficient single-GPU training, and a 15-cell Jupyter notebook that orchestrates all three training runs and produces side-by-side comparison charts. All logic is covered by 44 offline pytest tests (no GPU required).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      experiment.ipynb                               │
│  (master orchestrator — runs everything as subprocesses)            │
│                                                                     │
│  ┌──────────┐   ┌────────────────────┐   ┌──────────────────────┐  │
│  │ Dataset  │   │ Fine-Tune Scripts  │   │   evaluate.py (x4)   │  │
│  │Inspection│   │                    │   │  baseline + 3 ckpts  │  │
│  └──────────┘   │ finetune_          │   └──────────┬───────────┘  │
│                 │ transformers.py    │              │               │
│                 │  (no DeepSpeed)    │              │               │
│                 │                    │         ┌────▼────────────┐  │
│                 │ finetune_          │         │  results_*.json │  │
│                 │ transformers.py    │         │  metrics table  │  │
│                 │  (+ DeepSpeed)     │         │  bar charts     │  │
│                 │                    │         └─────────────────┘  │
│                 │ finetune_          │                              │
│                 │ unsloth.py         │                              │
│                 └────────┬───────────┘                             │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
  finetuned-         finetuned-        finetuned-
  transformers/   transformers-ds/      unsloth/
  (LoRA adapters)  (LoRA adapters)   (LoRA adapters)
  run_meta.json    run_meta.json     run_meta.json

─────────────────────────────────────────────────────

SHARED INPUTS (read by all scripts):

  data/c_coding_dataset.json          configs/ds_config.json
  ┌────────────────────────┐          ┌────────────────────┐
  │ "train": [50 samples]  │          │ ZeRO Stage 2       │
  │ "eval":  [10 samples]  │          │ fp16 enabled       │
  │ {instruction,          │          │ gas=4, micro_bs=1  │
  │  input, output}        │          │ AdamW + WarmupDecay│
  └────────────────────────┘          └────────────────────┘

SHARED PROMPT FORMAT (all scripts use identical template):

  "Below is a C programming task. Write a complete, compilable C solution.
   ### Instruction: {instruction}
   ### Input:       {input}
   ### Response:    {output}"
```

---

## Files Created/Modified

### `data/c_coding_dataset.json`

**Purpose**: Synthetic instruction dataset of 60 C coding problems used for training and evaluation.

**Structure**:
```json
{
  "train": [ <50 samples> ],
  "eval":  [ <10 samples> ]
}
```
Each sample:
```json
{
  "instruction": "Write a C function that swaps two integers using pointers.",
  "input": "",
  "output": "#include <stdio.h>\n\nvoid swap(int *a, int *b) { ... }"
}
```

**How it works**:
The dataset covers five C programming domains that any C competency test would require: **pointer manipulation** (dereferencing, arithmetic, multi-level), **dynamic memory management** (malloc/realloc/free, 2D and 3D arrays), **algorithms** (bubble/merge/quick/insertion sort, BST, linked lists, heap), **struct usage** (typedef, nested structs, self-referential pointers), and **file I/O** (fopen/fclose, binary fread/fwrite, fseek/ftell).

All 50 training samples are complete, compilable C programs — not pseudocode or fragments. The `input` field is always empty (tasks are self-contained), which keeps the instruction format simple for the model to learn. The 10 eval samples are held-out and deliberately cover the same domains, so evaluation measures generalisation within the domain distribution rather than recall of training examples.

The dataset was sized deliberately small (60 total) because the goal of this sprint is **pipeline demonstration**, not training a production model. Even 3 epochs on 50 samples should produce a measurable Pass@1 improvement since the model learns the `### Response:` prompt format.

---

### `evaluate.py`

**Purpose**: Computes Pass@1, BLEU-4, and Exact Match for any model checkpoint against the eval split, writing results to a JSON file.

**Key Functions**:
- `check_compiles(c_code)` — Writes C to a temp file, calls `gcc`, returns True/False
- `compute_bleu(predictions, references)` — Corpus BLEU-4 via sacrebleu (unigram fallback if not installed)
- `compute_exact_match(predictions, references)` — Fraction of stripped-identical outputs
- `save_results(metrics, output_path)` — Writes the metrics dict to JSON
- `load_model_and_tokenizer(model_path)` — Loads in 4-bit NF4 (matches training quantisation)
- `generate_response(model, tokenizer, prompt)` — Greedy decode, returns only newly generated tokens
- `evaluate(model_path, dataset_path, output_path, split)` — Full evaluation loop

**How it works**:

The most important design decision is `check_compiles()`. Rather than using a language model to judge code correctness (which requires a second model and is expensive), it uses the system's `gcc` compiler as ground truth:

```python
def check_compiles(c_code: str) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as src:
        src.write(c_code)
        src_path = src.name
    out_path = src_path.replace(".c", ".out")
    try:
        result = subprocess.run(
            ["gcc", "-o", out_path, src_path, "-lm", "-w"],
            capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    finally:
        for p in (src_path, out_path):
            if os.path.exists(p): os.unlink(p)
```

The `-w` flag suppresses warnings (only errors fail compilation), `-lm` links the math library (many C programs need `sqrt`/`pow`), and a 15-second timeout handles pathological inputs. Temp files are always cleaned up via `finally`.

For generation, the script decodes only the **newly generated tokens** (not the prompt):
```python
new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
return tokenizer.decode(new_tokens, skip_special_tokens=True)
```
This is critical — without slicing, the decoded string would include the prompt itself, inflating exact-match and BLEU scores artificially.

The script is reusable for baseline, post-Transformers, post-DeepSpeed, and post-Unsloth evaluation with a single `--model_path` swap. Results include a timestamp so JSON files from different runs don't silently overwrite each other's context.

---

### `finetune_transformers.py`

**Purpose**: QLoRA fine-tuning via HuggingFace Transformers + PEFT, with an optional `--use_deepspeed` flag to toggle ZeRO-2 without changing any other code.

**Key Functions**:
- `get_qlora_config()` → `BitsAndBytesConfig` — 4-bit NF4, double quantisation, fp16 compute
- `get_lora_config()` → `LoraConfig` — r=8, α=16, 7 target modules (all attention + MLP projections)
- `get_training_args(output_dir, use_deepspeed, ds_config_path)` → `TrainingArguments`
- `load_train_dataset(dataset_path)` → HuggingFace `Dataset` of formatted prompt strings
- `train(args)` — Full training loop: load → quantise → LoRA → train → save adapter + metadata

**How it works**:

QLoRA works by freezing the base model weights entirely (quantised to 4-bit) and adding a small set of trainable low-rank matrices alongside each attention and MLP layer. The key three-step setup:

```python
# Step 1: Load in 4-bit NF4
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, quantization_config=get_qlora_config(), device_map="auto"
)
model.config.use_cache = False   # required for gradient checkpointing

# Step 2: Prepare frozen layers for k-bit training
#   (casts LayerNorm and embedding to fp32 for numerical stability)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Step 3: Inject trainable LoRA adapters
model = get_peft_model(model, get_lora_config())
```

Without step 2 (`prepare_model_for_kbit_training`), gradient checkpointing fails because the frozen quantised weights cannot propagate gradients. PEFT's helper casts only the non-quantised components (layer norms, embeddings) to fp32 before enabling gradient checkpointing.

The **DeepSpeed toggle** is a single boolean: `get_training_args` sets `deepspeed=ds_config_path` when `--use_deepspeed` is passed, and `deepspeed=None` otherwise. The `deepspeed` launch wrapper handles everything else — no code duplication between the two paths.

The optimizer is `paged_adamw_8bit`: Adam's moment states are stored in CPU DRAM and paged to GPU in blocks, reducing VRAM pressure by ~1.5 GB vs standard Adam on an 8B model.

After training, `run_meta.json` is written alongside the adapter weights, capturing elapsed time, peak GPU memory (MB), and all hyperparameters — consumed by `experiment.ipynb` for the comparison table.

---

### `finetune_unsloth.py`

**Purpose**: Identical QLoRA fine-tuning using Unsloth's `FastLanguageModel` instead of standard PEFT, for a direct speed/memory comparison.

**Key Classes/Functions**:
- `ThroughputCallback` — Lightweight timer that estimates tokens/sec over the training run
- `load_model_with_unsloth(model_name, max_seq_length)` — Loads + patches model via `FastLanguageModel`
- `get_training_args(output_dir)` — Same hyperparameters as Transformers script (no DeepSpeed param)
- `train(args)` — Resets peak memory stats before loading, logs tokens/sec and peak MB

**How it works**:

Unsloth is a drop-in replacement for HuggingFace's model loading + PEFT path that achieves 1.5–2× faster training and ~30–50% lower VRAM by:
1. **Fused attention kernels** — rewrites flash attention in Triton to avoid intermediate tensor allocations
2. **Gradient checkpointing rewrite** — uses `use_gradient_checkpointing="unsloth"` which is memory-tighter than the default HF implementation
3. **RoPE embedding caching** — pre-computes rotary position embeddings rather than recomputing each forward pass

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,          # auto-detects fp16 on T4
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA,
    use_gradient_checkpointing="unsloth",  # key difference
    ...
)
```

**Unsloth does not support DeepSpeed** — its kernel patches conflict with DeepSpeed's process group model. This is not a limitation for a single T4 GPU: Unsloth's own optimisations achieve similar or better VRAM efficiency than ZeRO-2 on a single device.

`ThroughputCallback` estimates tokens/sec as `(num_samples × max_seq_len × epochs) / elapsed_seconds`. This is an upper-bound estimate (actual sequences are shorter due to padding/packing=False), but it's consistent across both scripts so the ratio is meaningful for comparison purposes.

Both training scripts write `run_meta.json` with a `"backend"` field (`"unsloth"` vs absent in Transformers), allowing the notebook to distinguish them programmatically.

---

### `configs/ds_config.json`

**Purpose**: DeepSpeed ZeRO Stage 2 configuration tuned for a single Tesla T4 GPU running 4-bit QLoRA.

**Key settings**:
- `zero_optimization.stage: 2` — partitions optimizer states and gradients across GPUs (even for 1 GPU, reduces peak memory vs stage 0)
- `fp16.enabled: true` — T4 excels at fp16; bf16 is disabled (T4 doesn't support it natively)
- `offload_optimizer: false`, `offload_param: false` — CPU offload not needed; 4-bit model + 8-bit optimizer fits in 16 GB
- `overlap_comm: true` — overlaps gradient communication with backward pass (matters for multi-GPU, no overhead on 1 GPU)
- `train_micro_batch_size_per_gpu: 1`, `gradient_accumulation_steps: 4` → effective batch size = 4

**How it works**:

ZeRO-2 partitions the **optimizer state** (Adam's m and v moment vectors) and **gradients** across all GPUs, but keeps a full copy of model parameters on each GPU. On a single GPU this doesn't reduce parameter memory, but it does reduce gradient memory during the backward pass and enables gradient checkpointing to work more efficiently with DeepSpeed's runtime.

The fp16 dynamic loss scaler settings (`loss_scale_window: 1000`, `initial_scale_power: 16`) are conservative — the scaler will halve the loss scale when it detects overflow and recover slowly, which is appropriate for 4-bit training where gradient norms can be less stable than full-precision runs.

---

### `experiment.ipynb`

**Purpose**: Master 15-cell notebook that orchestrates the full experiment from dataset inspection through all three fine-tuning runs to a final metrics comparison table and two bar charts.

**Cell structure**:

| # | Type | Content |
|---|---|---|
| 1 | Markdown | Title, experiment flow overview |
| 2 | Markdown | "1. Package Imports" header |
| 3 | Code | All package imports + version print |
| 4 | Markdown | "2. Dataset Inspection" header |
| 5 | Code | Load JSON, print sample count + example |
| 6 | Code | Domain distribution bar chart |
| 7 | Markdown | "3. Baseline Evaluation" header |
| 8 | Code | `run_evaluate()` helper + baseline run |
| 9 | Markdown | "4. Fine-Tuning — HF Transformers (no DS)" |
| 10 | Code | `finetune_transformers.py` subprocess (no DS) |
| 11 | Markdown | "5. Fine-Tuning — HF Transformers + DeepSpeed" |
| 12 | Code | `deepspeed --num_gpus=1 finetune_transformers.py --use_deepspeed` subprocess |
| 13 | Markdown | "6. Fine-Tuning — Unsloth" |
| 14 | Code | `finetune_unsloth.py` subprocess |
| 15 | Markdown | "7. Post Fine-Tuning Evaluation" header |
| 16 | Code | `run_evaluate()` on all 3 checkpoints |
| 17 | Markdown | "8. Metrics Comparison" header |
| 18 | Code | DataFrame table with `background_gradient` heatmap |
| 19 | Code | Training performance table (time, VRAM, tokens/sec) |
| 20 | Code | Eval metrics bar chart → `metrics_comparison.png` |
| 21 | Code | Training efficiency bar chart → `efficiency_comparison.png` |
| 22 | Markdown | Summary table + key takeaways |

**How it works**:

Each fine-tuning script is launched as a **subprocess** rather than imported directly. This is intentional: running them in-process would share the notebook kernel's GPU context, making peak memory measurements unreliable and risking OOM from cumulative allocations. The subprocess approach gives each run a clean GPU context.

The `run_evaluate()` helper is defined once and reused for all four evaluation calls (baseline + 3 checkpoints), keeping the notebook DRY:

```python
def run_evaluate(model_path: str, output_file: str) -> dict:
    cmd = [sys.executable, "evaluate.py",
           "--model_path", model_path, "--output", output_file, ...]
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(output_file) as f:
        return json.load(f)
```

The comparison table uses pandas `background_gradient` (green = better, red = worse) applied column-wise, so a reader can instantly see which run performed best on each metric without reading numbers.

---

### `requirements.txt`

**Purpose**: Pinned package versions for reproducible T4/CUDA 11.8 environments, with install-order instructions in comments.

**Key pinned versions**:

| Package | Version | Why pinned |
|---|---|---|
| `torch` | 2.3.1 | CUDA 11.8 wheel compatibility |
| `transformers` | 4.44.2 | SFTTrainer API stable at this version |
| `peft` | 0.12.0 | QLoRA `prepare_model_for_kbit_training` API |
| `bitsandbytes` | 0.43.3 | CUDA 11.8 compiled binary |
| `trl` | 0.9.6 | SFTTrainer `dataset_text_field` param |
| `deepspeed` | 0.15.1 | ZeRO-2 + fp16 stable |
| `accelerate` | 0.33.0 | Required by both Transformers and DeepSpeed |

Unsloth is intentionally **not** pinned in `requirements.txt` because its T4/CUDA 11.8 install requires a special URL: `pip install "unsloth[cu118-torch230] @ git+..."`. Pinning a git commit to `requirements.txt` would make the file uninstallable via a simple `pip install -r`.

---

### `tests/` (7 test files, 44 tests total)

**Purpose**: pytest test suite covering all sprint deliverables. All tests run offline — no GPU, no model downloads, no network.

| File | Tests | What they verify |
|---|---|---|
| `test_task1_structure.py` | 5 | data/ and configs/ dirs exist; requirements.txt exists, is non-empty, and lists all required packages |
| `test_task2_dataset.py` | 7 | JSON valid; train=50, eval=10; all samples have {instruction, input, output}; outputs contain C keywords; all 5 domains covered |
| `test_task3_evaluate.py` | 5 | evaluate.py exists; compute_bleu/compute_exact_match/check_compiles present; exact_match returns correct fraction (2/3); check_compiles returns True for valid C and False for garbage; save_results writes valid JSON with required keys |
| `test_task4_dsconfig.py` | 6 | ds_config.json valid JSON; ZeRO stage=2; fp16=True; no CPU offload; gradient_accumulation_steps present; train_batch_size = gas × micro_bs |
| `test_task5_finetune_transformers.py` | 8 | File exists; valid Python syntax; --use_deepspeed in source; nf4/load_in_4bit present; r=8, alpha=16 present; output dir is finetuned-transformers; get_qlora_config and get_lora_config importable |
| `test_task6_finetune_unsloth.py` | 7 | File exists; valid Python syntax; FastLanguageModel referenced; LORA_R=8, LORA_ALPHA=16 constants; output dir is finetuned-unsloth; "peak" and "tokens" and "sec" in source; 512 in source |
| `test_task7_notebook.py` | 6 | experiment.ipynb valid JSON; has cells key + nbformat; has both code and markdown cells; all 7 required sections present in cell source; torch/transformers/peft/datasets/evaluate/matplotlib all imported; ≥10 cells |

**Testing approach for GPU-dependent code**: The training and evaluation scripts import GPU libraries (`torch`, `transformers`, `peft`, `bitsandbytes`) at module level. To test their structure without a GPU, the test files use `sys.modules` injection to stub these imports before loading the scripts:

```python
def _make_stubs():
    for mod_name, attrs in stub_names.items():
        m = sys.modules.get(mod_name) or types.ModuleType(mod_name)
        for attr, val in attrs.items():
            setattr(m, attr, val)
        sys.modules[mod_name] = m
```

This lets tests call `importlib.util.module_from_spec` to load the script and assert that `hasattr(mod, "get_qlora_config")` etc., without any CUDA device present.

---

## Data Flow

```
data/c_coding_dataset.json (60 samples)
        │
        ├──► evaluate.py (base model)
        │        │
        │        ├── format_prompt() × 10 eval samples
        │        ├── generate_response() → raw C code string
        │        ├── check_compiles() → gcc subprocess → bool
        │        ├── compute_bleu() → sacrebleu corpus BLEU-4
        │        └── compute_exact_match() → fraction
        │                   │
        │              results_baseline.json
        │
        ├──► finetune_transformers.py (no DS)
        │        │
        │        ├── format_sample() × 50 train samples
        │        │     → "### Instruction / Input / Response" strings
        │        ├── BitsAndBytesConfig (NF4 4-bit)
        │        ├── prepare_model_for_kbit_training()
        │        ├── LoraConfig (r=8, 7 modules)
        │        ├── SFTTrainer.train() × 3 epochs
        │        └── save_model() → adapter_model.safetensors
        │                   │
        │          finetuned-transformers/ + run_meta.json
        │
        ├──► finetune_transformers.py (+ DeepSpeed)
        │        │ (same flow, deepspeed= set in TrainingArguments,
        │        │  launched via `deepspeed --num_gpus=1`)
        │          finetuned-transformers-ds/ + run_meta.json
        │
        ├──► finetune_unsloth.py
        │        │
        │        ├── FastLanguageModel.from_pretrained()
        │        │     (patches attention + GC kernels)
        │        ├── FastLanguageModel.get_peft_model()
        │        ├── SFTTrainer.train() × 3 epochs
        │        └── ThroughputCallback → tokens/sec
        │                   │
        │          finetuned-unsloth/ + run_meta.json
        │
        └──► evaluate.py × 3 (one per checkpoint)
                     │
                results_ft_transformers.json
                results_ft_ds.json
                results_ft_unsloth.json
                     │
             experiment.ipynb
                     │
                     ├── metrics DataFrame (Pass@1, BLEU-4, Exact Match)
                     ├── perf DataFrame (time, VRAM, tokens/sec)
                     ├── metrics_comparison.png
                     └── efficiency_comparison.png
```

---

## Test Coverage

- **Unit tests**: 44 tests across 7 files
  - Structure/schema validation: 18 tests (dirs, files, JSON fields, package lists)
  - Pure logic: 5 tests (exact_match fraction, gcc compile check True/False, save_results schema)
  - Static analysis (AST + source inspection): 21 tests (syntax validity, required functions, hyperparameter values, required strings)
- **Integration tests**: 0 — not run (GPU required; scripts are exercised via notebook cells)
- **E2E tests**: 0 — planned for v2 (actual training + eval round-trip with a small toy model)

---

## Security Measures

- `check_compiles()` uses a **fixed compiler command** (`gcc`) with no user-controlled arguments injected — the only user-provided content goes into a temp file, not the shell command string. No shell=True is used.
- `subprocess.run` calls in the notebook use **list form** (not shell strings), preventing shell injection.
- Temp files from `check_compiles()` are always deleted in a `finally` block regardless of outcome.
- The dataset file is read with `json.load()` (not `eval()`), preventing code execution from malformed dataset files.

---

## Known Limitations

**Dataset**
- 60 samples is far too small for meaningful generalisation improvements — expect noisy metrics. The point of this sprint is pipeline validation, not SOTA results.
- All samples have empty `"input"` fields. Tasks requiring an explicit input (e.g., "given this partially written function...") are not represented.
- No deduplication check — some algorithmic patterns (e.g., sorting) may appear conceptually similar across samples.

**Evaluation**
- `check_compiles()` only checks that the code **compiles**, not that it **runs correctly**. A function that compiles but returns wrong answers still scores as Pass@1=1. A true Pass@k evaluation would require test harnesses per problem.
- BLEU-4 is a poor metric for code — it rewards token-level overlap regardless of semantics. A program that swaps the variable names scores low despite being functionally identical.
- Exact match will be near-zero for any model (even perfect code rarely matches the reference character-for-character). It's included for completeness.

**Training**
- `total_num_steps: 150` is hardcoded in `ds_config.json`'s scheduler. If the dataset size or epoch count changes, this needs updating manually.
- Unsloth's `ThroughputCallback` estimates tokens/sec from `num_samples × max_seq_len × epochs` — this overestimates actual throughput since real sequences are shorter than `max_seq_length=512`. The ratio between runs is meaningful; the absolute number is not.
- `packing=False` in SFTTrainer means short sequences waste padding tokens. Enabling packing would improve GPU utilisation but requires careful handling of cross-sample attention masking.
- The model is not merged after fine-tuning — inference must load both the base model (4-bit) and the LoRA adapter. Merging would reduce inference VRAM by eliminating the adapter overhead.

**Notebook**
- Fine-tuning runs are launched as subprocesses from the notebook kernel. If the kernel is restarted mid-run, the subprocess continues but its stdout is lost. The `run_meta.json` files persist and can be read manually.
- The comparison cells use `'ft_no_ds_meta' in dir()` guard checks rather than proper state management. If cells are run out of order, these produce NaN silently.

---

## What's Next (v2 Priorities)

1. **Functional correctness evaluation** — replace the compile-only Pass@1 with a proper test runner that executes each generated program against expected stdin/stdout pairs. This is the most impactful improvement to evaluation quality.

2. **Merge LoRA adapters** — add a `merge_and_save.py` script that merges the trained adapters into the base model weights, producing a standalone model that doesn't require PEFT at inference time. This simplifies deployment and reduces inference VRAM.

3. **Larger dataset** — expand to 500–1000 samples using a mix of hand-curated and LLM-generated examples, with deduplication and domain balancing. Consider sourcing from public C repositories (filtered for compilability).

4. **Training metrics logging** — integrate WandB or TensorBoard to capture per-step loss curves, enabling comparison of Transformers vs Unsloth convergence behaviour.

5. **Integration tests** — add a pytest test that runs a full training loop on a tiny toy model (e.g., `facebook/opt-125m`) for 1 step to verify end-to-end script execution without a full T4 run.

6. **Fix `ds_config.json` `total_num_steps`** — derive this dynamically from `(num_samples / micro_bs) * epochs` rather than hardcoding 150, so the scheduler aligns correctly with the actual training duration.
