# Sprint v1 — Tasks

## Status: Complete

---

- [x] Task 1: Create project directory structure and requirements.txt (P0)
  - Acceptance: All directories exist (`data/`, `configs/`); `requirements.txt` lists all packages with pinned versions compatible with T4/CUDA
  - Files: `requirements.txt`, `data/.gitkeep`, `configs/.gitkeep`
  - Completed: 2026-04-02 — Created data/, configs/, tests/ dirs; requirements.txt with pinned versions for T4/CUDA 11.8 incl. transformers, peft, bitsandbytes, trl, accelerate, deepspeed, unsloth install notes, evaluate, sacrebleu, jupyter; 5 tests green

- [x] Task 2: Generate synthetic C coding instruction dataset (P0)
  - Acceptance: `data/c_coding_dataset.json` contains 60 samples (50 train, 10 eval) covering pointers, memory, algorithms, structs, file I/O; valid JSON matching schema `{instruction, input, output}`
  - Files: `data/c_coding_dataset.json`
  - Completed: 2026-04-02 — 50 train + 10 eval samples covering all 5 C domains; 7 tests green

- [x] Task 3: Write evaluation script with pre/post metrics (P0)
  - Acceptance: `evaluate.py` accepts a model path, runs inference on eval split, outputs Pass@1 (compile check), BLEU-4, and Exact Match to a JSON results file; works with both base model and fine-tuned checkpoints
  - Files: `evaluate.py`
  - Completed: 2026-04-02 — compute_bleu (sacrebleu w/ fallback), compute_exact_match, check_compiles (gcc subprocess), save_results, full CLI with --model_path/--output/--split args; 5 tests green

- [x] Task 4: Write DeepSpeed ZeRO-2 config for T4 (P0)
  - Acceptance: `configs/ds_config.json` is valid DeepSpeed config with ZeRO Stage 2, fp16, T4-safe settings; no CPU offload
  - Files: `configs/ds_config.json`
  - Completed: 2026-04-02 — ZeRO-2, fp16, no offload, AdamW+WarmupDecayLR, gas=4/micro_bs=1; 6 tests green

- [x] Task 5: Write QLoRA fine-tuning script via HuggingFace Transformers + PEFT (P0)
  - Acceptance: `finetune_transformers.py` loads model in 4-bit NF4, applies LoRA (r=8, α=16), trains on `data/c_coding_dataset.json` train split, saves checkpoint to `finetuned-transformers/`; accepts `--use_deepspeed` flag to toggle DeepSpeed; runs on T4 without OOM
  - Files: `finetune_transformers.py`
  - Completed: 2026-04-02 — get_qlora_config (NF4 4-bit), get_lora_config (r=8/α=16), SFTTrainer with paged_adamw_8bit, --use_deepspeed flag, run_meta.json output; 8 tests green

- [x] Task 6: Write QLoRA fine-tuning script via Unsloth (P0)
  - Acceptance: `finetune_unsloth.py` uses Unsloth's `FastLanguageModel` for 4-bit loading + LoRA, same hyperparameters as Task 5, saves checkpoint to `finetuned-unsloth/`; logs peak GPU memory and tokens/sec for comparison
  - Files: `finetune_unsloth.py`
  - Completed: 2026-04-02 — FastLanguageModel 4-bit, r=8/α=16 constants, ThroughputCallback logs tokens/sec + peak_mem_mb, run_meta.json with backend tag for comparison; 7 tests green

- [x] Task 7: Create master Jupyter notebook with all imports and experiment flow (P1)
  - Acceptance: all 7 sections present, 15 cells, all key imports, comparison table + 2 charts
  - Files: `experiment.ipynb`
  - Completed: 2026-04-02 — 15-cell notebook covering imports→dataset→baseline→3×fine-tuning→post-eval→comparison table+charts; 6 tests green

- [x] Task 8: Add README with setup instructions and experiment guide (P1)
  - Acceptance: `README.md` covers env setup, dataset description, how to run each script, expected outputs, and hardware notes for T4
  - Files: `README.md`
  - Completed: 2026-04-02 — Full README with env setup (CUDA 11.8 install order), dataset schema, all CLI commands for 4 runs, metric definitions, DS config notes, Unsloth vs Transformers comparison table, expected outputs list
