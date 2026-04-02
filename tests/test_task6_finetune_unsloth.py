"""Task 6 — Tests: finetune_unsloth.py structure and config (offline)."""
import ast
import os
import sys
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT_PATH = os.path.join(PROJECT_ROOT, "finetune_unsloth.py")


def test_file_exists():
    assert os.path.isfile(FT_PATH), "finetune_unsloth.py missing"


def test_valid_python_syntax():
    with open(FT_PATH) as f:
        source = f.read()
    try:
        ast.parse(source)
    except SyntaxError as e:
        assert False, f"Syntax error: {e}"


def test_uses_fast_language_model():
    with open(FT_PATH) as f:
        source = f.read()
    assert "FastLanguageModel" in source, "Must use unsloth.FastLanguageModel"


def test_lora_hyperparams_same_as_transformers():
    with open(FT_PATH) as f:
        source = f.read()
    has_r8 = "r=8" in source or '"r": 8' in source or "'r': 8" in source or "LORA_R = 8" in source
    has_alpha16 = ("lora_alpha=16" in source or '"lora_alpha": 16' in source
                   or "'lora_alpha': 16" in source or "LORA_ALPHA = 16" in source)
    assert has_r8, "LoRA rank must be 8"
    assert has_alpha16, "LoRA alpha must be 16"


def test_output_dir_is_finetuned_unsloth():
    with open(FT_PATH) as f:
        source = f.read()
    assert "finetuned-unsloth" in source, "Output dir must be finetuned-unsloth/"


def test_logs_peak_memory_and_tokens_per_sec():
    with open(FT_PATH) as f:
        source = f.read()
    assert "peak" in source.lower() and ("mem" in source.lower() or "memory" in source.lower()), \
        "Must log peak GPU memory"
    assert "tokens" in source.lower() and ("sec" in source.lower() or "per_sec" in source.lower()), \
        "Must log tokens/sec"


def test_max_seq_length_512():
    with open(FT_PATH) as f:
        source = f.read()
    assert "512" in source, "max_seq_length must default to 512"
