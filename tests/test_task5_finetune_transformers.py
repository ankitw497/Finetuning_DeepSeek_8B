"""Task 5 — Tests: finetune_transformers.py structure and config logic (offline)."""
import ast
import os
import sys
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT_PATH = os.path.join(PROJECT_ROOT, "finetune_transformers.py")


def test_file_exists():
    assert os.path.isfile(FT_PATH), "finetune_transformers.py missing"


def test_valid_python_syntax():
    with open(FT_PATH) as f:
        source = f.read()
    try:
        ast.parse(source)
    except SyntaxError as e:
        assert False, f"Syntax error in finetune_transformers.py: {e}"


def test_has_use_deepspeed_argument():
    with open(FT_PATH) as f:
        source = f.read()
    assert "use_deepspeed" in source, "finetune_transformers.py must support --use_deepspeed flag"


def test_has_qlora_config():
    with open(FT_PATH) as f:
        source = f.read()
    assert "nf4" in source.lower() or "NF4" in source, "Must use NF4 quantisation"
    assert "load_in_4bit" in source, "Must set load_in_4bit=True"


def test_lora_rank_and_alpha():
    with open(FT_PATH) as f:
        source = f.read()
    assert "r=8" in source or '"r": 8' in source or "'r': 8" in source, "LoRA rank must be 8"
    assert "lora_alpha=16" in source or '"lora_alpha": 16' in source or "'lora_alpha': 16" in source, \
        "LoRA alpha must be 16"


def test_output_dir_is_finetuned_transformers():
    with open(FT_PATH) as f:
        source = f.read()
    assert "finetuned-transformers" in source, "Output dir must be finetuned-transformers/"


def _make_stubs():
    """Inject stub modules with just enough attributes for finetune_transformers.py to load."""
    import enum

    class FakeEnum(enum.Enum):
        CAUSAL_LM = "CAUSAL_LM"

    stub_names = {
        "torch": {"float16": "float16", "cuda": types.SimpleNamespace(max_memory_allocated=lambda: 0)},
        "transformers": {
            "AutoModelForCausalLM": object,
            "AutoTokenizer": object,
            "BitsAndBytesConfig": object,
            "TrainingArguments": object,
        },
        "peft": {
            "LoraConfig": object,
            "TaskType": FakeEnum,
            "get_peft_model": lambda m, c: m,
            "prepare_model_for_kbit_training": lambda m, **kw: m,
        },
        "bitsandbytes": {},
        "trl": {"SFTTrainer": object},
        "accelerate": {},
        "datasets": {"Dataset": types.SimpleNamespace(from_list=lambda x: x)},
    }
    for mod_name, attrs in stub_names.items():
        m = sys.modules.get(mod_name) or types.ModuleType(mod_name)
        for attr, val in attrs.items():
            setattr(m, attr, val)
        sys.modules[mod_name] = m


def test_get_qlora_config_function():
    """Script must expose get_qlora_config()."""
    _make_stubs()
    import importlib.util
    spec = importlib.util.spec_from_file_location("ft_trans_q", FT_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    assert hasattr(mod, "get_qlora_config"), "Must define get_qlora_config()"


def test_get_lora_config_function():
    """Script must expose get_lora_config()."""
    _make_stubs()
    import importlib.util
    spec = importlib.util.spec_from_file_location("ft_trans_l", FT_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    assert hasattr(mod, "get_lora_config"), "Must define get_lora_config()"
