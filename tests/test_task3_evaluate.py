"""Task 3 — Tests: evaluate.py structure and logic (offline, no GPU required)."""
import importlib.util
import json
import os
import sys
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(PROJECT_ROOT, "evaluate.py")


def test_evaluate_py_exists():
    assert os.path.isfile(EVAL_PATH), "evaluate.py missing"


def test_evaluate_has_required_functions():
    """evaluate.py must define compute_bleu, compute_exact_match, check_compiles."""
    spec = importlib.util.spec_from_file_location("evaluate_mod", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Stub heavy imports so we don't need GPU packages
    for name in ["torch", "transformers", "peft", "bitsandbytes"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass  # import errors from missing packages are OK; we just check attributes
    for fn in ["compute_bleu", "compute_exact_match", "check_compiles"]:
        assert hasattr(mod, fn), f"evaluate.py missing function '{fn}'"


def test_compute_exact_match_logic():
    """Import and call compute_exact_match without GPU."""
    spec = importlib.util.spec_from_file_location("evaluate_mod2", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    for name in ["torch", "transformers", "peft", "bitsandbytes", "evaluate", "sacrebleu"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    if hasattr(mod, "compute_exact_match"):
        preds = ["int x = 1;", "void foo() {}", "return 0;"]
        refs  = ["int x = 1;", "void foo() {}", "int y = 2;"]
        score = mod.compute_exact_match(preds, refs)
        assert 0.0 <= score <= 1.0, "exact match score out of [0,1]"
        assert abs(score - 2/3) < 1e-6, f"Expected 0.667, got {score}"


def test_check_compiles_returns_bool():
    """check_compiles should return True for valid C and False for invalid."""
    spec = importlib.util.spec_from_file_location("evaluate_mod3", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    for name in ["torch", "transformers", "peft", "bitsandbytes", "evaluate", "sacrebleu"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    if hasattr(mod, "check_compiles"):
        valid_c = "#include <stdio.h>\nint main(){return 0;}"
        invalid_c = "this is not C code @@@"
        result_valid = mod.check_compiles(valid_c)
        result_invalid = mod.check_compiles(invalid_c)
        assert isinstance(result_valid, bool), "check_compiles must return bool"
        assert isinstance(result_invalid, bool), "check_compiles must return bool"
        assert result_valid is True, "Valid C should compile"
        assert result_invalid is False, "Invalid C should not compile"


def test_results_output_schema():
    """evaluate.py must define save_results that writes valid JSON with required keys."""
    spec = importlib.util.spec_from_file_location("evaluate_mod4", EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    for name in ["torch", "transformers", "peft", "bitsandbytes", "evaluate", "sacrebleu"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    if hasattr(mod, "save_results"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name
        metrics = {"pass_at_1": 0.8, "bleu4": 25.3, "exact_match": 0.5, "model_path": "test"}
        mod.save_results(metrics, tmp_path)
        with open(tmp_path) as f:
            data = json.load(f)
        for key in ["pass_at_1", "bleu4", "exact_match"]:
            assert key in data, f"Results JSON missing key '{key}'"
        os.unlink(tmp_path)
