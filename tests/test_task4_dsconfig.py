"""Task 4 — Tests: DeepSpeed ZeRO-2 config validity for T4."""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "ds_config.json")


def load_config():
    with open(DS_CONFIG_PATH) as f:
        return json.load(f)


def test_ds_config_exists():
    assert os.path.isfile(DS_CONFIG_PATH), "configs/ds_config.json missing"


def test_zero_stage_is_2():
    cfg = load_config()
    assert cfg.get("zero_optimization", {}).get("stage") == 2, "ZeRO stage must be 2"


def test_fp16_enabled():
    cfg = load_config()
    assert cfg.get("fp16", {}).get("enabled") is True, "fp16 must be enabled for T4"


def test_no_cpu_offload():
    cfg = load_config()
    zero = cfg.get("zero_optimization", {})
    assert not zero.get("offload_optimizer"), "CPU offload must be disabled"
    assert not zero.get("offload_param"), "Param offload must be disabled"


def test_gradient_accumulation_steps_present():
    cfg = load_config()
    assert "gradient_accumulation_steps" in cfg, "gradient_accumulation_steps required"


def test_train_batch_size_consistent():
    cfg = load_config()
    gas = cfg.get("gradient_accumulation_steps", 1)
    micro = cfg.get("train_micro_batch_size_per_gpu", 1)
    total = cfg.get("train_batch_size", gas * micro)
    assert total == gas * micro, (
        f"train_batch_size ({total}) != gas ({gas}) * micro_batch ({micro})"
    )
