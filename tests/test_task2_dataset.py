"""Task 2 — Tests: synthetic C coding dataset validity."""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "c_coding_dataset.json")


def load_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def test_dataset_file_exists():
    assert os.path.isfile(DATASET_PATH), "c_coding_dataset.json missing"


def test_dataset_has_train_and_eval_splits():
    data = load_dataset()
    assert "train" in data, "Missing 'train' split"
    assert "eval" in data, "Missing 'eval' split"


def test_train_split_size():
    data = load_dataset()
    assert len(data["train"]) == 50, f"Expected 50 train samples, got {len(data['train'])}"


def test_eval_split_size():
    data = load_dataset()
    assert len(data["eval"]) == 10, f"Expected 10 eval samples, got {len(data['eval'])}"


def test_schema_fields_present():
    data = load_dataset()
    for split in ("train", "eval"):
        for i, sample in enumerate(data[split]):
            assert "instruction" in sample, f"{split}[{i}] missing 'instruction'"
            assert "input" in sample, f"{split}[{i}] missing 'input'"
            assert "output" in sample, f"{split}[{i}] missing 'output'"


def test_outputs_contain_c_code():
    data = load_dataset()
    for split in ("train", "eval"):
        for i, sample in enumerate(data[split]):
            output = sample["output"]
            has_c = any(kw in output for kw in ["#include", "int ", "void ", "return", "{", "}"])
            assert has_c, f"{split}[{i}] output does not look like C code"


def test_covers_required_domains():
    data = load_dataset()
    all_instructions = " ".join(s["instruction"].lower() for s in data["train"] + data["eval"])
    domains = {
        "pointer": ["pointer", "ptr", "dereference", "address"],
        "memory": ["malloc", "free", "memory", "heap", "allocat"],
        "algorithm": ["sort", "search", "algorithm", "binary", "linked list", "stack", "queue", "recursi"],
        "struct": ["struct", "typedef", "member"],
        "file_io": ["file", "fopen", "fclose", "fread", "fwrite", "fprintf", "fscanf"],
    }
    for domain, keywords in domains.items():
        found = any(kw in all_instructions for kw in keywords)
        assert found, f"Dataset does not cover domain '{domain}'"
