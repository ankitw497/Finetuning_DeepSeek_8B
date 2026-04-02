"""Task 1 — Tests: project structure and requirements.txt validity."""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_data_dir_exists():
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "data")), "data/ directory missing"


def test_configs_dir_exists():
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "configs")), "configs/ directory missing"


def test_requirements_txt_exists():
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "requirements.txt")), "requirements.txt missing"


def test_requirements_txt_not_empty():
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    with open(req_path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    assert len(lines) > 5, "requirements.txt has too few packages"


def test_required_packages_listed():
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    with open(req_path) as f:
        content = f.read().lower()
    required = ["transformers", "peft", "bitsandbytes", "trl", "accelerate", "datasets"]
    for pkg in required:
        assert pkg in content, f"Package '{pkg}' not found in requirements.txt"
