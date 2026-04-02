"""Task 7 — Tests: experiment.ipynb structure and required sections."""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(PROJECT_ROOT, "experiment.ipynb")

REQUIRED_SECTIONS = [
    "package imports",
    "dataset",
    "baseline",
    "transformers",
    "unsloth",
    "evaluation",
    "comparison",
]


def load_notebook():
    with open(NB_PATH) as f:
        return json.load(f)


def test_notebook_exists():
    assert os.path.isfile(NB_PATH), "experiment.ipynb missing"


def test_valid_notebook_json():
    nb = load_notebook()
    assert "cells" in nb, "Notebook missing 'cells' key"
    assert "nbformat" in nb, "Notebook missing 'nbformat' key"


def test_has_code_and_markdown_cells():
    nb = load_notebook()
    cell_types = {c["cell_type"] for c in nb["cells"]}
    assert "code" in cell_types, "Notebook must have code cells"
    assert "markdown" in cell_types, "Notebook must have markdown cells"


def test_all_required_sections_present():
    nb = load_notebook()
    all_source = " ".join(
        "".join(c["source"]).lower()
        for c in nb["cells"]
    )
    for section in REQUIRED_SECTIONS:
        assert section in all_source, f"Notebook missing section covering '{section}'"


def test_key_imports_present():
    nb = load_notebook()
    import_source = " ".join(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )
    for pkg in ["torch", "transformers", "peft", "datasets", "evaluate", "matplotlib"]:
        assert pkg in import_source, f"Notebook missing import of '{pkg}'"


def test_notebook_cell_count():
    nb = load_notebook()
    assert len(nb["cells"]) >= 10, "Notebook should have at least 10 cells"
