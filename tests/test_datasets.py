"""Unit tests for Dataset loading, normalization, and Exemplar Retriever."""

import json
import pytest
from pathlib import Path
from src.dataset.loader import DatasetManager, DatasetItem
from src.dataset.retriever import ExemplarRetriever


@pytest.fixture
def temp_dataset_dir(tmp_path):
    d = tmp_path / "datasets"
    d.mkdir()

    # Create dummy CodeAlpaca
    codealpaca_data = [
        {
            "instruction": "Write a Python function to check if a number is prime.",
            "input": "",
            "output": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        }
    ]
    (d / "codealpaca.json").write_text(json.dumps(codealpaca_data), encoding="utf-8")

    # Create dummy LIMA (JSONL format)
    lima_line = json.dumps({
        "conversations": [
            "Explain what makes a good scientific hypothesis.",
            "A good scientific hypothesis must be testable, falsifiable, and grounded in empirical observations."
        ]
    })
    (d / "lima.jsonl").write_text(lima_line + "\n", encoding="utf-8")

    # Create dummy Alpaca
    alpaca_data = [
        {
            "instruction": "Summarize the key events of the Apollo 11 mission.",
            "input": "",
            "output": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969.",
        }
    ]
    (d / "alpaca.json").write_text(json.dumps(alpaca_data), encoding="utf-8")

    return d


def test_dataset_loader(temp_dataset_dir):
    mgr = DatasetManager(storage_dir=temp_dataset_dir)

    code_items = mgr.load_dataset("codealpaca")
    assert len(code_items) == 1
    assert code_items[0].category == "coding"
    assert "is_prime" in code_items[0].output

    lima_items = mgr.load_dataset("lima")
    assert len(lima_items) == 1
    assert lima_items[0].category == "conversation"
    assert "scientific hypothesis" in lima_items[0].instruction

    alpaca_items = mgr.load_dataset("alpaca")
    assert len(alpaca_items) == 1
    assert "Apollo 11" in alpaca_items[0].output


def test_exemplar_retriever(temp_dataset_dir):
    mgr = DatasetManager(storage_dir=temp_dataset_dir)
    retriever = ExemplarRetriever(manager=mgr)
    retriever.load_index()

    # Query matching code
    matches = retriever.find_relevant_exemplars("How to write a prime number function in Python", category="coding")
    assert len(matches) >= 1
    assert matches[0].source == "codealpaca"
