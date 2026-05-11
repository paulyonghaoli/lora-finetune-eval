from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from datasets import Dataset, load_dataset

from utils.seeding import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _canonicalize_example(example: Dict[str, Any]) -> Dict[str, str]:
    instruction = example.get("instruction") or example.get("prompt") or ""
    input_text = example.get("context") or example.get("input") or ""
    output = example.get("response") or example.get("output") or example.get("text") or ""
    if not instruction and "text" in example:
        instruction = str(example.get("text", ""))
    return {
        "instruction": str(instruction).strip(),
        "input": str(input_text).strip(),
        "output": str(output).strip(),
        "category": str(example.get("category", "")).strip(),
    }


def _save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summary_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(record["input"].split()) + len(record["output"].split()) for record in records]
    category_counts = Counter(record.get("category", "unknown") or "unknown" for record in records)
    avg_input = np.mean([len(record["input"].split()) for record in records]) if records else 0
    avg_output = np.mean([len(record["output"].split()) for record in records]) if records else 0
    return {
        "total_examples": len(records),
        "avg_input_length": float(avg_input),
        "avg_output_length": float(avg_output),
        "category_distribution": dict(category_counts),
    }


def _split_records(records: List[Dict[str, Any]], seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    set_seed(seed)
    shuffled = records.copy()
    np.random.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_dataset(
    output_dir: Path,
    dataset_name: str = "databricks/databricks-dolly-15k",
    category: str = "closed_qa",
    seed: int = 42,
    dataset: Optional[Dataset] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        logger.info("Loading dataset %s", dataset_name)
        dataset = load_dataset(dataset_name, split="train")

    if "category" in dataset.column_names:
        filtered = dataset.filter(lambda example: example.get("category", "") == category)
    else:
        logger.warning("Dataset does not include a category column; using full dataset")
        filtered = dataset

    records = [_canonicalize_example(record) for record in filtered]
    if not records:
        raise ValueError(f"No examples found for category {category}")

    splits = _split_records(records, seed=seed)
    _save_jsonl(splits["train"], output_dir / "train.jsonl")
    _save_jsonl(splits["val"], output_dir / "val.jsonl")
    _save_jsonl(splits["test"], output_dir / "test.jsonl")

    summary = _summary_statistics(records)
    logger.info("Saved %s records: train=%d val=%d test=%d", category, len(splits["train"]), len(splits["val"]))
    logger.info("Average input length: %.1f tokens", summary["avg_input_length"])
    logger.info("Average output length: %.1f tokens", summary["avg_output_length"])
    logger.info("Category distribution: %s", summary["category_distribution"])
    return splits


if __name__ == "__main__":
    prepare_dataset(Path("data"))
