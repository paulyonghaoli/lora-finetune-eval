from pathlib import Path

from datasets import Dataset

from data.prepare import prepare_dataset, _split_records


def test_split_records_no_overlap() -> None:
    records = [
        {"instruction": f"inst {i}", "input": "", "output": "out", "category": "closed_qa"}
        for i in range(10)
    ]
    splits = _split_records(records, seed=42)
    train_ids = {record["instruction"] for record in splits["train"]}
    val_ids = {record["instruction"] for record in splits["val"]}
    test_ids = {record["instruction"] for record in splits["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(records)


def test_prepare_dataset_writes_jsonl(tmp_path: Path) -> None:
    sample = [
        {"instruction": "Say hello", "input": "", "output": "Hello world.", "category": "closed_qa"},
        {"instruction": "Describe AI", "input": "", "output": "AI is intelligence.", "category": "closed_qa"},
        {"instruction": "List items", "input": "", "output": "- one\n- two", "category": "closed_qa"},
    ]
    dataset = Dataset.from_list(sample)
    splits = prepare_dataset(tmp_path, dataset_name="unused", category="closed_qa", dataset=dataset)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert sum(len(value) for value in splits.values()) == len(sample)
