# lora-finetune-eval

A clean, reproducible Python pipeline for fine-tuning a small language model with LoRA/PEFT and evaluating it rigorously before and after fine-tuning.

## What this repo does

- Prepares a filtered subset of `databricks/databricks-dolly-15k` for question-answering.
- Fine-tunes `gpt2` using LoRA on attention projections.
- Evaluates both the base and fine-tuned models on held-out test data.
- Logs experiments with MLflow and saves artifacts in a reproducible format.

## Project structure

- `data/prepare.py` — download, filter, format, and split the dataset
- `training/train.py` — apply LoRA, train with HuggingFace Trainer, log metrics to MLflow
- `evaluation/evaluate.py` — compute perplexity, ROUGE-L, exact match, format adherence, and length distribution
- `evaluation/error_analysis.py` — cluster failure cases and summarize hypotheses
- `main.py` — CLI for `prepare`, `train`, `evaluate`, and `all`
- `tests/` — unit tests for data preparation, metrics, and MLflow logging

## Getting started

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py prepare
python main.py train
python main.py evaluate
```

Run the full end-to-end pipeline:

```bash
python main.py all
```

## Notes

- Fixed seed: 42 for dataset shuffling, split, and training.
- MLflow tracking data is saved locally under `mlruns/`.
- Checkpoints are saved under `checkpoints/lora_adapter/` and `checkpoints/training_args.json`.

## How to swap model or dataset

- Change `model_name` in `main.py` or call `training.train_model` with `model_name='distilgpt2'`.
- Change `dataset_name` or `category` in `data/prepare.py`.
