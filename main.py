from __future__ import annotations

import argparse
from pathlib import Path

from data.prepare import prepare_dataset
from evaluation.evaluate import run_evaluation
from evaluation.error_analysis import run_error_analysis
from training.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="lora-finetune-eval pipeline CLI")
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "error-analysis", "all"], help="Pipeline stage to run")
    parser.add_argument("--model-name", default="gpt2", help="Base model name for training and evaluation")
    parser.add_argument("--dataset-name", default="databricks/databricks-dolly-15k", help="Dataset name to load from HuggingFace")
    parser.add_argument("--category", default="closed_qa", help="Dataset category to filter")
    args = parser.parse_args()

    work_dir = Path.cwd()
    if args.stage == "prepare":
        prepare_dataset(work_dir / "data", dataset_name=args.dataset_name, category=args.category)
    elif args.stage == "train":
        train_model(
            data_dir=work_dir / "data",
            checkpoint_dir=work_dir / "checkpoints",
            mlflow_dir=work_dir / "mlruns",
            model_name=args.model_name,
        )
    elif args.stage == "evaluate":
        run_evaluation(
            data_dir=work_dir / "data",
            results_dir=work_dir / "results",
            checkpoint_dir=work_dir / "checkpoints",
            model_name=args.model_name,
            mlflow_dir=work_dir / "mlruns",
        )
    elif args.stage == "error-analysis":
        run_error_analysis(
            results_dir=work_dir / "results",
        )
    elif args.stage == "all":
        prepare_dataset(work_dir / "data", dataset_name=args.dataset_name, category=args.category)
        train_model(
            data_dir=work_dir / "data",
            checkpoint_dir=work_dir / "checkpoints",
            mlflow_dir=work_dir / "mlruns",
            model_name=args.model_name,
        )
        run_evaluation(
            data_dir=work_dir / "data",
            results_dir=work_dir / "results",
            checkpoint_dir=work_dir / "checkpoints",
            model_name=args.model_name,
            mlflow_dir=work_dir / "mlruns",
        )
        run_error_analysis(work_dir / "results")


if __name__ == "__main__":
    main()
