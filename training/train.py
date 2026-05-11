from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from utils.seeding import set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _build_prompt(instruction: str, input_text: str, output: str = "") -> str:
    prompt = f"### Instruction:\n{instruction.strip()}\n"
    if input_text:
        prompt += f"### Input:\n{input_text.strip()}\n"
    prompt += "### Response:\n"
    if output:
        prompt += output.strip()
    return prompt


def _tokenize_batch(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    texts = [
        _build_prompt(i, inp, out)
        for i, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
    tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
    return tokenized


def _count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


class MLflowLoggerCallback(TrainerCallback):
    def __init__(self, mlflow_dir: Path, tags: Dict[str, str]):
        self.mlflow_dir = mlflow_dir
        self.tags = tags
        self.run = None

    def on_train_begin(self, args, state, control, **kwargs):
        mlflow.set_tracking_uri(f"file://{self.mlflow_dir.resolve()}")
        mlflow.set_experiment("lora-finetune-eval")
        self.run = mlflow.start_run()
        for name, value in self.tags.items():
            mlflow.set_tag(name, value)
        mlflow.log_param("train_batch_size", args.per_device_train_batch_size)
        mlflow.log_param("eval_batch_size", args.per_device_eval_batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_train_epochs", args.num_train_epochs)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None:
            return
        for metric_name, metric_value in logs.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, float(metric_value), step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.run is not None:
            mlflow.end_run()


def train_model(
    data_dir: Path,
    checkpoint_dir: Path,
    mlflow_dir: Path,
    model_name: str = "gpt2",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    max_length: int = 512,
    seed: int = 42,
) -> None:
    set_seed(seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mlflow_dir = Path(mlflow_dir)
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    logger.info("Applying LoRA configuration")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
        line_delimited=True,
    )

    tokenized = dataset.map(
        lambda examples: _tokenize_batch(examples, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        logging_dir=str(checkpoint_dir / "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=seed,
    )

    mlflow_callback = MLflowLoggerCallback(
        mlflow_dir=mlflow_dir,
        tags={
            "model_name": model_name,
            "lora_rank": "8",
            "dataset_name": "databricks/databricks-dolly-15k",
            "split_sizes": "80/10/10",
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        callbacks=[mlflow_callback],
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete")

    peft_dir = checkpoint_dir / "lora_adapter"
    logger.info("Saving LoRA adapter weights to %s", peft_dir)
    model.save_pretrained(peft_dir)

    params = _count_parameters(model)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.resolve()}")
    mlflow.set_experiment("lora-finetune-eval")
    with mlflow.start_run(nested=True):
        mlflow.log_param("trainable_parameters", params["trainable"])
        mlflow.log_param("total_parameters", params["total"])

    with open(checkpoint_dir / "training_args.json", "w", encoding="utf-8") as out_file:
        json.dump({k: v for k, v in vars(training_args).items() if not k.startswith("_")}, out_file, indent=2)

    logger.info("Saved training_args.json and LoRA adapter")


if __name__ == "__main__":
    train_model(Path("data"), Path("checkpoints"), Path("mlruns"))
