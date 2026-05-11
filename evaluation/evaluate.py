from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.seeding import set_seed
from utils.text import exact_match_rate, format_adherence_rate, normalize_text

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_prompt(example: Dict[str, Any]) -> str:
    prompt = f"### Instruction:\n{example['instruction'].strip()}\n"
    if example.get("input"):
        prompt += f"### Input:\n{example['input'].strip()}\n"
    prompt += "### Response:\n"
    return prompt


def _ensure_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_name: str, adapter_path: Path | None = None, device: str = "cpu") -> tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = _ensure_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if adapter_path is not None and adapter_path.exists():
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    return model, tokenizer


def _batch(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _generate_predictions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: str = "cpu",
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> List[str]:
    results: List[str] = []
    for batch_prompts in _batch(prompts, batch_size):
        tokenized = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        input_lengths = attention_mask.sum(dim=1).tolist()
        for i, length in enumerate(input_lengths):
            generated_ids = output_ids[i, length :]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(decoded)
    return results


def _compute_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    targets: List[str],
    device: str = "cpu",
    batch_size: int = 8,
) -> List[float]:
    results: List[float] = []
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    for batch_prompts, batch_targets in zip(_batch(prompts, batch_size), _batch(targets, batch_size)):
        prompt_texts = [p + t for p, t in zip(batch_prompts, batch_targets)]
        tokenized = tokenizer(prompt_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        labels = input_ids.clone()
        with torch.no_grad():
            prompt_tokenized = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        prompt_lengths = prompt_tokenized["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).tolist()
        for i, length in enumerate(prompt_lengths):
            labels[i, : length] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        label_slice = labels[:, 1:].contiguous()
        loss = loss_fct(logits.view(-1, logits.size(-1)), label_slice.view(-1)).view(label_slice.size())
        token_counts = (label_slice != -100).sum(dim=1).clamp(min=1)
        example_losses = loss.sum(dim=1) / token_counts
        perplexities = torch.exp(example_losses).cpu().tolist()
        results.extend(perplexities)
    return results


def _compute_rouge_l(predictions: List[str], references: List[str]) -> List[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(normalize_text(ref), normalize_text(pred))["rougeL"].fmeasure for pred, ref in zip(predictions, references)]


def compute_rouge_l_scores(predictions: List[str], references: List[str]) -> float:
    scores = _compute_rouge_l(predictions, references)
    return float(np.mean(scores)) if scores else 0.0


def perplexity_from_loss(loss: float) -> float:
    return float(np.exp(loss))


def _write_report(results_dir: Path, metrics: Dict[str, float], records: List[Dict[str, Any]]) -> None:
    report_path = results_dir / "eval_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Evaluation Report\n\n")
        handle.write("## Before / After Comparison\n\n")
        handle.write("| Metric | Base model | Fine-tuned model | Delta |\n")
        handle.write("|---|---|---|---|\n")
        for name in [
            "perplexity",
            "rouge_l",
            "exact_match",
            "format_adherence",
            "mean_length",
        ]:
            handle.write(
                f"| {name.replace('_', ' ').title()} | {metrics[f'base_{name}']:.4f} | {metrics[f'fine_{name}']:.4f} | {metrics[f'delta_{name}']:.4f} |\n"
            )
        handle.write("\n## Example outputs\n\n")
        for example in records[:10]:
            handle.write(f"### Instruction\n{example['instruction']}\n\n")
            if example.get('input'):
                handle.write(f"**Input:** {example['input']}\n\n")
            handle.write(f"**Reference:** {example['output']}\n\n")
            handle.write(f"**Base prediction:** {example['base_prediction']}\n\n")
            handle.write(f"**Fine-tuned prediction:** {example['fine_prediction']}\n\n")
            handle.write("---\n\n")
    logger.info("Wrote evaluation report to %s", report_path)


def run_evaluation(
    data_dir: Path,
    results_dir: Path,
    checkpoint_dir: Path,
    model_name: str = "gpt2",
    mlflow_dir: Path | None = None,
) -> None:
    set_seed(42)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_data = _load_jsonl(Path(data_dir) / "test.jsonl")
    prompts = [_make_prompt(record) for record in test_data]
    references = [record["output"] for record in test_data]

    logger.info("Loading base model for evaluation")
    base_model, tokenizer = _load_model(model_name, device=device)
    logger.info("Loading fine-tuned model for evaluation")
    fine_model, _ = _load_model(model_name, adapter_path=Path(checkpoint_dir) / "lora_adapter", device=device)

    logger.info("Generating base predictions")
    base_predictions = _generate_predictions(base_model, tokenizer, prompts, device=device)
    logger.info("Generating fine-tuned predictions")
    fine_predictions = _generate_predictions(fine_model, tokenizer, prompts, device=device)

    logger.info("Computing perplexities")
    base_perplexities = _compute_perplexity(base_model, tokenizer, prompts, references, device=device)
    fine_perplexities = _compute_perplexity(fine_model, tokenizer, prompts, references, device=device)

    logger.info("Computing ROUGE-L scores")
    base_rouge = _compute_rouge_l(base_predictions, references)
    fine_rouge = _compute_rouge_l(fine_predictions, references)

    exact_match_base = exact_match_rate(base_predictions, references)
    exact_match_fine = exact_match_rate(fine_predictions, references)
    format_base = format_adherence_rate(base_predictions, references)
    format_fine = format_adherence_rate(fine_predictions, references)

    base_lengths = [len(tokenizer.encode(pred, add_special_tokens=False)) for pred in base_predictions]
    fine_lengths = [len(tokenizer.encode(pred, add_special_tokens=False)) for pred in fine_predictions]

    records: List[Dict[str, Any]] = []
    for record, base_pred, fine_pred, base_ppl, fine_ppl in zip(
        test_data, base_predictions, fine_predictions, base_perplexities, fine_perplexities
    ):
        records.append(
            {
                "instruction": record["instruction"],
                "input": record.get("input", ""),
                "output": record["output"],
                "base_prediction": base_pred,
                "fine_prediction": fine_pred,
                "base_perplexity": base_ppl,
                "fine_perplexity": fine_ppl,
                "format_adherence": format_adherence_rate([fine_pred], [record["output"]]) == 1.0,
                "reference_length": len(tokenizer.encode(record["output"], add_special_tokens=False)),
                "fine_length": len(tokenizer.encode(fine_pred, add_special_tokens=False)),
            }
        )

    metrics = {
        "base_perplexity": float(np.mean(base_perplexities)),
        "fine_perplexity": float(np.mean(fine_perplexities)),
        "delta_perplexity": float(np.mean(base_perplexities) - np.mean(fine_perplexities)),
        "base_rouge_l": float(np.mean(base_rouge)),
        "fine_rouge_l": float(np.mean(fine_rouge)),
        "delta_rouge_l": float(np.mean(fine_rouge) - np.mean(base_rouge)),
        "base_exact_match": exact_match_base,
        "fine_exact_match": exact_match_fine,
        "delta_exact_match": exact_match_fine - exact_match_base,
        "base_format_adherence": format_base,
        "fine_format_adherence": format_fine,
        "delta_format_adherence": format_fine - format_base,
        "base_mean_length": float(np.mean(base_lengths)),
        "fine_mean_length": float(np.mean(fine_lengths)),
        "delta_mean_length": float(np.mean(fine_lengths) - np.mean(base_lengths)),
    }

    _save_jsonl(records, results_dir / "eval_outputs.jsonl")
    _write_report(results_dir, metrics, records)
    logger.info("Evaluation completed and saved to %s", results_dir)
