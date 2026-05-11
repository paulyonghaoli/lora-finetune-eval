from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_analysis(results_dir: Path, clusters: Dict[str, List[Dict[str, object]]]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = results_dir / "error_analysis.md"
    with analysis_path.open("w", encoding="utf-8") as handle:
        handle.write("# Error Analysis\n\n")
        for cluster_name, examples in clusters.items():
            handle.write(f"## {cluster_name.replace('_', ' ').title()}\n\n")
            if not examples:
                handle.write("No representative examples were detected.\n\n")
                continue
            for example in examples[:3]:
                handle.write(f"### Instruction\n{example.get('instruction', '')}\n\n")
                if example.get('input'):
                    handle.write(f"**Input:** {example.get('input')}\n\n")
                handle.write(f"**Reference:** {example.get('output', '')}\n\n")
                handle.write(f"**Fine prediction:** {example.get('fine_prediction', '')}\n\n")
                handle.write(f"**Perplexity:** {example.get('fine_perplexity', 0):.2f}\n\n")
                handle.write("---\n\n")
            hypothesis = {
                "high_perplexity": "High perplexity examples often indicate the model is uncertain or the prompt contains uncommon domain terms.",
                "format_violations": "Format errors often occur when the prompt expects a list or structured output but the model produces prose.",
                "very_short_outputs": "Very short outputs may indicate truncation or the model failing to condition on the full instruction.",
            }.get(cluster_name, "Review these examples for patterns in failures.")
            handle.write(f"**Hypothesis:** {hypothesis}\n\n")
    logger.info("Wrote error analysis to %s", analysis_path)


def run_error_analysis(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    eval_records = _load_jsonl(results_dir / "eval_outputs.jsonl")
    if not eval_records:
        raise ValueError("No evaluation records found; run evaluation first.")

    high_perplexity = sorted(eval_records, key=lambda rec: rec.get("fine_perplexity", 0), reverse=True)[:3]
    format_violations = [rec for rec in eval_records if not rec.get("format_adherence", True)][:3]
    very_short = [
        rec
        for rec in eval_records
        if float(rec.get("fine_length", 0)) < max(10.0, 0.5 * float(rec.get("reference_length", 0)))
    ][:3]

    clusters = {
        "high_perplexity": high_perplexity,
        "format_violations": format_violations,
        "very_short_outputs": very_short,
    }
    _write_analysis(results_dir, clusters)
