from __future__ import annotations

import re
from typing import Iterable, List


def normalize_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[“”‘’]", "'", normalized)
    return normalized


def exact_match_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    predictions = [normalize_text(p) for p in predictions]
    references = [normalize_text(r) for r in references]
    matches = sum(p == r for p, r in zip(predictions, references))
    return matches / max(len(predictions), 1)


def has_list_format(text: str) -> bool:
    list_markers = [r"^- ", r"^\* ", r"^\d+\. ", r"^• "]
    return any(re.search(marker, line) for line in text.splitlines() for marker in list_markers)


def format_adherence_rate(predictions: Iterable[str], references: Iterable[str]) -> float:
    checks: List[bool] = []
    for pred, ref in zip(predictions, references):
        if has_list_format(ref):
            checks.append(has_list_format(pred))
        else:
            checks.append(True)
    return sum(checks) / max(len(checks), 1)
