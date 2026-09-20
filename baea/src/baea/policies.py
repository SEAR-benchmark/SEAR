"""Controlled compilation of measured acoustic deviations into evidence packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .acoustic import FEATURE_GROUPS


CATEGORIES = tuple(FEATURE_GROUPS)
FEATURE_CATEGORY = {
    feature: category for category, features in FEATURE_GROUPS.items()
    for feature in features
}


@dataclass(frozen=True)
class CompilerConfig:
    classification_budget: int = 3
    rationale_budget: int = 5
    rationale_min_abs_z: float = 2.5


def _ranked(comparisons: Iterable[dict], categories: set[str] | None = None) -> list[dict]:
    rows = []
    for item in comparisons:
        feature = item.get("feature")
        z = item.get("robust_z")
        category = FEATURE_CATEGORY.get(feature)
        if z is None or category is None or (categories is not None and category not in categories):
            continue
        rows.append(dict(item, category=category, anomaly_score=abs(float(z))))
    return sorted(rows, key=lambda item: item["anomaly_score"], reverse=True)


def _category_balanced(ranked: list[dict], budget: int) -> list[dict]:
    selected, used = [], set()
    for item in ranked:
        if item["category"] not in used:
            selected.append(item)
            used.add(item["category"])
        if len(selected) == budget:
            return selected
    for item in ranked:
        if item not in selected:
            selected.append(item)
        if len(selected) == budget:
            break
    return selected


def compile_acoustic_evidence(
    comparisons: list[dict],
    objective: str,
    *,
    policy: str = "auto",
    selected_categories: list[str] | None = None,
    evidence_budget: int | None = None,
    config: CompilerConfig = CompilerConfig(),
) -> dict:
    """Return a label-neutral evidence packet from deterministic measurements.

    ``auto`` uses an objective-specific fixed policy. ``agent`` obeys only
    validated category and budget choices; it never executes model-generated
    code or accepts arbitrary feature names.
    """
    if objective not in {"classification", "rationale"}:
        raise ValueError(f"Unsupported compiler objective: {objective}")
    if policy not in {"auto", "agent"}:
        raise ValueError(f"Unsupported compiler policy: {policy}")

    invalid_categories = sorted(set(selected_categories or ()) - set(CATEGORIES))
    categories = set(selected_categories or CATEGORIES)
    default_budget = (
        config.classification_budget if objective == "classification"
        else config.rationale_budget
    )
    requested_budget = default_budget if evidence_budget is None else int(evidence_budget)
    budget = max(1, min(requested_budget, len(comparisons), 8))
    ranked = _ranked(comparisons, categories)

    if objective == "rationale":
        ranked = [
            item for item in ranked
            if item["anomaly_score"] >= config.rationale_min_abs_z
        ]
    selected = _category_balanced(ranked, budget) if objective == "classification" else ranked[:budget]

    return {
        "objective": objective,
        "policy": policy,
        "selected_cues": selected,
        "selection": {
            "requested_categories": list(selected_categories or CATEGORIES),
            "accepted_categories": sorted(categories & set(CATEGORIES)),
            "invalid_categories": invalid_categories,
            "requested_budget": requested_budget,
            "applied_budget": budget,
            "returned": len(selected),
            "rationale_min_abs_z": (
                config.rationale_min_abs_z if objective == "rationale" else None
            ),
        },
        "tool_calls": [{
            "tool": "compile_acoustic_evidence",
            "arguments": {
                "objective": objective,
                "policy": policy,
                "selected_categories": list(selected_categories or CATEGORIES),
                "evidence_budget": requested_budget,
            },
            "result_count": len(selected),
        }],
    }


def category_overview(comparisons: list[dict]) -> list[dict]:
    """Return one strongest grounded cue per category for a bounded follow-up prompt."""
    result = []
    for category in CATEGORIES:
        rows = _ranked(comparisons, {category})
        if rows:
            item = rows[0]
            result.append({
                key: item[key] for key in (
                    "category", "feature", "value", "reference_median",
                    "robust_z", "direction", "status", "anomaly_score",
                )
            })
    return result
