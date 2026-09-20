"""Leakage-safe Fixed and Adaptive BAEA orchestration."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Mapping

from .acoustic import FEATURE_GROUPS, FEATURE_NAMES, compute_acoustic_features
from .policies import CATEGORIES, compile_acoustic_evidence
from .reference import compare_with_reference

Planner = Callable[[Path, str], str]

PLAN_PROMPT = """Select controlled acoustic tool categories for {objective}.
Choose only spectral, cepstral, prosodic, energy. Return JSON only:
{{"categories":["..."],"budget":1-5,"follow_up":true_or_false}}.
Do not request labels, attack types, metadata, arbitrary code, or unregistered features."""

FOLLOW_UP_PROMPT = """Review the grounded evidence packet below. If another acoustic
category is needed, return JSON only: {{"categories":["..."],"budget":1-5}}.
Otherwise return an empty categories list. Use only spectral, cepstral, prosodic, energy.
Evidence packet: {packet}"""


def _validate_reference(document: Mapping) -> Mapping:
    source = document.get("reference_source")
    if not isinstance(source, Mapping) or source.get("split") != "train" or source.get("label") != "bonafide":
        raise ValueError("Reference provenance must be bona-fide training speech only")
    statistics = document.get("statistics")
    if not isinstance(statistics, Mapping) or set(FEATURE_NAMES) - set(statistics):
        raise ValueError("Reference artifact does not contain all 35 registered features")
    return statistics


def _measure(audio_path: Path, reference: Mapping, categories: list[str]) -> tuple[list[dict], list[dict]]:
    features = [name for category in categories for name in FEATURE_GROUPS[category]]
    values = compute_acoustic_features(audio_path, features)
    comparisons = compare_with_reference(values, reference)
    calls = [
        {"tool": f"{category}_analysis", "arguments": {"feature_names": list(FEATURE_GROUPS[category])}}
        for category in categories
    ]
    calls.append({
        "tool": "reference_deviation",
        "arguments": {"formula": "(x-median)/(1.4826*MAD+epsilon)", "source": "bonafide_train"},
        "result": comparisons,
    })
    return comparisons, calls


def _parse_plan(text: str, *, default_budget: int) -> dict:
    match = re.search(r"\{.*?\}", text, re.S)
    error = None
    try:
        raw = json.loads(match.group()) if match else {}
    except Exception as exc:
        raw, error = {}, f"{type(exc).__name__}: {exc}"
    requested = raw.get("categories", []) if isinstance(raw, dict) else []
    if not isinstance(requested, list):
        requested, error = [], error or "categories_not_list"
    categories = []
    for item in requested:
        name = str(item).strip().lower()
        if name in CATEGORIES and name not in categories:
            categories.append(name)
    try: budget = max(1, min(int(raw.get("budget", default_budget)), 8))
    except Exception: budget, error = default_budget, error or "invalid_budget"
    return {
        "raw": text, "categories": categories, "budget": budget,
        "follow_up": bool(raw.get("follow_up", False)),
        "invalid_categories": [x for x in requested if str(x).lower() not in CATEGORIES],
        "parse_error": error,
    }


def run_fixed(audio_path: str | Path, reference_document: Mapping, objective: str) -> dict:
    """Measure all features and apply the deterministic task-specific policy."""
    started = time.perf_counter(); path = Path(audio_path)
    comparisons, calls = _measure(path, _validate_reference(reference_document), list(CATEGORIES))
    packet = compile_acoustic_evidence(comparisons, objective, policy="auto")
    return {
        "policy": "fixed", "objective": objective, "selected_evidence": packet["selected_cues"],
        "tool_calls": calls + packet["tool_calls"], "planner_calls": [],
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_adaptive(
    audio_path: str | Path, reference_document: Mapping, objective: str, planner: Planner,
) -> dict:
    """Let a frozen ALM select only whitelisted categories and a bounded budget."""
    started = time.perf_counter(); path = Path(audio_path); default = 3 if objective == "classification" else 5
    first = _parse_plan(planner(path, PLAN_PROMPT.format(objective=objective)), default_budget=default)
    categories = first["categories"] or list(CATEGORIES)
    comparisons, calls = _measure(path, _validate_reference(reference_document), categories)
    packet = compile_acoustic_evidence(
        comparisons, objective, policy="agent", selected_categories=categories,
        evidence_budget=first["budget"],
    )
    plans = [first]
    repeated = 0
    if first["follow_up"]:
        second = _parse_plan(
            planner(path, FOLLOW_UP_PROMPT.format(packet=json.dumps(packet, ensure_ascii=False))),
            default_budget=first["budget"],
        )
        plans.append(second)
        repeated = len(set(categories) & set(second["categories"]))
        added = [x for x in second["categories"] if x not in categories]
        if added:
            categories += added
            comparisons, calls = _measure(path, _validate_reference(reference_document), categories)
            packet = compile_acoustic_evidence(
                comparisons, objective, policy="agent", selected_categories=categories,
                evidence_budget=max(first["budget"], second["budget"]),
            )
    return {
        "policy": "adaptive", "objective": objective, "selected_evidence": packet["selected_cues"],
        "tool_calls": calls + packet["tool_calls"], "planner_calls": plans,
        "invalid_tool_calls": sum(len(x["invalid_categories"]) for x in plans),
        "repeated_tool_calls": repeated,
        "elapsed_seconds": time.perf_counter() - started,
    }
