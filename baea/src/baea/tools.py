"""Controlled, JSON-serializable SER tool interface."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .acoustic import compute_acoustic_features as _compute
from .reference import compare_with_reference as _compare
from .reference import get_top_anomalies as _top


def compute_acoustic_features(audio_path: str | Path, feature_names: Iterable[str]) -> dict:
    return {"audio_path": str(audio_path), "features": _compute(audio_path, feature_names)}


def compare_with_reference(feature_values: Mapping[str, float | None], reference: Mapping) -> dict:
    return {"comparisons": _compare(feature_values, reference)}


def get_top_anomalies(comparisons: list[dict], k: int = 3) -> dict:
    return {"k": k, "anomalies": _top(comparisons, k=k)}
