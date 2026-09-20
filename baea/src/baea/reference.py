"""Training-genuine reference statistics and robust deviations."""

from __future__ import annotations

from typing import Mapping

import numpy as np

MAD_SCALE = 1.4826


def build_reference_statistics(
    feature_rows: list[Mapping[str, float | None]],
    feature_names: list[str] | tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for name in feature_names:
        values = np.asarray([
            row[name] for row in feature_rows
            if row.get(name) is not None and np.isfinite(float(row[name]))
        ], dtype=float)
        if values.size == 0:
            raise ValueError(f"No finite genuine-training values for {name}")
        median = float(np.median(values))
        stats[name] = {
            "median": median,
            "mad": float(np.median(np.abs(values - median))),
            "count": int(values.size),
        }
    return stats


def compare_with_reference(
    feature_values: Mapping[str, float | None],
    reference: Mapping[str, Mapping[str, float | int]],
    *,
    epsilon: float = 1e-8,
    anomaly_threshold: float = 2.5,
) -> list[dict]:
    """Return label-neutral robust deviations for measured features."""
    results = []
    for feature, raw_value in feature_values.items():
        if feature not in reference:
            raise KeyError(f"Missing reference statistics for {feature}")
        ref = reference[feature]
        median, mad = float(ref["median"]), float(ref["mad"])
        if raw_value is None or not np.isfinite(float(raw_value)):
            results.append({
                "feature": feature, "value": None, "reference_median": median,
                "reference_mad": mad, "robust_z": None, "direction": "unavailable",
                "status": "unavailable",
            })
            continue
        value = float(raw_value)
        z = (value - median) / (MAD_SCALE * mad + epsilon)
        direction = "high" if z > 0 else "low" if z < 0 else "typical"
        status = f"unusually_{direction}" if abs(z) >= anomaly_threshold else "within_reference_range"
        results.append({
            "feature": feature, "value": value, "reference_median": median,
            "reference_mad": mad, "robust_z": float(z), "direction": direction,
            "status": status,
        })
    return results


def get_top_anomalies(comparisons: list[dict], k: int = 3) -> list[dict]:
    if k < 1:
        raise ValueError("k must be at least 1")
    available = [item for item in comparisons if item.get("robust_z") is not None]
    return sorted(available, key=lambda item: abs(float(item["robust_z"])), reverse=True)[:k]
