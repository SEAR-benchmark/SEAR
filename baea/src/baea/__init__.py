"""Bona-Fide-Based Acoustic Evidence Agent (BAEA)."""

from .acoustic import FEATURE_GROUPS, FEATURE_NAMES, compute_acoustic_features
from .agent import run_adaptive, run_fixed
from .reference import build_reference_statistics, compare_with_reference, get_top_anomalies

__all__ = [
    "FEATURE_GROUPS", "FEATURE_NAMES", "build_reference_statistics",
    "compare_with_reference", "compute_acoustic_features", "get_top_anomalies",
    "run_adaptive", "run_fixed",
]
