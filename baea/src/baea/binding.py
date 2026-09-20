"""Task-aware, label-neutral selection of controlled acoustic evidence."""

from __future__ import annotations

from collections import defaultdict
import math
import re

from .acoustic import FEATURE_GROUPS, FEATURE_NAMES


# Specific aliases precede general ones to avoid mean/std collisions.
FEATURE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rms_cv", (r"rms.*coefficient of variation", r"rms.*energy variation", r"uniform energy dynamics")),
    ("rms_mean", (r"mean rms", r"rms (?:energy )?mean", r"overall rms", r"mean energy")),
    ("silence_ratio", (r"silence frame ratio", r"silent frame ratio", r"silence ratio", r"speech[- ]silence")),
    ("f0_jump_rate", (r"f0 jump", r"pitch jump", r"fundamental frequency jump")),
    ("f0_std", (r"f0 standard deviation", r"f0 variation", r"pitch variation", r"pitch standard deviation")),
    ("f0_mean", (r"mean f0", r"f0 mean", r"mean fundamental frequency", r"mean pitch")),
    ("voiced_ratio", (r"voiced frame ratio", r"voiced ratio", r"proportion of voiced")),
    ("rolloff_std", (r"rolloff.*standard deviation", r"rolloff.*variation", r"rolloff std")),
    ("rolloff_mean", (r"mean spectral rolloff", r"rolloff mean", r"spectral rolloff frequency(?! standard deviation| variation| std)")),
    ("centroid_std", (r"centroid.*standard deviation", r"centroid.*variation", r"centroid std")),
    ("centroid_mean", (r"mean spectral centroid", r"centroid mean", r"spectral centroid(?! standard deviation| variation| std)")),
    ("bandwidth_mean", (r"spectral bandwidth", r"frequency bandwidth")),
    ("flatness_mean", (r"spectral flatness",)),
    ("high_freq_ratio", (r"high[- ]frequency (?:energy )?ratio", r"high[- ]frequency energy")),
    ("zcr_mean", (r"zero[- ]crossing rate", r"\bzcr\b")),
) + tuple(
    (f"mfcc_{i}_mean", (
        rf"\bmfcc[- _]?{i}\b",
        rf"\bmfcc coefficient {i}(?: mean)?\b",
        rf"\b{i}(?:st|nd|rd|th) mfcc\b",
    ))
    for i in range(1, 21)
)

FEATURE_TO_GROUP = {
    feature: group for group, features in FEATURE_GROUPS.items() for feature in features
}

FEATURE_RELIABILITY = defaultdict(lambda: 0.9, {
    "silence_ratio": 0.6,
    "rms_mean": 0.7,
    "rms_cv": 0.8,
    "f0_mean": 0.75,
    "f0_std": 0.75,
    "f0_jump_rate": 0.7,
    "voiced_ratio": 0.8,
})


def map_text_to_features(text: str) -> list[str]:
    """Map natural-language acoustic concepts to a controlled feature whitelist."""
    found = [
        feature for feature in FEATURE_NAMES
        if re.search(rf"(?<![a-z0-9]){re.escape(feature)}(?![a-z0-9])", text.lower())
    ]
    lowered = text.lower()
    for feature, patterns in FEATURE_ALIASES:
        if any(re.search(pattern, lowered) for pattern in patterns):
            if feature not in found:
                found.append(feature)
    return found


def _ranked(comparisons: list[dict]) -> list[dict]:
    return sorted(
        (item for item in comparisons if item.get("robust_z") is not None),
        key=lambda item: min(abs(float(item["robust_z"])), 10.0),
        reverse=True,
    )


def category_balanced(comparisons: list[dict], k: int) -> list[dict]:
    """Select diverse anomalies, initially at most one per feature category."""
    ranked = _ranked(comparisons)
    selected, used_groups = [], set()
    for item in ranked:
        group = FEATURE_TO_GROUP[item["feature"]]
        if group not in used_groups:
            selected.append(dict(item, category=group, rank_score=min(abs(float(item["robust_z"])), 10.0)))
            used_groups.add(group)
        if len(selected) == k:
            return selected
    for item in ranked:
        if item["feature"] not in {x["feature"] for x in selected}:
            selected.append(dict(item, category=FEATURE_TO_GROUP[item["feature"]], rank_score=min(abs(float(item["robust_z"])), 10.0)))
        if len(selected) == k:
            break
    return selected


def select_task_evidence(qa: dict, comparisons: list[dict]) -> dict:
    """Select evidence using only question/options, never gold answers or labels."""
    by_feature = {item["feature"]: item for item in comparisons}
    question_features = map_text_to_features(qa.get("question", ""))
    option_features: dict[str, list[str]] = {
        letter: map_text_to_features(text)
        for letter, text in (qa.get("options") or {}).items()
    }
    task = qa.get("task")
    requested = []
    if task == "rationale_grounding":
        requested = question_features
    elif task == "forgery_cue_identification":
        requested = list(dict.fromkeys(
            feature for letter in sorted(option_features) for feature in option_features[letter]
        ))
    elif task == "confidence_calibration":
        requested = question_features

    evidence = []
    for feature in requested:
        if feature not in by_feature:
            continue
        item = dict(by_feature[feature])
        item["category"] = FEATURE_TO_GROUP[feature]
        item["requested_by_question"] = feature in question_features
        item["candidate_options"] = [
            letter for letter, features in option_features.items() if feature in features
        ]
        decimals = qa.get("metadata", {}).get("decimals")
        if decimals is not None and item.get("value") is not None:
            item["display_value"] = round(float(item["value"]), int(decimals))
        evidence.append(item)

    target_k = {
        "binary_classification": 4,
        "rationale_generation": 5,
        "confidence_calibration": 4,
        "forgery_cue_identification": max(4, len(evidence)),
        "rationale_grounding": max(1, len(evidence)),
    }.get(task, 4)
    if task not in {"rationale_grounding", "forgery_cue_identification"}:
        supplements = category_balanced(comparisons, target_k)
        for item in supplements:
            if item["feature"] not in {x["feature"] for x in evidence}:
                evidence.append(item)
            if len(evidence) >= target_k:
                break

    available = [item for item in comparisons if item.get("robust_z") is not None]
    abs_z = [abs(float(item["robust_z"])) for item in available]
    summary = {
        "requested_features": requested,
        "mapped_option_features": option_features,
        "returned_feature_count": len(evidence),
        "anomalous_feature_count_at_2_5": sum(value >= 2.5 for value in abs_z),
        "maximum_absolute_robust_z": max(abs_z) if abs_z else None,
        "mean_selected_absolute_robust_z": None,
    }
    selected_z = [
        abs(float(item["robust_z"])) for item in evidence
        if item.get("robust_z") is not None
    ]
    summary["mean_selected_absolute_robust_z"] = (
        sum(selected_z) / len(selected_z) if selected_z else None
    )
    return {"evidence": evidence, "summary": summary}


def _numbers(text: str) -> list[tuple[float, int]]:
    # Do not mistake digits embedded in feature identifiers (for example the
    # ``1`` in ``mfcc_1_mean``) for a claimed acoustic value.
    matches = re.findall(r"(?<![A-Za-z0-9_])-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    result = []
    for raw in matches:
        cleaned = raw.replace(",", "")
        decimals = len(cleaned.rsplit(".", 1)[1]) if "." in cleaned else 0
        result.append((float(cleaned), decimals))
    return result


def _claimed_direction(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(low|lower|narrow|reduced|small)\b", lowered):
        return "low"
    if re.search(r"\b(high|higher|wide|elevated|large|increased)\b", lowered):
        return "high"
    return None


def validate_candidate_options(qa: dict, comparisons: list[dict]) -> list[dict]:
    """Verify option claims against measurements without consulting the gold answer."""
    by_feature = {item["feature"]: item for item in comparisons}
    validations = []
    for letter, option_text in sorted((qa.get("options") or {}).items()):
        features = map_text_to_features(option_text)
        claims = _numbers(option_text)
        best = None
        for feature in features:
            item = by_feature.get(feature)
            if not item or item.get("value") is None:
                continue
            measured = float(item["value"])
            for claimed, decimals in claims:
                candidates = [(measured, "native")]
                if 0.0 <= measured <= 1.0 and abs(claimed) > 1.0:
                    candidates.append((measured * 100.0, "percent"))
                compared, representation = min(
                    candidates, key=lambda pair: abs(pair[0] - claimed)
                )
                absolute_error = abs(compared - claimed)
                tolerance = 0.5 * (10 ** -decimals) + 1e-9
                normalized_error = absolute_error / (abs(claimed) + tolerance)
                candidate = (
                    normalized_error,
                    feature,
                    item,
                    claimed,
                    decimals,
                    compared,
                    representation,
                    absolute_error,
                    tolerance,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            validations.append({
                "option": letter, "option_text": option_text,
                "mapped_features": features, "status": "not_verifiable",
                "support_score": 0.0,
            })
            continue
        (normalized_error, feature, item, claimed, decimals, compared,
         representation, absolute_error, tolerance) = best
        z = item.get("robust_z")
        direction_claim = _claimed_direction(option_text)
        direction_match = (
            direction_claim is None or item.get("direction") == direction_claim
        )
        measurement_match = absolute_error <= tolerance
        anomaly_supported = z is not None and abs(float(z)) >= 2.5 and direction_match
        reliability = float(FEATURE_RELIABILITY[feature])
        numeric_score = math.exp(-10.0 * normalized_error)
        anomaly_score = min(abs(float(z)), 10.0) / 10.0 if z is not None else 0.0
        support_score = numeric_score * (0.7 + 0.3 * anomaly_score) * reliability
        validations.append({
            "option": letter, "option_text": option_text,
            "feature": feature, "category": FEATURE_TO_GROUP[feature],
            "claimed_value": claimed, "claimed_decimals": decimals,
            "measured_value": float(item["value"]),
            "compared_value": compared, "value_representation": representation,
            "absolute_error": absolute_error, "normalized_error": normalized_error,
            "measurement_match": measurement_match,
            "claimed_direction": direction_claim,
            "measured_direction": item.get("direction"),
            "direction_match": direction_match,
            "robust_z": z, "anomaly_supported": anomaly_supported,
            "measurement_reliability": reliability,
            "support_score": support_score,
            "status": "supported" if measurement_match and anomaly_supported else "unsupported",
        })
    return validations


def select_verified_candidate(validations: list[dict]) -> dict:
    """Select a cue deterministically, with numeric consistency taking priority.

    Task 2 asks which candidate describes the measured signal.  Robust-z and
    feature reliability are useful evidence annotations, but must not outweigh
    a closer match to the deterministic measurement.  The returned decision is
    label-blind and contains enough detail to audit the controller's choice.
    """
    verifiable = [
        item for item in validations
        if item.get("status") != "not_verifiable"
        and item.get("normalized_error") is not None
    ]
    if not verifiable:
        return {
            "prediction": None, "binding": False, "reason": "no_numeric_claim",
            "best_normalized_error": None, "runner_up_normalized_error": None,
            "error_gap": None,
        }

    ranked = sorted(
        verifiable,
        key=lambda item: (
            float(item["normalized_error"]),
            not bool(item.get("direction_match", True)),
            -abs(float(item.get("robust_z") or 0.0)),
            item["option"],
        ),
    )
    best = ranked[0]
    runner_up_error = (
        float(ranked[1]["normalized_error"]) if len(ranked) > 1 else None
    )
    best_error = float(best["normalized_error"])
    return {
        "prediction": best["option"],
        # A real numeric claim is a deterministic tool result. The ALM may
        # explain it, but the agent controller does not let it overwrite it.
        "binding": True,
        "reason": "closest_deterministic_measurement",
        "best_normalized_error": best_error,
        "runner_up_normalized_error": runner_up_error,
        "error_gap": (
            runner_up_error - best_error if runner_up_error is not None else None
        ),
        "measurement_match": bool(best.get("measurement_match")),
        "direction_match": bool(best.get("direction_match", True)),
    }
