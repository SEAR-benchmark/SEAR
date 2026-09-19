"""Recoverable per-audio feature cache with configuration fingerprinting."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .acoustic import (
    FEATURE_NAMES, F0_JUMP_HZ, F0_MAX_HZ, F0_MIN_HZ, HIGH_FREQ_HZ,
    HOP_LENGTH, N_FFT, SAMPLE_RATE, SILENCE_DB, compute_acoustic_features,
)

EXTRACTOR_CONFIG = {
    "schema_version": 1,
    "sample_rate": SAMPLE_RATE,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "silence_db": SILENCE_DB,
    "high_freq_hz": HIGH_FREQ_HZ,
    "f0_min_hz": F0_MIN_HZ,
    "f0_max_hz": F0_MAX_HZ,
    "f0_jump_hz": F0_JUMP_HZ,
    "feature_names": list(FEATURE_NAMES),
}
CONFIG_FINGERPRINT = hashlib.sha256(
    json.dumps(EXTRACTOR_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()[:16]


def cache_path(cache_dir: str | Path, file_id: str) -> Path:
    return Path(cache_dir) / CONFIG_FINGERPRINT / f"{file_id}.json"


def compute_cached(audio_path: str, file_id: str, cache_dir: str | Path) -> dict:
    path = cache_path(cache_dir, file_id)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
        if doc.get("config_fingerprint") == CONFIG_FINGERPRINT:
            return doc["features"]
    features = compute_acoustic_features(audio_path, FEATURE_NAMES)
    doc = {
        "file_id": file_id,
        "audio_path": audio_path,
        "config_fingerprint": CONFIG_FINGERPRINT,
        "features": features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{file_id}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return features
