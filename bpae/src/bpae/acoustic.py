"""Deterministic computation of the 35 acoustic features used by SEAR.

The definitions intentionally match ``benchmark/1-acoustic_features.py``.
All requested features are computed after a single audio load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 1_024
HOP_LENGTH = 256
N_MFCC = 20
SILENCE_DB = -30.0
HIGH_FREQ_HZ = 2_000.0
F0_MIN_HZ = 50.0
F0_MAX_HZ = 400.0
F0_JUMP_HZ = 3.0

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "spectral": (
        "high_freq_ratio", "bandwidth_mean", "rolloff_mean", "rolloff_std",
        "centroid_mean", "centroid_std", "flatness_mean", "zcr_mean",
    ),
    "cepstral": tuple(f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)),
    "prosodic": ("f0_mean", "f0_std", "f0_jump_rate", "voiced_ratio"),
    "energy": ("rms_mean", "rms_cv", "silence_ratio"),
}
FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS.values() for name in group
)


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def compute_acoustic_features(
    audio_path: str | Path,
    feature_names: Iterable[str] | None = None,
) -> dict[str, float | None]:
    """Compute a controlled subset of canonical features for one audio file."""
    requested = list(FEATURE_NAMES if feature_names is None else feature_names)
    unknown = sorted(set(requested) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"Unknown acoustic feature(s): {', '.join(unknown)}")
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    wanted = set(requested)
    values: dict[str, float | None] = {}

    spectral_names = wanted.intersection(FEATURE_GROUPS["spectral"])
    if spectral_names:
        magnitude = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        if "bandwidth_mean" in wanted:
            x = librosa.feature.spectral_bandwidth(S=magnitude, freq=frequencies)[0]
            values["bandwidth_mean"] = _finite_or_none(np.mean(x))
        if wanted.intersection(("rolloff_mean", "rolloff_std")):
            x = librosa.feature.spectral_rolloff(S=magnitude, sr=SAMPLE_RATE)[0]
            values.update(rolloff_mean=_finite_or_none(np.mean(x)), rolloff_std=_finite_or_none(np.std(x)))
        if wanted.intersection(("centroid_mean", "centroid_std")):
            x = librosa.feature.spectral_centroid(S=magnitude, sr=SAMPLE_RATE)[0]
            values.update(centroid_mean=_finite_or_none(np.mean(x)), centroid_std=_finite_or_none(np.std(x)))
        if "flatness_mean" in wanted:
            x = librosa.feature.spectral_flatness(S=magnitude)[0]
            values["flatness_mean"] = _finite_or_none(np.mean(x))
        if "high_freq_ratio" in wanted:
            high = magnitude[frequencies >= HIGH_FREQ_HZ].sum()
            total = magnitude.sum()
            values["high_freq_ratio"] = _finite_or_none(high / (total + 1e-10))
        if "zcr_mean" in wanted:
            x = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
            values["zcr_mean"] = _finite_or_none(np.mean(x))

    if wanted.intersection(FEATURE_GROUPS["energy"]):
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        if "rms_mean" in wanted:
            values["rms_mean"] = _finite_or_none(np.mean(rms))
        if "rms_cv" in wanted:
            values["rms_cv"] = _finite_or_none(np.std(rms) / (np.mean(rms) + 1e-10))
        if "silence_ratio" in wanted:
            rms_db = librosa.amplitude_to_db(rms)
            values["silence_ratio"] = _finite_or_none(np.mean(rms_db < SILENCE_DB))

    if wanted.intersection(FEATURE_GROUPS["cepstral"]):
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        for i in range(1, N_MFCC + 1):
            name = f"mfcc_{i}_mean"
            if name in wanted:
                values[name] = _finite_or_none(np.mean(mfcc[i - 1]))

    if wanted.intersection(FEATURE_GROUPS["prosodic"]):
        try:
            f0, voiced, _ = librosa.pyin(
                y, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ,
                sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
            )
            voiced = np.asarray(voiced, dtype=bool) if voiced is not None else np.zeros_like(f0, dtype=bool)
            valid = voiced & np.isfinite(f0)
            voiced_f0 = f0[valid]
            if "f0_mean" in wanted:
                values["f0_mean"] = _finite_or_none(np.mean(voiced_f0)) if voiced_f0.size > 5 else None
            if "f0_std" in wanted:
                values["f0_std"] = _finite_or_none(np.std(voiced_f0)) if voiced_f0.size > 5 else None
            if "voiced_ratio" in wanted:
                values["voiced_ratio"] = _finite_or_none(np.mean(voiced))
            if "f0_jump_rate" in wanted:
                adjacent = valid[1:] & valid[:-1]
                diffs = np.abs(np.diff(f0))[adjacent]
                values["f0_jump_rate"] = _finite_or_none(np.mean(diffs > F0_JUMP_HZ)) if diffs.size else None
        except (ValueError, FloatingPointError):
            for name in wanted.intersection(FEATURE_GROUPS["prosodic"]):
                values[name] = None

    return {name: values[name] for name in requested}
