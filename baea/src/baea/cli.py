"""Dataset-neutral BAEA command-line interface."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .acoustic import FEATURE_NAMES, compute_acoustic_features
from .agent import run_fixed
from .reference import build_reference_statistics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def build_reference(args: argparse.Namespace) -> None:
    rows = []
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "audio_path" not in reader.fieldnames:
            raise ValueError("Manifest must contain an audio_path column")
        forbidden = {"label", "attack", "attack_type"}.intersection(reader.fieldnames)
        if forbidden:
            raise ValueError(
                "Reference manifest must already contain bona-fide training audio only; "
                f"remove inference-sensitive columns: {sorted(forbidden)}"
            )
        for record in reader:
            audio_path = Path(record["audio_path"])
            rows.append(compute_acoustic_features(audio_path, FEATURE_NAMES))
    if not rows:
        raise ValueError("Reference manifest is empty")
    payload = {
        "schema_version": "baea_reference_v1",
        "reference_source": {
            "dataset": args.dataset, "split": "train", "label": "bonafide",
        },
        "manifest_sha256": _sha256(args.manifest),
        "sample_count": len(rows),
        "feature_names": list(FEATURE_NAMES),
        "statistics": build_reference_statistics(rows, FEATURE_NAMES),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_new(args.output, payload)


def fixed(args: argparse.Namespace) -> None:
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    result = run_fixed(args.audio, reference, args.objective)
    _write_new(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser(prog="baea")
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("build-reference")
    reference.add_argument("--manifest", type=Path, required=True)
    reference.add_argument("--dataset", required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.set_defaults(function=build_reference)
    run = commands.add_parser("fixed")
    run.add_argument("--audio", type=Path, required=True)
    run.add_argument("--reference", type=Path, required=True)
    run.add_argument("--objective", choices=("classification", "rationale"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(function=fixed)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
