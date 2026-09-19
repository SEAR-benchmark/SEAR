# SEAR

**Spoofing Evidence-Grounded Audio Reasoning for Audio Deepfake Detection**

SEAR evaluates whether audio language models (ALMs) can identify and quantify
signal-level acoustic anomalies and use them as evidence for audio deepfake detection and
forensic rationale generation. This repository provides the implementation of the
**Bona-Fide-Prior Acoustic Evidence Agent (BPAE)** and leakage-safe evaluation utilities.

> The SEAR AQA annotations are distributed separately through Hugging Face. ASVspoof audio
> is not redistributed and must be obtained from its official source.

## Overview

SEAR contains four tasks:

| Task | Capability |
|---|---|
| T1 | Binary authenticity classification |
| T2 | Forgery-cue identification |
| T3 | Acoustic-feature measurement |
| T4 | Forensic-rationale generation |

BPAE leaves the foundation ALM frozen. Deterministic tools compute 35 spectral,
cepstral, prosodic, and energy features and compare them with median/MAD statistics
estimated exclusively from bona-fide training speech. The resulting structured evidence
is then supplied to the ALM for task-conditioned reasoning.

```text
Audio + SEAR question
        |
        v
Fixed or Adaptive evidence policy
        |
        v
Controlled librosa measurements
        |
        v
Bona-fide training median/MAD reference
        |
        v
Structured acoustic deviations + complete tool trace
        |
        v
Frozen ALM response
```

## Policies

- **BPAE-Fixed** computes all 35 features and deterministically selects the strongest
  task-conditioned deviations.
- **BPAE-Adaptive** lets the frozen ALM choose among four whitelisted acoustic categories
  and a bounded evidence budget. The controller validates every request and permits at
  most one follow-up tool-selection step.

Neither policy exposes evaluation labels, attack identities, or test-set statistics.

## Repository structure

```text
.
├── bpae/                     # Installable method package
│   ├── src/bpae/             # Features, prior, tools, policies, binding
│   └── tests/                # Core invariants and leakage checks
├── examples/
│   └── qwen25_omni.py        # Frozen Qwen2.5-Omni planner adapter
└── evaluation/
    └── detection_metrics.py  # Label-separated ACC/F1/AUC/EER
```

## Installation

Core tools:

```bash
python -m pip install -e ./bpae
```

Qwen2.5-Omni example:

```bash
python -m pip install -e './bpae[qwen]'
```

## Quick start

Prepare a CSV containing only bona-fide training audio:

```csv
file_id,audio_path
LA_T_000001,/path/to/LA_T_000001.flac
```

Build the reference prior:

```bash
bpae build-reference \
  --manifest bonafide_train.csv \
  --dataset ASVspoof2019_LA \
  --output train_bonafide_reference.json
```

Run BPAE-Fixed:

```bash
bpae fixed \
  --audio example.flac \
  --reference train_bonafide_reference.json \
  --objective classification \
  --output example.fixed.json
```

Run either policy with Qwen2.5-Omni:

```bash
python examples/qwen25_omni.py \
  --policy adaptive \
  --objective rationale \
  --generate-final \
  --audio example.flac \
  --reference train_bonafide_reference.json \
  --output example.adaptive.json
```

## Leakage-safe evaluation

Inference artifacts contain identifiers, model scores, evidence, and tool traces, but not
ground-truth labels or attack types. Labels are joined only afterward:

```bash
python evaluation/detection_metrics.py \
  --scores predictions.jsonl \
  --labels private_labels.jsonl \
  --output detection_metrics.json
```

Choose decision thresholds on development data and reuse them unchanged for evaluation.
Do not report the evaluation-set EER operating point as a deployable threshold.

## Dataset

The [SEAR dataset](https://huggingface.co/datasets/Rosa21/sear) is released on Hugging
Face. Its four configurations (`t1`, `t2`, `t3`, and `t4`) contain the complete AQA
annotations. Audio paths are identifiers aligned with locally obtained ASVspoof data;
source audio is not redistributed here.

## Reproducibility

- Acoustic extraction: 16 kHz, FFT 1024, hop 256.
- Reference: bona-fide training speech only.
- Robust deviation: `(x - median) / (1.4826 * MAD + epsilon)`.
- Default evidence budgets: 3 for classification and 5 for rationale generation.
- Maximum applied evidence budget: 8.
- Foundation ALM parameters are never updated.
- Complete planner and tool-call traces are retained.
- Raw final-generation attempts are retained; only an unambiguous misspelled verdict key
  with a valid `genuine`/`spoof` value is normalized by the output controller.

## Citation

The paper citation will be added when the manuscript is publicly available.

## License

The code is released under the Apache License 2.0. The source audio remains subject to the
terms of its original providers and is not included here.
