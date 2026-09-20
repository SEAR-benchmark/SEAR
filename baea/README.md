# BAEA: Bona-Fide-Based Acoustic Evidence Agent

BAEA augments a frozen audio language model (ALM) with controlled, deterministic
signal-analysis tools. The ALM may select tools and organize their outputs, but it cannot
execute arbitrary Python, modify acoustic measurements, or access labels and attack types.

## Method

For each recording, BAEA computes up to 35 utterance-level descriptors grouped as:

- spectral: 8 features;
- cepstral: 20 MFCC means;
- prosodic: 4 pitch/voicing features;
- energy: 3 energy/silence features.

Each measurement is compared with a reference distribution estimated exclusively from
bona-fide training speech:

```text
z = (x - median) / (1.4826 * MAD + epsilon)
```

The returned evidence retains the feature name, raw value, reference median and MAD,
robust z-score, direction, and status.

BAEA exposes two policies:

- **Fixed** measures all 35 features and applies a deterministic task-conditioned evidence
  budget (three cues for classification and five for rationale generation).
- **Adaptive** lets a frozen ALM select only from `spectral`, `cepstral`, `prosodic`, and
  `energy`, together with a bounded evidence budget. The controller validates the plan and
  permits at most one follow-up selection after observing the first grounded packet.

Both policies use identical feature definitions, reference statistics, and structured
outputs. No ALM parameter is updated.

## Installation

```bash
python -m pip install -e .
```

## Python API

```python
import json
from baea import run_fixed, run_adaptive

reference = json.load(open("train_bonafide_reference.json"))

fixed = run_fixed("example.flac", reference, objective="classification")

def frozen_alm_planner(audio_path, prompt):
    # Connect a frozen ALM here and return its JSON planning response.
    return '{"categories":["spectral","prosodic"],"budget":3,"follow_up":false}'

adaptive = run_adaptive(
    "example.flac", reference,
    objective="rationale",
    planner=frozen_alm_planner,
)
```

Every result contains `selected_evidence`, `tool_calls`, `planner_calls`, and elapsed time.
Adaptive results additionally report invalid and repeated tool calls.

## Command line

Create a CSV containing only bona-fide training audio paths:

```csv
file_id,audio_path
train_0001,/path/to/train_0001.flac
train_0002,/path/to/train_0002.flac
```

Then build the prior and run Fixed BAEA:

```bash
baea build-reference \
  --manifest bonafide_train.csv \
  --dataset ASVspoof2019_LA \
  --output train_bonafide_reference.json

baea fixed \
  --audio example.flac \
  --reference train_bonafide_reference.json \
  --objective classification \
  --output example.fixed.json
```

The reference builder rejects manifests containing label or attack columns. The supplied
manifest must already be restricted to bona-fide training audio.

## Reference artifact

A valid reference JSON has the following structure:

```json
{
  "reference_source": {"dataset": "...", "split": "train", "label": "bonafide"},
  "statistics": {
    "centroid_mean": {"median": 1960.2, "mad": 155.4, "count": 2580}
  }
}
```

`statistics` must contain all 35 registered features. Evaluation or test statistics must
never be used to build this artifact.

## Leakage controls

- Reference statistics must come only from bona-fide training speech.
- Development data may be used for selecting budgets or thresholds, not normalization.
- Evaluation labels, attack identities, and test statistics are unavailable to the agent.
- Tool names and return fields are label-neutral.
- The complete tool trace is retained for auditing.
- Oracle benchmark evidence is not an input to BAEA and must be evaluated separately.

## Repository scope

This directory contains the clean method core. Dataset-specific launchers and model
adapters should remain thin wrappers around this API. It intentionally excludes audio,
model weights, API credentials, cached features, predictions, and private scoring labels.
