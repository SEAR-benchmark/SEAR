import numpy as np

from bpae.acoustic import FEATURE_NAMES
from bpae.agent import FOLLOW_UP_PROMPT, _parse_plan
from bpae.reference import build_reference_statistics, compare_with_reference, get_top_anomalies


def test_feature_inventory_has_35_unique_names():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 35


def test_robust_deviation_and_ranking():
    rows = [{name: float(i) for name in FEATURE_NAMES} for i in range(1, 6)]
    stats = build_reference_statistics(rows, FEATURE_NAMES)
    values = {name: 3.0 for name in FEATURE_NAMES}
    values[FEATURE_NAMES[0]] = 30.0
    compared = compare_with_reference(values, stats)
    assert get_top_anomalies(compared, 1)[0]["feature"] == FEATURE_NAMES[0]
    assert np.isfinite(get_top_anomalies(compared, 1)[0]["robust_z"])


def test_planner_is_whitelisted_and_bounded():
    plan = _parse_plan(
        '{"categories":["spectral","python","spectral"],"budget":99,"follow_up":true}',
        default_budget=3,
    )
    assert plan["categories"] == ["spectral"]
    assert plan["invalid_categories"] == ["python"]
    assert plan["budget"] == 8
    assert plan["follow_up"] is True


def test_follow_up_prompt_accepts_evidence_packet():
    prompt = FOLLOW_UP_PROMPT.format(packet='{"selected_cues": []}')
    assert '{"categories":["..."],"budget":1-5}' in prompt
    assert '{"selected_cues": []}' in prompt
