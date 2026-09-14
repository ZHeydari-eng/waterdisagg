"""Tests for event segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from waterdisagg.rates import (
    _bridge,
    _hysteresis_mask,
    _runs,
    plateau_of,
    segment,
)


def _sequential_hysteresis(values, on, off):
    """Plain-loop reference implementation of the Schmitt trigger.

    The production version is vectorised for speed over multi-year traces;
    this is the definition it must agree with.
    """
    out = np.zeros(values.size, dtype=bool)
    state = False
    for i, x in enumerate(values):
        if not state and x > on:
            state = True
        elif state and x < off:
            state = False
        out[i] = state
    return out


@pytest.mark.parametrize("seed", range(20))
def test_hysteresis_matches_sequential_reference(seed):
    """Fuzz the vectorised trigger against the loop definition."""
    rng = np.random.default_rng(seed)
    for _ in range(150):
        n = int(rng.integers(1, 80))
        values = rng.normal(0.3, 0.6, n)
        on, off = 0.5, 0.25
        np.testing.assert_array_equal(
            _hysteresis_mask(values, on, off),
            _sequential_hysteresis(values, on, off),
        )


def test_hysteresis_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        _hysteresis_mask(np.zeros(5), on_threshold=0.1, off_threshold=0.9)


def test_hysteresis_prevents_fragmentation():
    """A dip below the on-threshold but above the off-threshold stays on."""
    values = np.array([0.0, 1.0, 1.0, 0.4, 1.0, 1.0, 0.0])
    mask = _hysteresis_mask(values, on_threshold=0.5, off_threshold=0.25)
    assert mask.tolist() == [False, True, True, True, True, True, False]
    assert len(_runs(mask)) == 1


def test_runs_and_bridge():
    mask = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=bool)
    assert _runs(mask) == [(1, 3), (5, 7), (8, 9)]
    # Gaps here are 2 samples (3->5) and 1 sample (7->8). At max_gap=2 both
    # are bridged; at max_gap=1 only the second is; at max_gap=0 none are.
    assert _bridge(_runs(mask), max_gap=2) == [(1, 9)]
    assert _bridge(_runs(mask), max_gap=1) == [(1, 3), (5, 9)]
    assert _bridge(_runs(mask), max_gap=0) == [(1, 3), (5, 7), (8, 9)]
    assert _bridge([], max_gap=5) == []


def test_plateau_ignores_ramp_and_decay():
    """A filtered meter ramps; the plateau estimate must not be dragged down."""
    ramp = np.linspace(0, 2.8, 12)
    hold = np.full(30, 2.8)
    decay = np.linspace(2.8, 0, 8)
    values = np.concatenate([ramp, hold, decay])
    assert plateau_of(values) == pytest.approx(2.8, abs=0.05)


def test_plateau_too_short_returns_nan():
    assert np.isnan(plateau_of(np.array([1.0, 2.0, 3.0])))


def _series(values, start="2024-07-12 00:00:00"):
    index = pd.date_range(start, periods=len(values), freq="1s")
    return pd.Series(values, index=index)


def test_segment_finds_single_event_with_correct_volume():
    # 60 s at 3.0 gpm is exactly 3.0 gallons.
    flow = _series([0.0] * 10 + [3.0] * 60 + [0.0] * 10)
    events = segment(
        flow,
        on_threshold=1.5,
        off_threshold=0.75,
        min_duration_s=5,
        bridge_gap_s=3,
    )
    assert len(events) == 1
    event = events[0]
    assert event.duration_s == 60
    assert event.volume_gal == pytest.approx(3.0)
    assert event.plateau_gpm == pytest.approx(3.0)


def test_segment_drops_short_activations():
    flow = _series([0.0] * 5 + [3.0] * 2 + [0.0] * 5)
    events = segment(
        flow,
        on_threshold=1.5,
        off_threshold=0.75,
        min_duration_s=5,
        bridge_gap_s=0,
    )
    assert events == []


def test_segment_bridges_pause_within_one_event():
    """A shower paused briefly is one event, not two."""
    flow = _series([0.0] * 5 + [2.2] * 30 + [0.0] * 8 + [2.2] * 30 + [0.0] * 5)
    bridged = segment(
        flow, on_threshold=1.1, off_threshold=0.55,
        min_duration_s=5, bridge_gap_s=15,
    )
    assert len(bridged) == 1
    split = segment(
        flow, on_threshold=1.1, off_threshold=0.55,
        min_duration_s=5, bridge_gap_s=2,
    )
    assert len(split) == 2


def test_segment_breaks_events_at_data_gaps():
    """Flow during a dropout is unknown, so an event cannot span one."""
    flow = _series([3.0] * 60)
    valid = pd.Series(True, index=flow.index)
    valid.iloc[25:35] = False
    events = segment(
        flow,
        on_threshold=1.5,
        off_threshold=0.75,
        min_duration_s=5,
        bridge_gap_s=0,
        valid=valid,
    )
    assert len(events) == 2


def test_segment_empty_input():
    assert segment(
        _series([]), on_threshold=1.0, off_threshold=0.5,
        min_duration_s=5, bridge_gap_s=3,
    ) == []
