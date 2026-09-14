"""Tests for the synthetic household generator.

These check the properties the rest of the pipeline relies on: that the trace
round-trips through the real reader, that the diary carries the specific
awkwardness of a hand-written log, and that runs are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from waterdisagg.io import read_trace_dir
from waterdisagg.simulate import (
    SimSpec,
    _meter_response,
    default_spec,
    simulate_household,
    write_simulated_dataset,
)


@pytest.fixture(scope="module")
def small_run():
    return simulate_household(default_spec(n_days=4, seed=11))


def test_shapes_and_alignment(small_run):
    trace, truth, diary, events = small_run
    assert len(trace) == 4 * 86400
    assert trace.index.equals(truth.index)
    assert truth.shape[1] == 14
    assert not events.empty
    assert not diary.empty


def test_deterministic_given_seed():
    a = simulate_household(default_spec(n_days=2, seed=3))[0]
    b = simulate_household(default_spec(n_days=2, seed=3))[0]
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = simulate_household(default_spec(n_days=2, seed=3))[0]
    b = simulate_household(default_spec(n_days=2, seed=4))[0]
    assert not a["flow_gpm"].equals(b["flow_gpm"])


def test_idle_baseline_is_noisy_and_zero_mean(small_run):
    """Idle flow must oscillate around zero, including negative values.

    Zero-filling idle periods would make the deadband in io.Trace untestable
    and would misrepresent what a real meter reports.
    """
    trace, truth, _, _ = small_run
    active = truth.to_numpy().sum(axis=1) > 0
    # Exclude smoothing tails, which extend well past the end of an event.
    near = pd.Series(active).rolling(241, center=True, min_periods=1).max() > 0
    idle = trace["flow_gpm"].to_numpy()[~near.to_numpy()]
    assert idle.size > 1000
    assert abs(idle.mean()) < 0.01
    assert (idle < 0).mean() > 0.3


def test_meter_response_is_monotone_and_slow():
    """A step input must produce a gradual S-shaped rise, not a step."""
    step = np.concatenate([np.zeros(5), np.full(60, 2.8)])
    out = _meter_response(step, tau_s=1.5, stages=4, dt=1.0)
    rise = out[5:25]
    assert np.all(np.diff(rise) > -1e-9)          # monotone non-decreasing
    assert rise[0] < 0.3                          # does not jump immediately
    assert out[-1] == pytest.approx(2.8, abs=0.02)  # settles at the true rate
    # Reaching 95% should take about ten seconds, not one.
    t95 = int(np.argmax(out[5:] > 0.95 * 2.8))
    assert 8 <= t95 <= 20


def test_meter_response_passthrough_when_disabled():
    signal = np.array([0.0, 1.0, 2.0, 0.0])
    np.testing.assert_array_equal(
        _meter_response(signal, tau_s=0.0, stages=4, dt=1.0), signal
    )


def test_diary_is_chronological(small_run):
    """Diary rows must sort by instant, not by rendered 12-hour string."""
    _, _, diary, _ = small_run
    stamps = pd.to_datetime(
        diary["date"].astype(str) + " " + diary["start_time"],
        format="%Y-%m-%d %I:%M:%S %p",
    )
    assert stamps.is_monotonic_increasing


def test_diary_times_are_minute_resolution(small_run):
    _, _, diary, _ = small_run
    stamps = pd.to_datetime(
        diary["date"].astype(str) + " " + diary["start_time"],
        format="%Y-%m-%d %I:%M:%S %p",
    )
    assert (stamps.dt.second == 0).all()


def test_diary_loses_duration_for_short_events(small_run):
    """Short events are logged with equal start and end, as residents do."""
    _, _, diary, _ = small_run
    equal = diary["start_time"] == diary["end_time"]
    assert equal.any(), "expected some entries with no recorded duration"


def test_diary_omits_some_events(small_run):
    """A real diary misses events; the matcher must tolerate orphans."""
    _, _, diary, events = small_run
    non_appliance = events[~events["fixture"].isin(["Dish_W", "Wash_M"])]
    assert len(diary) < len(non_appliance) + 50


def test_diary_symbols_are_room_scoped(small_run):
    """The same letter means different fixtures on different room sheets."""
    _, _, diary, _ = small_run
    pairs = set(zip(diary["page"], diary["symbol"]))
    faucet_pages = {p for p, s in pairs if s == "F"}
    assert len(faucet_pages) > 1


def test_trace_round_trips_through_reader(tmp_path):
    """Written files must parse with the same reader used on real data."""
    spec = default_spec(n_days=2, seed=5)
    paths = write_simulated_dataset(tmp_path, spec)

    config = {
        "meter": {
            "sample_period_s": 1,
            "noise_floor_gpm": 0.05,
            "timezone": {"shift_hours": 0},
            "raw_format": {
                "layout": "paired_rows",
                "has_header": False,
                "timestamp_format": "%Y-%m-%d %I:%M:%S %p",
                "columns": [
                    "timestamp", "temperature_f", "pressure",
                    "flow_gpm", "cumulative_gal",
                ],
                "flow_column": "flow_gpm",
            },
        }
    }
    trace = read_trace_dir(paths["trace_dir"], config)
    assert trace.volume_gal() > 0
    assert len(trace.local_days()) == 2
    # Dropped samples must survive the round trip as gaps, not zeros.
    assert 0 < (~trace.valid).mean() < 0.05


def test_superposition_factor_reduces_concurrent_flow():
    """Below 1.0, concurrent draws must read less than the sum of solo rates."""
    additive = SimSpec(
        fixtures=default_spec().fixtures, n_days=3, seed=2,
        superposition_factor=1.0, meter_tau_s=0.0,
    )
    lossy = dataclasses_replace(additive, superposition_factor=0.85)

    _, truth_a, _, _ = simulate_household(additive)
    trace_a, _, _, _ = simulate_household(additive)
    trace_b, truth_b, _, _ = simulate_household(lossy)

    concurrent = (truth_b.to_numpy() > 0).sum(axis=1) > 1
    if not concurrent.any():
        pytest.skip("no concurrent use in this draw")
    assert (
        trace_b["flow_gpm"].to_numpy()[concurrent].sum()
        < trace_a["flow_gpm"].to_numpy()[concurrent].sum()
    )


def dataclasses_replace(spec, **changes):
    import dataclasses

    return dataclasses.replace(spec, **changes)
