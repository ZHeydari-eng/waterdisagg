"""Tests for label construction, partitioning and event assembly.

The leakage tests are the ones worth reading. Because adjacent windows share
most of their samples at the default stride, a partitioning bug does not
produce an error -- it produces flattering numbers. These tests assert the
structural property directly.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from waterdisagg import labels as L
from waterdisagg.events import (
    assemble_events,
    match_events,
    merge_close_events,
)
from waterdisagg.io import read_trace_dir
from waterdisagg.rates import fit_rates
from waterdisagg.simulate import default_spec, write_simulated_dataset
from waterdisagg.windows import (
    build_partitions,
    make_windows,
    partition_days,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "household.yaml"


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def prepared(config):
    """A full simulated dataset carried through matching and labelling."""
    tmp = Path(tempfile.mkdtemp())
    paths = write_simulated_dataset(tmp, default_spec(n_days=12, seed=41))
    trace = read_trace_dir(paths["trace_dir"], config)
    diary = L.read_diary(paths["diary"], config)
    matched, report = L.match_diary(trace, diary, config)
    rate_fit = fit_rates(trace, matched, config)
    label_set = L.build_label_matrix(
        trace, matched, config, rates=rate_fit.rates, report=report
    )
    return {
        "tmp": tmp,
        "paths": paths,
        "trace": trace,
        "diary": diary,
        "matched": matched,
        "rates": rate_fit,
        "labels": label_set,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Diary reading and matching
# ---------------------------------------------------------------------------


def test_diary_codes_resolve_to_fixtures(prepared):
    diary = prepared["diary"]
    assert diary["_fixture"].notna().all()
    assert not prepared["report"].unknown_codes


def test_diary_records_unknown_durations(prepared):
    """Short uses are logged with equal start and end and must be flagged."""
    assert (~prepared["diary"]["_duration_known"]).any()


def test_match_rate_is_high(prepared):
    report = prepared["report"]
    assert report.match_rate > 0.85, f"match rate {report.match_rate:.2f}"
    assert report.n_matched > 0


def test_matched_events_take_boundaries_from_trace(prepared):
    """Diary times are minute-resolution; event boundaries must not be.

    If boundaries came from the diary, every start would fall on a whole
    minute. Taking them from the flow trace gives second-level precision.
    """
    matched = prepared["matched"]
    seconds = matched["start"].dt.second
    assert (seconds != 0).mean() > 0.5


def test_match_identity_is_correct_against_truth(prepared):
    """Fixture identity must agree with what the simulator actually ran."""
    truth = pd.read_csv(prepared["paths"]["diary_with_truth"])
    truth["_true_start"] = pd.to_datetime(truth["_true_start"])
    correct = total = 0
    for _, row in prepared["matched"].iterrows():
        near = truth[
            (truth["_true_start"] - row["start"]).abs()
            < pd.Timedelta(seconds=120)
        ]
        if len(near):
            total += 1
            correct += int((near["_true_fixture"] == row["fixture"]).any())
    assert total > 20
    assert correct / total > 0.9, f"identity accuracy {correct / total:.2f}"


def test_unmatched_and_orphans_are_reported(prepared):
    report = prepared["report"]
    # A simulated diary omits some events, so orphan flow events must exist
    # and must be surfaced rather than silently dropped.
    assert report.n_orphan_events > 0
    assert report.orphan_events is not None


# ---------------------------------------------------------------------------
# Label matrix
# ---------------------------------------------------------------------------


def test_label_matrix_aligned_and_non_negative(prepared, config):
    matrix = prepared["labels"].matrix
    assert matrix.index.equals(prepared["trace"].flow.index)
    assert list(matrix.columns) == [f["code"] for f in config["fixtures"]]
    assert (matrix.to_numpy() >= 0).all()


def test_excluded_fixture_absent_from_labels(prepared):
    assert "Ad_B" not in prepared["labels"].fixtures


def test_solo_intervals_receive_the_observed_aggregate(prepared, config):
    """Where one fixture runs, its label must be the measured flow.

    Labelling a solo event with a flat fitted rate would discard genuine
    within-event variation and teach the model a waveform that does not occur.
    """
    matrix = prepared["labels"].matrix
    floor = config["meter"]["noise_floor_gpm"]
    active = (matrix > floor).sum(axis=1)
    solo = active == 1
    assert solo.any()
    aggregate = prepared["trace"].deadbanded()
    difference = (matrix[solo].sum(axis=1) - aggregate[solo]).abs()
    assert float(difference.max()) < 1e-3


def test_concurrent_intervals_conserve_the_total(prepared, config):
    """Overlapping use must divide the aggregate, not duplicate it."""
    matrix = prepared["labels"].matrix
    floor = config["meter"]["noise_floor_gpm"]
    concurrent = (matrix > floor).sum(axis=1) > 1
    if not concurrent.any():
        pytest.skip("no concurrent use in this draw")
    aggregate = prepared["trace"].deadbanded()
    difference = (
        matrix[concurrent].sum(axis=1) - aggregate[concurrent]
    ).abs()
    assert float(difference.max()) < 1e-3


def test_labelled_days_reflect_coverage_not_trace_extent(prepared):
    """Partitioning must be drawn from days that carry labels."""
    label_set = prepared["labels"]
    assert 0 < len(label_set.labelled_days) <= len(
        prepared["trace"].local_days()
    )


def test_load_label_matrix_round_trip(prepared, config, tmp_path):
    """A written matrix must read back identically, in the wide format."""
    label_set = prepared["labels"]
    trace = prepared["trace"]

    wide = label_set.matrix.copy()
    wide.insert(0, "Flow", trace.flow.reindex(wide.index).to_numpy())
    wide.insert(0, "Time", wide.index.strftime("%H:%M:%S"))
    wide.insert(0, "Date", wide.index.strftime("%Y-%m-%d"))
    wide["Ad_B"] = 0.0                       # must be ignored on load
    path = tmp_path / "labels.csv"
    wide.to_csv(path, index=True)            # leading index column

    loaded = L.load_label_matrix(path, trace, config)
    assert "Ad_B" not in loaded.fixtures
    assert loaded.fixtures == label_set.fixtures
    np.testing.assert_allclose(
        loaded.matrix.to_numpy(), label_set.matrix.to_numpy(), atol=1e-5
    )


def test_load_label_matrix_rejects_unknown_columns(prepared, config, tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"Date": ["2024-06-06"], "Time": ["00:00:00"],
                  "NotAFixture": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no configured fixture columns"):
        L.load_label_matrix(path, prepared["trace"], config)


def test_concurrency_is_reported(prepared):
    concurrency = prepared["labels"].concurrency()
    assert not concurrency.empty
    assert concurrency.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Partitioning -- the leakage tests
# ---------------------------------------------------------------------------


def test_partitions_share_no_windows(prepared, config):
    """The central guarantee: no window appears in two partitions.

    At a 15 s stride over 60 s windows, neighbours share 75% of their
    samples, so overlap between partitions would inflate every metric while
    raising no error.
    """
    sets, _ = build_partitions(
        prepared["trace"], prepared["labels"], config
    )
    for a, b in itertools.combinations(sets, 2):
        assert not (
            set(sets[a].starts.tolist()) & set(sets[b].starts.tolist())
        ), f"{a} and {b} share windows"


def test_partitions_share_no_days(prepared, config):
    _, partition = build_partitions(
        prepared["trace"], prepared["labels"], config
    )
    assert not set(partition.train) & set(partition.val)
    assert not set(partition.train) & set(partition.test)
    assert not set(partition.val) & set(partition.test)


def test_no_window_spans_a_day_boundary(prepared, config):
    """A window crossing midnight would straddle two partitions."""
    sets, _ = build_partitions(
        prepared["trace"], prepared["labels"], config
    )
    length = config["windows"]["length_s"]
    for window_set in sets.values():
        if not len(window_set):
            continue
        starts = window_set.timestamps
        ends = starts + pd.Timedelta(seconds=length - 1)
        assert (starts.normalize() == ends.normalize()).all()


def test_partition_sizes_scale_with_available_days(config):
    """Sizes are fractions of what exists, not fixed counts."""
    for n_days in (10, 30, 60):
        days = pd.date_range("2024-06-01", periods=n_days, freq="D")
        partition = partition_days(list(days), config)
        total = (
            len(partition.train) + len(partition.val) + len(partition.test)
        )
        assert total == n_days
        assert len(partition.test) >= config["split"]["min_test_days"]


def test_partition_handles_very_few_days(config):
    """A short campaign must degrade gracefully rather than fail."""
    days = list(pd.date_range("2024-06-01", periods=4, freq="D"))
    partition = partition_days(days, config)
    assert len(partition.train) >= 1
    assert len(partition.train) + len(partition.val) + len(partition.test) == 4


def test_partition_rejects_empty_input(config):
    with pytest.raises(ValueError):
        partition_days([], config)


def test_stratification_puts_weekends_on_both_sides(config):
    """Weekday/weekend contrast is a reported result; both sides need both."""
    days = list(pd.date_range("2024-06-01", periods=42, freq="D"))
    partition = partition_days(days, config)
    assert any(d.dayofweek >= 5 for d in partition.train)
    assert any(d.dayofweek >= 5 for d in partition.test)


def test_windows_only_contain_requested_days(prepared, config):
    days = prepared["labels"].labelled_days[:3]
    window_set = make_windows(
        prepared["trace"], prepared["labels"], config, days=days
    )
    wanted = {pd.Timestamp(d).normalize() for d in days}
    assert set(window_set.timestamps.normalize()) <= wanted


def test_windows_require_activity(prepared, config):
    """Idle windows are excluded, or the model can win by predicting zero."""
    window_set = make_windows(
        prepared["trace"], prepared["labels"], config, require_active=True
    )
    floor = config["meter"]["noise_floor_gpm"]
    assert (window_set.y > floor).any(axis=(1, 2)).all()


def test_window_shapes_are_consistent(prepared, config):
    window_set = make_windows(prepared["trace"], prepared["labels"], config)
    n = len(window_set)
    length = config["windows"]["length_s"]
    assert window_set.x.shape == (n, length)
    assert window_set.y.shape == (n, length, len(window_set.fixtures))
    assert window_set.valid.shape == (n, length)


# ---------------------------------------------------------------------------
# Event assembly
# ---------------------------------------------------------------------------


def test_assembled_events_recover_labelled_events(prepared, config):
    """Assembling from the labels themselves should reproduce them.

    An upper bound on the assembly rules, independent of model quality: if
    perfect per-second input does not yield the right events, the thresholds
    or gap tolerances are wrong.
    """
    events = assemble_events(
        prepared["labels"].matrix,
        config,
        aggregate=prepared["trace"].deadbanded(),
        valid=prepared["trace"].valid,
    )
    assert not events.empty
    _, counts = match_events(events, prepared["matched"], min_iou=0.5)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    assert f1 > 0.85, f"self-consistency F1 only {f1:.3f}"


def test_event_table_carries_analysis_columns(prepared, config):
    events = assemble_events(prepared["labels"].matrix, config)
    for column in (
        "fixture", "start", "end", "duration_min", "volume_gal",
        "time_of_day_h", "day_type", "date",
    ):
        assert column in events.columns
    assert events["time_of_day_h"].between(0, 24).all()
    assert set(events["day_type"]) <= {"weekday", "weekend"}


def test_match_events_ignoring_fixture_separates_detection(prepared, config):
    """Detection without attribution must be at least as good as with it."""
    events = assemble_events(prepared["labels"].matrix, config)
    _, strict = match_events(events, prepared["matched"], per_fixture=True)
    _, loose = match_events(events, prepared["matched"], per_fixture=False)
    assert loose["tp"] >= strict["tp"]


def test_match_events_handles_empty_inputs():
    empty = pd.DataFrame()
    pairs, counts = match_events(empty, empty)
    assert pairs.empty
    assert counts["tp"] == 0


def test_merge_close_events_combines_and_conserves_volume():
    events = pd.DataFrame(
        {
            "fixture": ["Ad_S", "Ad_S", "Ad_S"],
            "start": pd.to_datetime(
                ["2024-06-06 07:00:00", "2024-06-06 07:10:00",
                 "2024-06-06 09:00:00"]
            ),
            "end": pd.to_datetime(
                ["2024-06-06 07:09:00", "2024-06-06 07:19:00",
                 "2024-06-06 09:10:00"]
            ),
            "duration_s": [540.0, 540.0, 600.0],
            "duration_min": [9.0, 9.0, 10.0],
            "volume_gal": [20.0, 20.0, 22.0],
            "peak_gpm": [2.3, 2.4, 2.3],
        }
    )
    merged = merge_close_events(events, min_gap_s=300)
    assert len(merged) == 2
    assert merged["volume_gal"].sum() == pytest.approx(62.0)
    assert merged.iloc[0]["duration_min"] == pytest.approx(19.0)


def test_merge_close_events_is_identity_at_zero():
    events = pd.DataFrame(
        {
            "fixture": ["Ad_S"],
            "start": pd.to_datetime(["2024-06-06 07:00:00"]),
            "end": pd.to_datetime(["2024-06-06 07:09:00"]),
            "duration_s": [540.0],
            "duration_min": [9.0],
            "volume_gal": [20.0],
            "peak_gpm": [2.3],
        }
    )
    pd.testing.assert_frame_equal(merge_close_events(events, 0), events)
