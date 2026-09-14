"""Windowing and partitioning.

Turns a continuous trace and its label matrix into fixed-length training
examples, and divides those examples into training, validation and test sets.

The partitioning is the part that needs care. At the default 60-second window
and 15-second stride, consecutive windows share three quarters of their
samples. Assigning windows to partitions independently therefore places
near-duplicates on both sides of the boundary, and every reported metric
becomes a measure of memorisation rather than generalisation. Partitions are
instead drawn over whole calendar days, and windows spanning a boundary are
discarded.

Days are also the natural unit for a household: behaviour is organised
around them, and holding out complete days asks the model the question that
matters -- can it disaggregate a day it has never seen.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "WindowSet",
    "DayPartition",
    "make_windows",
    "partition_days",
    "build_partitions",
    "WindowDataset",
    "make_loaders",
]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WindowSet:
    """A collection of aligned windows.

    Attributes
    ----------
    x:
        ``(n, T)`` aggregate flow in gpm.
    y:
        ``(n, T, F)`` labelled per-fixture flow in gpm.
    valid:
        ``(n, T)`` sample-validity mask, false where the logger dropped a
        sample. Carried through so the loss can ignore those positions.
    starts:
        Index of the first sample of each window within the source trace.
    timestamps:
        Start time of each window, retained for diagnostics and for mapping
        predictions back onto the timeline.
    fixtures:
        Fixture codes, in the column order of ``y``.
    """

    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    starts: np.ndarray
    timestamps: pd.DatetimeIndex
    fixtures: list[str]

    def __len__(self) -> int:
        return len(self.x)

    @property
    def window_length(self) -> int:
        return self.x.shape[1]

    def event_counts(self, noise_floor: float = 0.05) -> pd.Series:
        """Windows in which each fixture appears.

        Not a count of events -- a long shower spans many windows -- but the
        right measure of how much a fixture contributes to the objective.
        """
        active = (self.y > noise_floor).any(axis=1)
        return pd.Series(active.sum(axis=0), index=self.fixtures)

    def summary(self) -> dict:
        return {
            "windows": len(self),
            "length": self.window_length,
            "fixtures": len(self.fixtures),
            "span": (
                f"{self.timestamps.min()} to {self.timestamps.max()}"
                if len(self) else None
            ),
        }

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(
            x=self.x[mask],
            y=self.y[mask],
            valid=self.valid[mask],
            starts=self.starts[mask],
            timestamps=self.timestamps[mask],
            fixtures=self.fixtures,
        )


@dataclasses.dataclass
class DayPartition:
    """Assignment of calendar days to partitions."""

    train: list[pd.Timestamp]
    val: list[pd.Timestamp]
    test: list[pd.Timestamp]

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"day": str(day.date()), "partition": name,
             "day_type": "weekend" if day.dayofweek >= 5 else "weekday"}
            for name, days in (
                ("train", self.train), ("val", self.val), ("test", self.test)
            )
            for day in days
        ]
        return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)

    def summary(self) -> dict:
        def weekends(days):
            return sum(1 for d in days if d.dayofweek >= 5)

        return {
            "train_days": len(self.train),
            "val_days": len(self.val),
            "test_days": len(self.test),
            "train_weekend": weekends(self.train),
            "val_weekend": weekends(self.val),
            "test_weekend": weekends(self.test),
        }


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def make_windows(
    trace,
    label_set,
    config: dict,
    *,
    days: Sequence[pd.Timestamp] | None = None,
    require_active: bool | None = None,
) -> WindowSet:
    """Extract windows from a trace and its labels.

    Parameters
    ----------
    days:
        Restrict output to windows falling entirely within these calendar
        days. Windows straddling a day not in the set are dropped, which is
        what keeps partitions disjoint.
    require_active:
        Keep only windows containing at least one active fixture. Defaults to
        the configured value. Retaining every idle window would let the model
        minimise its loss by predicting nothing, since the household is idle
        most of the time.

    Notes
    -----
    Windows are gathered by fancy indexing rather than a Python loop. On a
    multi-year trace the difference is minutes against hours.
    """
    spec = config["windows"]
    length = int(spec["length_s"] / config["meter"]["sample_period_s"])
    stride = int(spec["stride_s"] / config["meter"]["sample_period_s"])
    if require_active is None:
        require_active = spec.get("require_active_fixture", True)
    floor = config["meter"]["noise_floor_gpm"]

    index = trace.flow.index
    flow = trace.flow.to_numpy(dtype=np.float32)
    valid = trace.valid.to_numpy(dtype=bool)
    labels = label_set.matrix.reindex(index).fillna(0.0)
    y_all = labels.to_numpy(dtype=np.float32)

    n = len(flow)
    if n < length:
        log.warning("trace shorter than one window; no windows produced")
        return WindowSet(
            np.empty((0, length), np.float32),
            np.empty((0, length, y_all.shape[1]), np.float32),
            np.empty((0, length), bool),
            np.empty(0, int),
            pd.DatetimeIndex([]),
            list(labels.columns),
        )

    starts = np.arange(0, n - length + 1, stride)

    # Restrict to requested days. Day codes are monotone along the trace, so
    # a window lies within one permitted day exactly when its first and last
    # samples share a code that is permitted.
    if days is not None:
        wanted = {pd.Timestamp(d).normalize() for d in days}
        day_index = index.normalize()
        unique = np.array(sorted(set(day_index)))
        codes = np.searchsorted(unique, day_index.to_numpy())
        permitted = np.array([pd.Timestamp(d) in wanted for d in unique])

        first, last = codes[starts], codes[starts + length - 1]
        same_day = first == last
        keep = np.zeros(len(starts), dtype=bool)
        keep[same_day] = permitted[first[same_day]]

        guard = spec.get("boundary_guard_s")
        dropped_boundary = int((~same_day).sum())
        if dropped_boundary:
            log.debug(
                "%d windows dropped for spanning a day boundary",
                dropped_boundary,
            )
        starts = starts[keep]

    if require_active and starts.size:
        # Prefix sum makes the activity test O(1) per window.
        active_any = (y_all > floor).any(axis=1).astype(np.int32)
        cumulative = np.concatenate([[0], np.cumsum(active_any)])
        has_activity = (cumulative[starts + length] - cumulative[starts]) > 0
        starts = starts[has_activity]

    if starts.size == 0:
        log.warning("no windows survived filtering")
        return WindowSet(
            np.empty((0, length), np.float32),
            np.empty((0, length, y_all.shape[1]), np.float32),
            np.empty((0, length), bool),
            np.empty(0, int),
            pd.DatetimeIndex([]),
            list(labels.columns),
        )

    offsets = np.arange(length)
    rows = starts[:, None] + offsets[None, :]

    window_set = WindowSet(
        x=flow[rows],
        y=y_all[rows],
        valid=valid[rows],
        starts=starts,
        timestamps=index[starts],
        fixtures=list(labels.columns),
    )
    log.info("windows: %s", window_set.summary())
    return window_set


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def partition_days(
    labelled_days: Sequence[pd.Timestamp], config: dict
) -> DayPartition:
    """Divide labelled days into training, validation and test sets.

    Sizes are fractions of the number of days actually available rather than
    fixed counts, since diary coverage within a stated collection window is
    usually intermittent and the usable total is a property of the data.

    When ``stratify_by: day_type`` is set, weekdays and weekend days are
    sampled separately. Without this, a small held-out set can easily contain
    no weekend at all, which matters when weekday/weekend contrast is among
    the results being reported.
    """
    spec = config["split"]
    days = sorted(pd.Timestamp(d).normalize() for d in labelled_days)
    if not days:
        raise ValueError("no labelled days to partition")

    n = len(days)
    n_test = max(int(round(n * spec.get("test_frac", 0.18))),
                 spec.get("min_test_days", 1))
    n_val = max(int(round(n * spec.get("val_frac", 0.14))),
                spec.get("min_val_days", 1))

    if n_test + n_val >= n:
        # Degrade gracefully on very short campaigns rather than failing.
        n_test = max(1, n // 4)
        n_val = max(1, n // 5)
        if n_test + n_val >= n:
            n_val = max(0, n - n_test - 1)
        log.warning(
            "only %d labelled days; reduced to %d test and %d validation days",
            n, n_test, n_val,
        )

    rng = np.random.default_rng(spec.get("seed", 0))

    if spec.get("stratify_by") == "day_type":
        weekday = [d for d in days if d.dayofweek < 5]
        weekend = [d for d in days if d.dayofweek >= 5]
        test = _stratified_take(weekday, weekend, n_test, rng)
        remaining_weekday = [d for d in weekday if d not in test]
        remaining_weekend = [d for d in weekend if d not in test]
        val = _stratified_take(remaining_weekday, remaining_weekend, n_val, rng)
    else:
        shuffled = list(days)
        rng.shuffle(shuffled)
        test, val = shuffled[:n_test], shuffled[n_test: n_test + n_val]

    held_out = set(test) | set(val)
    train = [d for d in days if d not in held_out]

    partition = DayPartition(
        train=sorted(train), val=sorted(val), test=sorted(test)
    )
    log.info("day partition: %s", partition.summary())
    if not partition.test:
        log.warning("test partition is empty")
    return partition


def _stratified_take(
    weekday: Sequence[pd.Timestamp],
    weekend: Sequence[pd.Timestamp],
    n: int,
    rng: np.random.Generator,
) -> list[pd.Timestamp]:
    """Take ``n`` days, keeping the weekday/weekend ratio where possible."""
    if n <= 0:
        return []
    total = len(weekday) + len(weekend)
    if total == 0:
        return []
    n_weekend = int(round(n * len(weekend) / total))
    n_weekend = min(n_weekend, len(weekend))
    n_weekday = min(n - n_weekend, len(weekday))
    # If one stratum is exhausted, make up the shortfall from the other.
    shortfall = n - n_weekday - n_weekend
    if shortfall > 0:
        n_weekend = min(n_weekend + shortfall, len(weekend))

    picked = []
    if n_weekday:
        picked += list(rng.choice(np.array(weekday), n_weekday, replace=False))
    if n_weekend:
        picked += list(rng.choice(np.array(weekend), n_weekend, replace=False))
    return sorted(pd.Timestamp(d) for d in picked)


def build_partitions(
    trace, label_set, config: dict
) -> tuple[dict[str, WindowSet], DayPartition]:
    """Partition by day, then window each partition independently.

    Windowing after partitioning is what guarantees no window contributes to
    more than one partition.
    """
    partition = partition_days(label_set.labelled_days, config)
    sets: dict[str, WindowSet] = {}
    for name, days in (
        ("train", partition.train),
        ("val", partition.val),
        ("test", partition.test),
    ):
        if not days:
            continue
        sets[name] = make_windows(trace, label_set, config, days=days)
        log.info("%s: %d windows over %d days", name, len(sets[name]), len(days))

    _warn_on_thin_support(sets, config)
    return sets, partition


def _warn_on_thin_support(
    sets: dict[str, WindowSet], config: dict
) -> None:
    """Report fixtures with too little held-out data to evaluate.

    Reported rather than corrected: per-fixture scores computed from one or
    two events are not informative, and resampling to manufacture test
    examples would misrepresent what was actually validated.
    """
    threshold = config["split"].get("min_test_events_warn", 3)
    test = sets.get("test")
    if test is None or not len(test):
        return
    counts = test.event_counts(config["meter"]["noise_floor_gpm"])
    thin = counts[counts < threshold]
    if not thin.empty:
        log.warning(
            "fixtures with fewer than %d test windows: %s -- per-fixture "
            "metrics for these will not be meaningful",
            threshold, dict(thin.astype(int)),
        )


# ---------------------------------------------------------------------------
# PyTorch plumbing
# ---------------------------------------------------------------------------


class WindowDataset:
    """Adapts a :class:`WindowSet` to the PyTorch ``Dataset`` interface.

    Kept as a plain class so that ``windows.py`` can be imported without
    PyTorch installed; the tensor conversion happens on access.
    """

    def __init__(self, window_set: WindowSet):
        self.window_set = window_set

    def __len__(self) -> int:
        return len(self.window_set)

    def __getitem__(self, i: int) -> dict:
        import torch

        return {
            "x": torch.from_numpy(self.window_set.x[i]).unsqueeze(-1),
            "y": torch.from_numpy(self.window_set.y[i]),
            "valid": torch.from_numpy(self.window_set.valid[i]),
        }


def make_loaders(
    sets: dict[str, WindowSet], config: dict, **kwargs
) -> dict[str, "object"]:
    """Build ``DataLoader`` objects for each partition.

    Only the training loader is shuffled; validation and test are kept in
    chronological order so that diagnostics remain interpretable.
    """
    from torch.utils.data import DataLoader

    batch_size = kwargs.pop("batch_size", None) or config.get(
        "model", {}
    ).get("batch_size", 64)

    loaders = {}
    for name, window_set in sets.items():
        if not len(window_set):
            continue
        loaders[name] = DataLoader(
            WindowDataset(window_set),
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
            **kwargs,
        )
    return loaders
