"""Event segmentation and per-fixture flow-rate estimation.

Two jobs live here.

**Segmentation** turns a flow series into discrete candidate events. A single
threshold is not sufficient: the signal is noisy near the switching point, so
a lone threshold fragments one physical event into several. Segmentation
therefore uses hysteresis (separate rising and falling thresholds), bridges
short gaps, and drops activations below a minimum duration. The same routine
runs on the aggregate trace (to find candidates for diary matching) and on
per-fixture model output (to build the final event table).

**Rate fitting** estimates each fixture's steady-state flow rate, ``r_f``,
from labelled events that ran in isolation. These rates matter in three
places: they set the per-fixture thresholds as fractions of ``r_f`` so the
configuration transfers between homes; they allocate the aggregate across
fixtures during periods of concurrent use; and they let the synthesiser scale
templates realistically.

Only the *plateau* of an event is used for fitting. Meters of this kind apply
substantial low-pass filtering, so the leading and trailing thirds of a short
event are dominated by instrument response rather than the true rate.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "Event",
    "segment",
    "segment_aggregate",
    "plateau_of",
    "fit_rates",
    "RateFit",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Event:
    """A contiguous period of flow attributed to one source.

    ``start`` and ``end`` are inclusive and exclusive respectively, matching
    Python slice conventions, so ``duration_s`` is ``end - start`` in seconds.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    fixture: str | None = None
    mean_gpm: float = float("nan")
    peak_gpm: float = float("nan")
    plateau_gpm: float = float("nan")
    volume_gal: float = float("nan")

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    def overlaps(self, other: "Event") -> bool:
        return self.start < other.end and other.start < self.end

    def to_row(self) -> dict:
        return {
            "fixture": self.fixture,
            "start": self.start,
            "end": self.end,
            "duration_s": self.duration_s,
            "mean_gpm": self.mean_gpm,
            "peak_gpm": self.peak_gpm,
            "plateau_gpm": self.plateau_gpm,
            "volume_gal": self.volume_gal,
        }


def events_to_frame(events: Iterable[Event]) -> pd.DataFrame:
    """Collect events into a tidy frame, sorted by start time."""
    rows = [e.to_row() for e in events]
    if not rows:
        return pd.DataFrame(
            columns=[
                "fixture", "start", "end", "duration_s",
                "mean_gpm", "peak_gpm", "plateau_gpm", "volume_gal",
            ]
        )
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _hysteresis_mask(
    values: np.ndarray, on_threshold: float, off_threshold: float
) -> np.ndarray:
    """Schmitt-trigger style thresholding.

    The state turns on when the signal rises above ``on_threshold`` and stays
    on until it falls below ``off_threshold``. With a single threshold, noise
    around the switching point chops one event into many; separating the two
    edges removes that failure mode.
    """
    if off_threshold > on_threshold:
        raise ValueError("off_threshold must not exceed on_threshold")

    above_on = values > on_threshold
    below_off = values < off_threshold

    # Vectorised equivalent of a sequential trigger: within each run of
    # samples that are not below the off-threshold, the state is on from the
    # first sample that exceeded the on-threshold onward.
    state = np.zeros(values.shape, dtype=bool)
    # Boundaries of runs where the signal never dips below off_threshold.
    run_id = np.cumsum(below_off)
    # For each run, has the on-threshold been crossed yet?
    order = np.arange(values.size)
    crossed = np.zeros(values.size, dtype=bool)
    if above_on.any():
        frame = pd.DataFrame({"run": run_id, "on": above_on, "i": order})
        first_on = frame[frame["on"]].groupby("run")["i"].min()
        threshold_index = frame["run"].map(first_on)
        crossed = (~below_off) & threshold_index.notna().to_numpy() & (
            order >= threshold_index.fillna(np.inf).to_numpy()
        )
    state = crossed
    return state


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` index pairs for each True run."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _bridge(runs: Sequence[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by fewer than ``max_gap`` samples."""
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def plateau_of(
    values: np.ndarray, lo_frac: float = 0.3, hi_frac: float = 0.85
) -> float:
    """Median flow across the steady-state interior of an event.

    Excludes the leading and trailing portions, which on a filtered meter are
    dominated by instrument response. Returns ``nan`` when the interior is too
    short to be meaningful.
    """
    n = values.size
    if n < 6:
        return float("nan")
    lo, hi = int(n * lo_frac), max(int(n * hi_frac), int(n * lo_frac) + 1)
    core = values[lo:hi]
    if core.size < 3:
        return float("nan")
    return float(np.median(core))


def segment(
    flow: pd.Series,
    *,
    on_threshold: float,
    off_threshold: float,
    min_duration_s: float,
    bridge_gap_s: float,
    sample_period_s: int = 1,
    fixture: str | None = None,
    valid: pd.Series | None = None,
) -> list[Event]:
    """Segment a flow series into events.

    Parameters
    ----------
    flow:
        Flow rate in gpm, on a regular index.
    on_threshold, off_threshold:
        Rising and falling thresholds in gpm. See :func:`_hysteresis_mask`.
    min_duration_s:
        Activations shorter than this are discarded as noise.
    bridge_gap_s:
        Activations separated by less than this are merged into one event.
        Set per fixture: a toilet needs a few seconds, a shower tens of
        seconds, a dishwasher several minutes.
    valid:
        Optional sample-validity mask. Events are not permitted to span a
        stretch of missing data, since the flow there is unknown.
    """
    if flow.empty:
        return []

    values = flow.to_numpy(dtype=float)
    mask = _hysteresis_mask(values, on_threshold, off_threshold)

    if valid is not None:
        # Break events at gaps: a dropout is not evidence of continuity.
        mask = mask & valid.to_numpy(dtype=bool)

    gap_samples = int(round(bridge_gap_s / sample_period_s))
    min_samples = int(round(min_duration_s / sample_period_s))

    runs = _bridge(_runs(mask), gap_samples)

    events: list[Event] = []
    for start_i, end_i in runs:
        if end_i - start_i < max(min_samples, 1):
            continue
        segment_values = values[start_i:end_i]
        positive = np.clip(segment_values, 0.0, None)
        events.append(
            Event(
                start=flow.index[start_i],
                end=flow.index[end_i - 1] + pd.Timedelta(seconds=sample_period_s),
                fixture=fixture,
                mean_gpm=float(positive.mean()),
                peak_gpm=float(positive.max()),
                plateau_gpm=plateau_of(positive),
                volume_gal=float(positive.sum() * sample_period_s / 60.0),
            )
        )
    return events


def segment_aggregate(trace, config: dict) -> list[Event]:
    """Segment the aggregate trace into candidate events.

    Used before any fixture identity is known -- by :func:`fit_rates` to find
    isolated events, and by ``labels.py`` to find candidates to match against
    diary rows. Thresholds come from ``segmentation.aggregate`` in the config,
    expressed in absolute gpm since no ``r_f`` is available yet.
    """
    agg = config["segmentation"]["aggregate"]
    return segment(
        trace.deadbanded(),
        on_threshold=agg["on_threshold_gpm"],
        off_threshold=agg["off_threshold_gpm"],
        min_duration_s=agg["min_duration_s"],
        bridge_gap_s=agg["bridge_gap_s"],
        sample_period_s=trace.sample_period_s,
        valid=trace.valid,
    )


# ---------------------------------------------------------------------------
# Rate fitting
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RateFit:
    """Fitted steady-state rates, with the evidence behind each estimate."""

    rates: dict[str, float]
    n_events: dict[str, int]
    spread: dict[str, float]          # IQR of plateau estimates, gpm
    fallback: dict[str, str]          # fixtures that fell back, and why

    def report(self) -> pd.DataFrame:
        codes = sorted(set(self.rates) | set(self.fallback))
        return pd.DataFrame(
            [
                {
                    "fixture": c,
                    "rate_gpm": round(self.rates.get(c, float("nan")), 3),
                    "n_isolated_events": self.n_events.get(c, 0),
                    "plateau_iqr_gpm": round(self.spread.get(c, float("nan")), 3),
                    "note": self.fallback.get(c, ""),
                }
                for c in codes
            ]
        )

    def separability(self, min_ratio: float = 0.08) -> pd.DataFrame:
        """Flag fixture pairs whose rates are too close to distinguish.

        Fixture-level disaggregation from a single aggregate signal depends on
        fixtures having measurably different rates. Pairs separated by less
        than ``min_ratio`` in relative terms are unlikely to be reliably
        separated by flow magnitude alone, and any reported per-fixture
        accuracy for them should be read with that in mind.
        """
        codes = [c for c, r in self.rates.items() if np.isfinite(r)]
        rows = []
        for i, a in enumerate(codes):
            for b in codes[i + 1:]:
                ra, rb = self.rates[a], self.rates[b]
                rel = abs(ra - rb) / max(ra, rb)
                rows.append(
                    {
                        "fixture_a": a,
                        "fixture_b": b,
                        "rate_a": round(ra, 3),
                        "rate_b": round(rb, 3),
                        "relative_gap": round(rel, 4),
                        "at_risk": bool(rel < min_ratio),
                    }
                )
        frame = pd.DataFrame(rows).sort_values("relative_gap").reset_index(drop=True)
        return frame


def fit_rates(
    trace,
    labels: pd.DataFrame,
    config: dict,
    *,
    min_events: int = 3,
) -> RateFit:
    """Estimate ``r_f`` for each fixture from isolated labelled events.

    Parameters
    ----------
    trace:
        The conditioned aggregate trace.
    labels:
        Matched event table with at least ``fixture``, ``start`` and ``end``
        columns -- the output of ``labels.match_diary``. Events that overlap
        any other labelled event are excluded, because the aggregate during
        concurrent use reflects more than one fixture.
    min_events:
        Below this many isolated events, the configured ``rate_gpm`` is used
        instead and the fixture is recorded in ``fallback``. Sparse fixtures
        (a rarely-used shower, a laundry faucet) routinely hit this path.

    Notes
    -----
    The median of per-event plateaus is used rather than the mean: a single
    mis-matched diary row would otherwise drag the estimate. The reported IQR
    is the honest measure of how well-determined each rate is.
    """
    configured = {f["code"]: f.get("rate_gpm") for f in config["fixtures"]}
    dead = trace.deadbanded()

    rates: dict[str, float] = {}
    n_events: dict[str, int] = {}
    spread: dict[str, float] = {}
    fallback: dict[str, str] = {}

    if labels.empty:
        for code, rate in configured.items():
            if rate is not None:
                rates[code] = float(rate)
                fallback[code] = "no labelled events; using configured rate"
        return RateFit(rates, n_events, spread, fallback)

    frame = labels.sort_values("start").reset_index(drop=True)
    starts = frame["start"].to_numpy()
    ends = frame["end"].to_numpy()

    # An event is isolated when it overlaps no other labelled event.
    isolated = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        others = (starts < ends[i]) & (ends > starts[i])
        others[i] = False
        if others.any():
            isolated[i] = False

    log.info(
        "rate fitting: %d of %d labelled events are isolated",
        int(isolated.sum()), len(frame),
    )

    for code in configured:
        sel = frame[isolated & (frame["fixture"] == code)]
        plateaus = []
        for _, row in sel.iterrows():
            window = dead.loc[row["start"]: row["end"]]
            value = plateau_of(window.to_numpy(dtype=float))
            if np.isfinite(value) and value > 0:
                plateaus.append(value)

        n_events[code] = len(plateaus)
        if len(plateaus) >= min_events:
            arr = np.asarray(plateaus)
            rates[code] = float(np.median(arr))
            spread[code] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        elif configured[code] is not None:
            rates[code] = float(configured[code])
            spread[code] = float("nan")
            fallback[code] = (
                f"only {len(plateaus)} isolated event(s); using configured rate"
            )
        else:
            fallback[code] = (
                f"only {len(plateaus)} isolated event(s) and no configured rate"
            )

    fit = RateFit(rates, n_events, spread, fallback)
    log.info("fitted rates:\n%s", fit.report().to_string(index=False))
    return fit


def fixture_thresholds(rate_fit: RateFit, config: dict) -> dict[str, dict]:
    """Build per-fixture segmentation parameters from fitted rates.

    Thresholds are stored in the configuration as fractions of ``r_f`` so that
    the same file works for a household whose fixtures flow at different
    rates. Durations and bridge gaps are absolute, since they reflect how
    people and appliances behave rather than plumbing capacity.
    """
    defaults = config["segmentation"]
    out: dict[str, dict] = {}
    for fixture in config["fixtures"]:
        code = fixture["code"]
        rate = rate_fit.rates.get(code)
        if rate is None or not np.isfinite(rate):
            log.warning("no rate for %s; skipping threshold derivation", code)
            continue
        local = fixture.get("segmentation", {}) or {}
        out[code] = {
            "rate_gpm": rate,
            "on_threshold": rate * local.get(
                "on_threshold_frac", defaults["on_threshold_frac"]
            ),
            "off_threshold": rate * local.get(
                "off_threshold_frac", defaults["off_threshold_frac"]
            ),
            "min_duration_s": local.get(
                "min_duration_s", defaults["min_duration_s"]
            ),
            "bridge_gap_s": local.get("bridge_gap_s", defaults["bridge_gap_s"]),
        }
    return out
