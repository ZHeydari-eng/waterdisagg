"""Synthetic household generator.

Produces a complete, self-consistent example dataset: an aggregate flow trace
in the same layout as a real meter, and a hand-style water diary describing
the events inside it. Because the generator knows exactly which fixture
contributed what at every second, it also emits a per-fixture truth matrix.

This exists for three reasons:

1. **A runnable demo.** The pipeline can be exercised end to end without
   access to any household's data, which real traces and diaries usually
   cannot be shared for.
2. **A test fixture.** Every stage downstream -- diary matching, rate
   fitting, windowing, training, event reconstruction -- can be checked
   against known truth. A matcher that recovers 60% of synthetic events will
   not do better on real data.
3. **Sensitivity analysis.** Fixture rates, overlap frequency, meter
   smoothing and diary sloppiness are all parameters, so it is possible to
   ask how far apart two fixtures must flow before they become separable at
   all.

Realism is deliberately limited to properties that affect the pipeline:
filtered instrument response, a non-zero idle baseline, dropped samples,
minute-resolution diary entries with unrecorded durations for short events,
and concurrent use. It does not attempt to model plumbing hydraulics.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["SimSpec", "simulate_household", "write_simulated_dataset"]


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FixtureSpec:
    """Behaviour of one simulated fixture.

    Parameters
    ----------
    rate_gpm:
        Steady-state flow rate.
    duration_range_s:
        Range of event durations, sampled log-uniformly so short events
        dominate, as they do in practice.
    per_day:
        Mean number of activations per day (Poisson).
    hours:
        Preferred hours of use, as ``(centre, spread)`` pairs in hours. Times
        are drawn from a mixture of these, giving the bimodal
        morning/evening structure real fixtures show.
    cycles:
        Number of separate draws per activation. Appliances fill several
        times per cycle, separated by ``cycle_gap_s``.
    """

    code: str
    rate_gpm: float
    duration_range_s: tuple[float, float]
    per_day: float
    hours: list[tuple[float, float]]
    cycles: int = 1
    cycle_gap_s: float = 0.0
    rate_jitter: float = 0.03


@dataclasses.dataclass
class SimSpec:
    """Full simulation specification."""

    fixtures: list[FixtureSpec]
    n_days: int = 28
    start_date: str = "2024-06-06"
    sample_period_s: int = 1

    # Instrument response. Real meters of this class low-pass the signal
    # heavily: a valve that opens in well under a second produces a reading
    # that takes ten seconds or more to reach its plateau. Modelled as
    # cascaded first-order lags; the defaults below were fitted to recorded
    # flush waveforms from the study meter (see _meter_response).
    meter_tau_s: float = 1.5
    meter_stages: int = 4

    # Idle baseline: zero-mean noise, including negative excursions.
    noise_sd_gpm: float = 0.012
    # Occasional larger excursions, as seen in real idle traces.
    noise_spike_rate: float = 2e-4
    noise_spike_sd_gpm: float = 0.12

    # Fraction of samples the logger drops.
    dropout_rate: float = 0.0025
    # Length of a typical dropout, in samples.
    dropout_mean_len: float = 4.0

    # Superposition: observed aggregate over the sum of solo rates during
    # concurrent use. 1.0 is exact additivity; below 1.0 models pressure
    # loss when several fixtures draw at once.
    superposition_factor: float = 1.0

    # Diary realism.
    diary_resolution_s: int = 60      # entries recorded to the minute
    diary_jitter_s: float = 45.0      # human error in noting the time
    diary_miss_rate: float = 0.04     # events the residents forgot to log
    diary_short_event_s: float = 90.0 # below this, no usable duration recorded

    seed: int = 7


def default_spec(n_days: int = 28, seed: int = 7) -> SimSpec:
    """A four-person household roughly matching the study home.

    Rates are spread deliberately: two pairs are placed close together
    (within about 5%) so that the separability limits of the approach show up
    in the demo rather than being hidden by a conveniently easy problem.
    """
    morning = [(7.0, 1.2)]
    evening = [(21.0, 1.5)]
    both = [(7.5, 1.3), (21.0, 1.8)]
    daytime = [(9.0, 3.0), (18.0, 3.0)]

    fixtures = [
        FixtureSpec("Ad_S", 2.26, (300, 1200), 1.1, morning + evening),
        FixtureSpec("Kid_S", 2.44, (240, 1100), 1.0, both),
        FixtureSpec("Dwn_S", 2.58, (300, 900), 0.05, daytime),
        FixtureSpec("Ad_T", 2.84, (30, 70), 4.5, daytime + morning),
        FixtureSpec("Kid_T", 2.99, (30, 70), 3.0, daytime),
        FixtureSpec("Dwn_T", 1.24, (45, 90), 3.5, daytime),
        FixtureSpec("Ad_F", 1.41, (5, 60), 4.0, both),
        FixtureSpec("Kid_F", 0.90, (5, 50), 3.5, both),
        FixtureSpec("Dwn_F", 0.54, (4, 40), 3.5, daytime),
        FixtureSpec("Kitch_F", 1.20, (4, 180), 20.0, daytime),
        FixtureSpec("Kitch_r", 0.40, (3, 20), 5.0, daytime),
        FixtureSpec("Bsmnt_F", 1.55, (10, 120), 0.2, daytime),
        FixtureSpec("Dish_W", 1.24, (60, 120), 0.8, evening, cycles=3, cycle_gap_s=600),
        FixtureSpec("Wash_M", 2.99, (90, 150), 1.0, daytime, cycles=4, cycle_gap_s=420),
    ]
    return SimSpec(fixtures=fixtures, n_days=n_days, seed=seed)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------


def _meter_response(signal: np.ndarray, tau_s: float, stages: int, dt: float) -> np.ndarray:
    """Apply cascaded first-order lags to model instrument smoothing.

    Electromagnetic meters of the class used here filter heavily: a valve
    that opens in well under a second produces a reading that takes ten
    seconds or more to reach its plateau. The consequence for
    disaggregation is significant -- onset and offset transients carry much
    of what distinguishes one fixture from another, and the filter smears
    them.

    A single lag rises too sharply to match observation. Cascading four lags
    with ``tau_s = 1.5`` reproduces the S-shaped ramp of the study meter to
    within about 0.15 gpm on a toilet flush, including the characteristically
    slow first two or three samples. These defaults were fitted against
    recorded flush waveforms; a different instrument will want different
    values, obtained by the same comparison.
    """
    if tau_s <= 0 or stages < 1:
        return signal
    alpha = dt / (tau_s + dt)
    out = np.asarray(signal, dtype=float)
    # Each stage is a one-pole IIR filter: y[n] = y[n-1] + alpha*(x[n]-y[n-1]),
    # i.e. y[n] = (1-alpha)*y[n-1] + alpha*x[n]. lfilter runs the recurrence
    # in compiled code, which matters at multi-million-sample lengths.
    for _ in range(stages):
        out = _one_pole(out, alpha)
    return out


def _one_pole(x: np.ndarray, alpha: float) -> np.ndarray:
    """Single-pole low-pass filter, initialised to the first sample."""
    try:
        from scipy.signal import lfilter, lfilter_zi

        b = np.array([alpha])
        a = np.array([1.0, -(1.0 - alpha)])
        if x.size == 0:
            return x
        zi = lfilter_zi(b, a) * x[0]
        y, _ = lfilter(b, a, x, zi=zi)
        return y
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        y = np.empty_like(x)
        state = x[0] if x.size else 0.0
        for i, value in enumerate(x):
            state += alpha * (value - state)
            y[i] = state
        return y


def _draw_time_of_day(rng: np.random.Generator, hours: list[tuple[float, float]]) -> float:
    """Sample an hour of day from a mixture of normal components."""
    centre, spread = hours[rng.integers(len(hours))]
    return float(np.clip(rng.normal(centre, spread), 0.0, 23.999))


def _draw_duration(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Log-uniform duration, so short events dominate as they do in reality."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def simulate_household(spec: SimSpec | None = None):
    """Generate a synthetic household dataset.

    Returns
    -------
    trace:
        ``DataFrame`` indexed by second with a ``flow_gpm`` column and a
        ``valid`` mask, representing what the meter would record.
    truth:
        ``DataFrame`` indexed identically, one column per fixture, giving the
        exact contribution of each fixture at each second before instrument
        smoothing is applied to the aggregate.
    diary:
        ``DataFrame`` of diary rows as a resident would have written them:
        room page, symbol, minute-resolution start and end, with duration
        unrecorded for short events and some events missing entirely.
    events:
        ``DataFrame`` of true events, for scoring the reconstruction.
    """
    spec = spec or default_spec()
    rng = np.random.default_rng(spec.seed)

    start = pd.Timestamp(spec.start_date)
    n_samples = spec.n_days * 86400 // spec.sample_period_s
    index = pd.date_range(start, periods=n_samples, freq=f"{spec.sample_period_s}s")

    codes = [f.code for f in spec.fixtures]
    truth = np.zeros((n_samples, len(codes)), dtype=np.float32)

    event_rows: list[dict] = []

    for col, fixture in enumerate(spec.fixtures):
        for day in range(spec.n_days):
            n_events = rng.poisson(fixture.per_day)
            for _ in range(n_events):
                hour = _draw_time_of_day(rng, fixture.hours)
                offset = day * 86400 + int(hour * 3600) + int(rng.integers(0, 60))
                rate = fixture.rate_gpm * (
                    1.0 + rng.normal(0.0, fixture.rate_jitter)
                )

                for cycle in range(fixture.cycles):
                    duration = _draw_duration(rng, *fixture.duration_range_s)
                    begin = offset + int(
                        cycle * (fixture.cycle_gap_s + duration)
                    )
                    finish = begin + int(round(duration))
                    if finish >= n_samples:
                        continue
                    truth[begin:finish, col] += rate
                    event_rows.append(
                        {
                            "fixture": fixture.code,
                            "start": index[begin],
                            "end": index[finish],
                            "duration_s": finish - begin,
                            "rate_gpm": rate,
                            "cycle": cycle,
                            "group": len(event_rows) if cycle == 0 else None,
                        }
                    )

    truth_frame = pd.DataFrame(truth, index=index, columns=codes)

    # Aggregate. Superposition loss is applied only where more than one
    # fixture is drawing, since a single fixture is unaffected by it.
    ideal = truth_frame.to_numpy().sum(axis=1)
    concurrent = (truth_frame.to_numpy() > 0).sum(axis=1) > 1
    if spec.superposition_factor != 1.0:
        ideal = np.where(ideal * spec.superposition_factor, ideal, ideal)
        ideal[concurrent] *= spec.superposition_factor

    observed = _meter_response(
        ideal, spec.meter_tau_s, spec.meter_stages, spec.sample_period_s
    )

    # Idle baseline noise, plus occasional larger excursions.
    observed = observed + rng.normal(0.0, spec.noise_sd_gpm, n_samples)
    spikes = rng.random(n_samples) < spec.noise_spike_rate
    observed[spikes] += rng.normal(0.0, spec.noise_spike_sd_gpm, int(spikes.sum()))

    # Logger dropouts, in runs rather than isolated samples.
    valid = np.ones(n_samples, dtype=bool)
    n_dropouts = int(n_samples * spec.dropout_rate / max(spec.dropout_mean_len, 1))
    for _ in range(n_dropouts):
        begin = int(rng.integers(0, n_samples))
        length = 1 + int(rng.exponential(spec.dropout_mean_len))
        valid[begin: begin + length] = False

    trace = pd.DataFrame(
        {"flow_gpm": observed.astype(np.float32), "valid": valid}, index=index
    )

    events = pd.DataFrame(event_rows).sort_values("start").reset_index(drop=True)
    diary = _write_diary(events, spec, rng)

    log.info(
        "simulated %d days, %d events, %.0f gal, %.2f%% samples dropped",
        spec.n_days,
        len(events),
        observed.clip(0).sum() / 60.0,
        100.0 * (~valid).mean(),
    )
    return trace, truth_frame, diary, events


# ---------------------------------------------------------------------------
# Diary generation
# ---------------------------------------------------------------------------

# Which room sheet each fixture is logged on, and the symbol used there.
_DIARY_LAYOUT = {
    "Ad_S": ("primary_bathroom", "S"),
    "Ad_T": ("primary_bathroom", "T"),
    "Ad_F": ("primary_bathroom", "F"),
    "Kid_S": ("upstairs_bathroom", "S"),
    "Kid_T": ("upstairs_bathroom", "T"),
    "Kid_F": ("upstairs_bathroom", "F"),
    "Dwn_S": ("downstairs_bathroom", "S"),
    "Dwn_T": ("downstairs_bathroom", "T"),
    "Dwn_F": ("downstairs_bathroom", "F"),
    "Kitch_F": ("kitchen", "F"),
    "Kitch_r": ("kitchen", "R"),
    "Dish_W": ("kitchen", "D"),
    "Bsmnt_F": ("laundry", "F"),
    "Wash_M": ("laundry", "W"),
}


def _write_diary(events: pd.DataFrame, spec: SimSpec, rng: np.random.Generator) -> pd.DataFrame:
    """Render true events as a resident-written diary.

    Reproduces the properties that make real diaries awkward to align:

    - times rounded to the minute, so a timestamp is a window not an instant;
    - human error of up to about a minute in what was written down;
    - short events logged with identical start and end, carrying no duration;
    - multi-cycle appliances logged as one entry spanning the whole cycle;
    - a small fraction of events not logged at all.
    """
    rows = []
    # Appliance cycles are logged once, spanning all draws.
    grouped = []
    for code, group in events.groupby("fixture"):
        if code in ("Dish_W", "Wash_M"):
            group = group.sort_values("start")
            block = []
            for _, row in group.iterrows():
                if block and (row["start"] - block[-1]["end"]).total_seconds() > 1800:
                    grouped.append(
                        {"fixture": code, "start": block[0]["start"], "end": block[-1]["end"]}
                    )
                    block = []
                block.append(row)
            if block:
                grouped.append(
                    {"fixture": code, "start": block[0]["start"], "end": block[-1]["end"]}
                )
        else:
            for _, row in group.iterrows():
                grouped.append(
                    {"fixture": code, "start": row["start"], "end": row["end"]}
                )

    resolution = spec.diary_resolution_s
    for entry in grouped:
        if rng.random() < spec.diary_miss_rate:
            continue
        page, symbol = _DIARY_LAYOUT[entry["fixture"]]

        jitter_start = rng.normal(0.0, spec.diary_jitter_s)
        noted_start = entry["start"] + pd.Timedelta(seconds=jitter_start)
        noted_start = noted_start.round(f"{resolution}s")

        duration = (entry["end"] - entry["start"]).total_seconds()
        if duration < spec.diary_short_event_s:
            # Too short to have a meaningful minute-resolution end time; the
            # resident writes the same value in both columns, or one minute on.
            noted_end = noted_start + pd.Timedelta(
                seconds=resolution * int(rng.random() < 0.35)
            )
        else:
            jitter_end = rng.normal(0.0, spec.diary_jitter_s)
            noted_end = (
                entry["end"] + pd.Timedelta(seconds=jitter_end)
            ).round(f"{resolution}s")
            if noted_end <= noted_start:
                noted_end = noted_start + pd.Timedelta(seconds=resolution)

        rows.append(
            {
                "page": page,
                "symbol": symbol,
                "date": noted_start.date(),
                "start_time": noted_start.strftime("%I:%M:%S %p").lstrip("0"),
                "end_time": noted_end.strftime("%I:%M:%S %p").lstrip("0"),
                "_noted_start": noted_start,
                "_true_fixture": entry["fixture"],
                "_true_start": entry["start"],
                "_true_end": entry["end"],
            }
        )

    diary = pd.DataFrame(rows)
    if diary.empty:
        return diary
    # Sort on the underlying instant. Sorting on the rendered 12-hour string
    # would interleave morning and evening entries, since "10:05:00 AM" and
    # "10:05:00 PM" differ only in the final field.
    return (
        diary.sort_values("_noted_start")
        .drop(columns=["_noted_start"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_simulated_dataset(
    outdir: Path | str, spec: SimSpec | None = None
) -> dict[str, Path]:
    """Write a simulated dataset to disk in the same layout as real input.

    The trace is written one file per day in the interleaved paired-row form
    the study meter produces, so that ``io.read_trace`` exercises the same
    parsing path it would on real data.
    """
    outdir = Path(outdir)
    (outdir / "trace").mkdir(parents=True, exist_ok=True)

    trace, truth, diary, events = simulate_household(spec)

    written = []
    for day, chunk in trace.groupby(trace.index.normalize()):
        path = outdir / "trace" / f"parsed_data_{day.date()}.csv"
        lines = []
        cumulative = 0.0
        for timestamp, row in chunk.iterrows():
            if not row["valid"]:
                continue
            stamp = timestamp.strftime("%Y-%m-%d %I:%M:%S %p")
            cumulative += max(row["flow_gpm"], 0.0) / 60.0
            # Row one: cumulative only. Row two: instantaneous channels.
            lines.append(f"{stamp},,,,{cumulative:.2f}")
            lines.append(
                f"{stamp},{63.0:.4f},{292.0:.4f},{row['flow_gpm']:.6f},"
            )
        path.write_text("\r\n".join(lines) + "\r\n")
        written.append(path)

    diary_path = outdir / "diary.csv"
    diary.drop(columns=[c for c in diary.columns if c.startswith("_")]).to_csv(
        diary_path, index=False
    )

    truth_path = outdir / "truth_flows.parquet"
    try:
        truth.to_parquet(truth_path)
    except Exception:
        truth_path = outdir / "truth_flows.csv.gz"
        truth.to_csv(truth_path)

    events_path = outdir / "truth_events.csv"
    events.to_csv(events_path, index=False)

    diary_truth_path = outdir / "diary_with_truth.csv"
    diary.to_csv(diary_truth_path, index=False)

    log.info("wrote %d trace files to %s", len(written), outdir / "trace")
    return {
        "trace_dir": outdir / "trace",
        "diary": diary_path,
        "truth_flows": truth_path,
        "truth_events": events_path,
        "diary_with_truth": diary_truth_path,
    }
