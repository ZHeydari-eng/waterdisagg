"""Construction of per-fixture flow labels.

Training requires a value for every fixture at every second: a matrix
``Y`` of shape ``(T, F)`` aligned to the aggregate trace. This module
produces it by either of two routes.

**Loading.** If per-second labels already exist -- for instance because
diary entries were aligned to the trace in a spreadsheet -- they can be read
directly. This is the common case for a dataset that has already been
prepared, and it bypasses matching entirely.

**Matching.** Otherwise the matrix is built from a raw diary. This is not a
matter of applying diary intervals to the trace: diaries are written by hand
to the nearest minute, and short uses are routinely logged with identical
start and end times, carrying no duration at all. A diary timestamp is
therefore a window, not an instant. The procedure is instead to segment the
aggregate into candidate events, match each diary entry to a candidate, and
take **boundaries from the trace** and **identity from the diary**.

Where a single fixture was running, the labelled flow is the observed
aggregate, which preserves genuine within-event variation -- a shower whose
valve is adjusted mid-use does not hold a constant rate, and labelling it as
a flat rectangle would teach the model a shape that does not occur. Where
several fixtures overlap, the aggregate is divided among them in proportion
to their fitted steady-state rates.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .rates import Event, segment_aggregate

log = logging.getLogger(__name__)

__all__ = [
    "LabelSet",
    "MatchReport",
    "load_label_matrix",
    "read_diary",
    "match_diary",
    "build_label_matrix",
]


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LabelSet:
    """Per-fixture flow labels aligned to a trace.

    Attributes
    ----------
    matrix:
        ``(T, F)`` frame of labelled flow in gpm, indexed identically to the
        trace, with one column per modelled fixture in config order.
    events:
        Event table underlying the matrix, with fixture identity and
        trace-derived boundaries. Empty when labels were loaded rather than
        matched.
    labelled_days:
        Calendar days on which any label is present. Partitioning is drawn
        from these days rather than from the full extent of the trace, since
        diary coverage is usually intermittent.
    report:
        Matching diagnostics, or ``None`` for loaded labels.
    """

    matrix: pd.DataFrame
    events: pd.DataFrame
    labelled_days: list[pd.Timestamp]
    report: "MatchReport | None" = None

    @property
    def fixtures(self) -> list[str]:
        return list(self.matrix.columns)

    def active_mask(self, noise_floor: float = 0.05) -> pd.Series:
        """True where any fixture is labelled as running."""
        return (self.matrix > noise_floor).any(axis=1)

    def summary(self) -> dict:
        active = self.active_mask()
        return {
            "fixtures": len(self.matrix.columns),
            "labelled_days": len(self.labelled_days),
            "active_samples": int(active.sum()),
            "active_fraction": float(active.mean()),
            "labelled_volume_gal": round(
                float(self.matrix.to_numpy().sum() / 60.0), 2
            ),
            "events": len(self.events),
        }

    def concurrency(self, noise_floor: float = 0.05) -> pd.Series:
        """Distribution of how many fixtures run simultaneously.

        Reported because it bounds what can be learned about concurrent use
        from observed data alone. In practice the figure is small, which is
        the motivation for synthesising overlapping examples.
        """
        counts = (self.matrix > noise_floor).sum(axis=1)
        active = counts[counts > 0]
        if active.empty:
            return pd.Series(dtype="float64")
        return active.value_counts(normalize=True).sort_index()


@dataclasses.dataclass
class MatchReport:
    """Diagnostics from matching a diary against a trace.

    Low match rates and large orphan counts are the signal that tolerances,
    segmentation thresholds, or the diary itself need attention, so they are
    surfaced rather than silently absorbed.
    """

    n_diary_rows: int = 0
    n_matched: int = 0
    n_unmatched_diary: int = 0
    n_orphan_events: int = 0
    per_fixture: dict[str, dict] = dataclasses.field(default_factory=dict)
    unmatched_rows: pd.DataFrame | None = None
    orphan_events: pd.DataFrame | None = None
    unknown_codes: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def match_rate(self) -> float:
        if not self.n_diary_rows:
            return 0.0
        return self.n_matched / self.n_diary_rows

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "fixture": code,
                "diary_rows": stats.get("diary_rows", 0),
                "matched": stats.get("matched", 0),
                "match_rate": round(
                    stats.get("matched", 0) / max(stats.get("diary_rows", 1), 1), 3
                ),
                "median_offset_s": stats.get("median_offset_s"),
            }
            for code, stats in sorted(self.per_fixture.items())
        ]
        return pd.DataFrame(rows)

    def log_summary(self) -> None:
        log.info(
            "diary matching: %d/%d rows matched (%.1f%%), "
            "%d unmatched, %d orphan flow events",
            self.n_matched, self.n_diary_rows, 100 * self.match_rate,
            self.n_unmatched_diary, self.n_orphan_events,
        )
        if self.unknown_codes:
            log.warning(
                "diary contained codes absent from the config: %s",
                dict(self.unknown_codes),
            )
        if self.match_rate < 0.7 and self.n_diary_rows:
            log.warning(
                "match rate below 70%%; check matching.start_tolerance_s and "
                "the aggregate segmentation thresholds"
            )


# ---------------------------------------------------------------------------
# Route 1: load pre-built labels
# ---------------------------------------------------------------------------


def load_label_matrix(
    path: Path | str | Iterable[Path | str],
    trace,
    config: dict,
    *,
    date_column: str = "Date",
    time_column: str = "Time",
    flow_column: str | None = "Flow",
) -> LabelSet:
    """Read per-second labels that have already been aligned to the trace.

    Expects a wide table with one row per second and one column per fixture,
    holding the flow attributed to that fixture. Columns not named in the
    configuration are ignored, which is how fixtures excluded from the model
    (such as an unused bathtub) are dropped.

    Parameters
    ----------
    path:
        A file or files to concatenate.
    trace:
        The conditioned trace to align against. Labels are reindexed onto its
        index; timestamps outside it are discarded with a warning.
    date_column, time_column:
        Columns holding the date and time-of-day. When ``time_column`` is
        ``None``, ``date_column`` is parsed as a full timestamp.
    flow_column:
        Column holding the aggregate. Not used for labels, but when present
        it is compared against the trace as a consistency check -- a
        mismatch usually means the two files were prepared under different
        timezone conventions.

    Notes
    -----
    No attempt is made to renormalise the labels against the observed
    aggregate. Labels recorded as each fixture's solo rate can exceed the
    measured total slightly when several fixtures run at once, since supply
    pressure falls under concurrent draw. The discrepancy is reported so its
    size is known.
    """
    if isinstance(path, (str, Path)):
        path = [path]
    frames = [pd.read_csv(p) for p in path]
    raw = pd.concat(frames, ignore_index=True)

    # A leading unnamed column is produced by writing a DataFrame with its
    # index; it carries no information.
    raw = raw.loc[:, ~raw.columns.str.match(r"^Unnamed")]

    if date_column not in raw.columns:
        raise ValueError(
            f"date column {date_column!r} not found; got {list(raw.columns)}"
        )

    if time_column is None:
        stamps = pd.to_datetime(raw[date_column], errors="coerce")
    else:
        if time_column not in raw.columns:
            raise ValueError(
                f"time column {time_column!r} not found; got {list(raw.columns)}"
            )
        stamps = pd.to_datetime(
            raw[date_column].astype(str).str.strip()
            + " "
            + raw[time_column].astype(str).str.strip(),
            errors="coerce",
        )

    bad = int(stamps.isna().sum())
    if bad:
        log.warning("dropped %d label rows with unparseable timestamps", bad)
    raw = raw.loc[stamps.notna()].copy()
    raw.index = pd.DatetimeIndex(stamps.dropna())
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()

    codes = [f["code"] for f in config["fixtures"]]
    present = [c for c in codes if c in raw.columns]
    missing = [c for c in codes if c not in raw.columns]
    if missing:
        log.warning("label file has no column for: %s", missing)
    if not present:
        raise ValueError(
            f"no configured fixture columns found in label file; "
            f"expected some of {codes}, got {list(raw.columns)}"
        )

    ignored = [
        c for c in raw.columns
        if c not in codes and c not in {date_column, time_column, flow_column}
    ]
    if ignored:
        log.info("ignoring non-fixture columns: %s", ignored)

    matrix = (
        raw[present]
        .apply(pd.to_numeric, errors="coerce")
        .reindex(trace.flow.index)
        .fillna(0.0)
        .astype("float32")
    )
    for code in missing:
        matrix[code] = np.float32(0.0)
    matrix = matrix[codes]

    outside = len(raw) - int(raw.index.isin(trace.flow.index).sum())
    if outside:
        log.warning(
            "%d label rows fell outside the trace index and were dropped; "
            "check that both were prepared in the same timezone",
            outside,
        )

    if flow_column and flow_column in raw.columns:
        _check_aggregate_agreement(raw[flow_column], trace, matrix)

    labelled_days = _days_with_labels(matrix, config)
    events = events_from_matrix(matrix, config)

    label_set = LabelSet(
        matrix=matrix, events=events, labelled_days=labelled_days
    )
    log.info("loaded labels: %s", label_set.summary())
    return label_set


def _check_aggregate_agreement(
    declared: pd.Series, trace, matrix: pd.DataFrame
) -> None:
    """Compare the label file's own aggregate column against the trace."""
    aligned = pd.to_numeric(declared, errors="coerce").reindex(
        trace.flow.index
    )
    both = aligned.notna() & trace.valid
    if both.sum() < 100:
        return
    difference = (aligned[both] - trace.flow[both]).abs()
    if float(difference.mean()) > 0.05:
        log.warning(
            "label file's Flow column differs from the trace by %.3f gpm on "
            "average; the two may be misaligned in time",
            float(difference.mean()),
        )

    # How closely do solo-rate labels reconcile with the measured total?
    labelled = matrix.sum(axis=1)[both]
    active = labelled > 0.05
    if active.sum() > 100:
        ratio = float(
            (trace.flow[both][active] / labelled[active]).median()
        )
        log.info(
            "median ratio of measured aggregate to summed labels: %.3f "
            "(1.0 indicates exact additivity)", ratio
        )


def _days_with_labels(matrix: pd.DataFrame, config: dict) -> list[pd.Timestamp]:
    """Calendar days on which at least one fixture is labelled active."""
    floor = config["meter"]["noise_floor_gpm"]
    active = (matrix > floor).any(axis=1)
    if not active.any():
        return []
    return sorted({pd.Timestamp(d) for d in matrix.index[active].normalize()})


def events_from_matrix(matrix: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Recover a per-fixture event table from a per-second label matrix.

    Uses each fixture's configured segmentation parameters so that a
    dishwasher's genuine inter-fill gaps are not read as separate events.
    """
    from .rates import segment

    floor = config["meter"]["noise_floor_gpm"]
    defaults = config["segmentation"]
    period = config["meter"]["sample_period_s"]
    by_code = {f["code"]: f for f in config["fixtures"]}

    rows: list[Event] = []
    for code in matrix.columns:
        series = matrix[code]
        if not (series > floor).any():
            continue
        local = (by_code.get(code, {}) or {}).get("segmentation", {}) or {}
        rows.extend(
            segment(
                series,
                on_threshold=floor,
                off_threshold=floor / 2,
                min_duration_s=local.get(
                    "min_duration_s", defaults["min_duration_s"]
                ),
                bridge_gap_s=local.get("bridge_gap_s", defaults["bridge_gap_s"]),
                sample_period_s=period,
                fixture=code,
            )
        )

    from .rates import events_to_frame

    return events_to_frame(rows)


# ---------------------------------------------------------------------------
# Route 2: match a raw diary
# ---------------------------------------------------------------------------


def read_diary(path: Path | str, config: dict) -> pd.DataFrame:
    """Read a hand-recorded water diary and resolve fixture identities.

    Two identification schemes are supported, selected by which columns are
    present. Either a single column names the fixture directly using its
    configured code, or a room column and a symbol column are combined --
    matching the layout of paper diary sheets, which are headed by room and
    use a single letter per end use, so that the same letter means different
    fixtures on different sheets.
    """
    diary = pd.read_csv(path)
    diary = diary.loc[:, ~diary.columns.str.match(r"^Unnamed")]
    spec = config.get("matching", {}) or {}

    fixture_col = spec.get("fixture_column", "fixture")
    page_col = spec.get("page_column", "page")
    symbol_col = spec.get("symbol_column", "symbol")

    lookup: dict[tuple[str, str], str] = {}
    for fixture in config["fixtures"]:
        entry = fixture.get("diary") or {}
        if "page" in entry and "symbol" in entry:
            lookup[(str(entry["page"]), str(entry["symbol"]))] = fixture["code"]
    # Excluded fixtures are resolved too, so that logged bathtub use is
    # recognised and dropped rather than reported as an unknown code.
    excluded = set()
    for fixture in config.get("excluded_fixtures", []) or []:
        entry = fixture.get("diary") or {}
        if "page" in entry and "symbol" in entry:
            lookup[(str(entry["page"]), str(entry["symbol"]))] = fixture["code"]
        excluded.add(fixture["code"])

    if fixture_col in diary.columns:
        diary["_fixture"] = diary[fixture_col].astype(str).str.strip()
    elif page_col in diary.columns and symbol_col in diary.columns:
        keys = list(
            zip(
                diary[page_col].astype(str).str.strip(),
                diary[symbol_col].astype(str).str.strip(),
            )
        )
        diary["_fixture"] = [lookup.get(k) for k in keys]
    else:
        raise ValueError(
            f"diary must contain either {fixture_col!r} or both {page_col!r} "
            f"and {symbol_col!r}; got {list(diary.columns)}"
        )

    diary["_start"] = _parse_diary_times(diary, spec, which="start")
    diary["_end"] = _parse_diary_times(diary, spec, which="end")

    # Entries with equal start and end carry no usable duration: the resident
    # noted a single minute for a use shorter than the diary's resolution.
    diary["_duration_known"] = (
        diary["_end"].notna()
        & diary["_start"].notna()
        & (diary["_end"] > diary["_start"])
    )

    diary["_excluded"] = diary["_fixture"].isin(excluded)
    before = len(diary)
    diary = diary.loc[diary["_start"].notna()].copy()
    if len(diary) < before:
        log.warning(
            "dropped %d diary rows with unparseable start times",
            before - len(diary),
        )
    return diary.sort_values("_start").reset_index(drop=True)


def _parse_diary_times(
    diary: pd.DataFrame, spec: dict, which: str
) -> pd.Series:
    """Assemble timestamps from whatever date and time columns are present."""
    time_col = spec.get(f"{which}_time_column", f"{which}_time")
    stamp_col = spec.get(f"{which}_column", f"{which}_datetime")
    date_col = spec.get("date_column", "date")
    fmt = spec.get("time_format")

    if stamp_col in diary.columns:
        return pd.to_datetime(diary[stamp_col], errors="coerce")
    if time_col not in diary.columns:
        return pd.Series(pd.NaT, index=diary.index)
    if date_col not in diary.columns:
        raise ValueError(
            f"diary has {time_col!r} but no {date_col!r} column to date it by"
        )

    combined = (
        diary[date_col].astype(str).str.strip()
        + " "
        + diary[time_col].astype(str).str.strip()
    )
    if fmt:
        return pd.to_datetime(combined, format=fmt, errors="coerce")
    return pd.to_datetime(combined, errors="coerce")


def match_diary(
    trace,
    diary: pd.DataFrame,
    config: dict,
    *,
    candidates: Sequence[Event] | None = None,
) -> tuple[pd.DataFrame, MatchReport]:
    """Assign diary entries to flow events in the trace.

    Matching is global rather than greedy: all admissible pairings are scored
    and the assignment minimising total cost is chosen. A greedy pass gives
    different answers depending on row order, which matters when several
    entries fall close together -- a household getting ready in the morning
    produces exactly that pattern.

    A candidate flow event may receive more than one diary entry, which is
    how concurrent use is represented: two fixtures running together produce
    a single merged excursion in the aggregate.

    Returns
    -------
    matched:
        One row per successful match, carrying the diary's fixture identity
        and the trace-derived boundaries.
    report:
        Diagnostics, including unmatched entries and orphan flow events.
    """
    spec = config.get("matching", {}) or {}
    tolerance = float(spec.get("start_tolerance_s", 180))

    if candidates is None:
        candidates = segment_aggregate(trace, config)
    candidates = list(candidates)

    usable = diary.loc[~diary["_excluded"] & diary["_fixture"].notna()]
    report = MatchReport(n_diary_rows=len(usable))

    unknown = diary.loc[diary["_fixture"].isna()]
    if not unknown.empty:
        page_col = spec.get("page_column", "page")
        symbol_col = spec.get("symbol_column", "symbol")
        if page_col in unknown.columns and symbol_col in unknown.columns:
            keys = unknown[page_col].astype(str) + "/" + unknown[symbol_col].astype(str)
            report.unknown_codes = keys.value_counts().to_dict()

    if usable.empty or not candidates:
        report.n_orphan_events = len(candidates)
        report.log_summary()
        return pd.DataFrame(), report

    # Cost matrix: seconds between the diary's noted start and the candidate's
    # observed start, with inadmissible pairings left out.
    diary_starts = usable["_start"].to_numpy()
    event_starts = np.array([e.start.to_datetime64() for e in candidates])
    offsets = (
        (diary_starts[:, None] - event_starts[None, :])
        .astype("timedelta64[s]")
        .astype(float)
    )
    cost = np.abs(offsets)

    # Duration agreement, where the diary recorded one, breaks ties between
    # candidates at similar distance.
    duration_tolerance = float(spec.get("duration_tolerance_frac", 0.5))
    event_durations = np.array([e.duration_s for e in candidates])
    known = usable["_duration_known"].to_numpy()
    diary_durations = np.where(
        known,
        (usable["_end"] - usable["_start"]).dt.total_seconds().to_numpy(),
        np.nan,
    )
    with np.errstate(invalid="ignore"):
        relative = np.abs(
            event_durations[None, :] - diary_durations[:, None]
        ) / np.maximum(diary_durations[:, None], 1.0)
    penalty = np.where(
        np.isnan(relative), 0.0, np.minimum(relative, 4.0) * tolerance * 0.25
    )
    cost = cost + penalty
    cost[np.abs(offsets) > tolerance] = np.inf

    assignment = _assign(cost)

    rows = []
    matched_events: set[int] = set()
    for diary_i, event_i in assignment.items():
        entry = usable.iloc[diary_i]
        event = candidates[event_i]
        matched_events.add(event_i)
        rows.append(
            {
                "fixture": entry["_fixture"],
                "start": event.start,
                "end": event.end,
                "duration_s": event.duration_s,
                "mean_gpm": event.mean_gpm,
                "plateau_gpm": event.plateau_gpm,
                "volume_gal": event.volume_gal,
                "diary_start": entry["_start"],
                "offset_s": (entry["_start"] - event.start).total_seconds(),
                "candidate": event_i,
                "shared_candidate": False,
            }
        )

    matched = pd.DataFrame(rows)
    if not matched.empty:
        shared = matched["candidate"].duplicated(keep=False)
        matched["shared_candidate"] = shared
        matched = matched.sort_values("start").reset_index(drop=True)

    report.n_matched = len(matched)
    report.n_unmatched_diary = len(usable) - len(matched)
    report.n_orphan_events = len(candidates) - len(matched_events)

    unmatched_idx = set(range(len(usable))) - set(assignment)
    if unmatched_idx:
        report.unmatched_rows = usable.iloc[sorted(unmatched_idx)][
            ["_fixture", "_start", "_end"]
        ].rename(
            columns={"_fixture": "fixture", "_start": "start", "_end": "end"}
        )
    orphan_idx = set(range(len(candidates))) - matched_events
    if orphan_idx:
        from .rates import events_to_frame

        report.orphan_events = events_to_frame(
            [candidates[i] for i in sorted(orphan_idx)]
        )

    for code, group in (
        matched.groupby("fixture") if not matched.empty else []
    ):
        report.per_fixture[code] = {
            "diary_rows": int((usable["_fixture"] == code).sum()),
            "matched": len(group),
            "median_offset_s": round(float(group["offset_s"].median()), 1),
        }
    for code in usable["_fixture"].unique():
        report.per_fixture.setdefault(
            code,
            {
                "diary_rows": int((usable["_fixture"] == code).sum()),
                "matched": 0,
                "median_offset_s": None,
            },
        )

    report.log_summary()
    return matched, report


def _assign(cost: np.ndarray) -> dict[int, int]:
    """Minimum-cost assignment of diary rows to candidate events.

    Uses the Hungarian algorithm where SciPy is available. Because a single
    candidate may legitimately host several diary entries -- concurrent use
    appears as one merged excursion -- the problem is solved repeatedly,
    allowing one entry per candidate per round, until no admissible pairing
    remains.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return _assign_greedy(cost)

    remaining = cost.copy()
    result: dict[int, int] = {}

    while True:
        admissible = np.isfinite(remaining)
        if not admissible.any():
            break
        rows = np.flatnonzero(admissible.any(axis=1))
        cols = np.flatnonzero(admissible.any(axis=0))
        block = remaining[np.ix_(rows, cols)]
        # linear_sum_assignment cannot accept infinities; substitute a value
        # large enough to never be selected in preference to a real pairing.
        big = np.nanmax(block[np.isfinite(block)]) * 1e3 + 1.0
        finite = np.where(np.isfinite(block), block, big)
        r, c = linear_sum_assignment(finite)

        progressed = False
        for ri, ci in zip(r, c):
            if not np.isfinite(block[ri, ci]):
                continue
            result[int(rows[ri])] = int(cols[ci])
            remaining[rows[ri], :] = np.inf
            progressed = True
        if not progressed:
            break
    return result


def _assign_greedy(cost: np.ndarray) -> dict[int, int]:
    """Fallback assignment, taking the cheapest admissible pairing first."""
    result: dict[int, int] = {}
    work = cost.copy()
    while np.isfinite(work).any():
        flat = int(np.nanargmin(np.where(np.isfinite(work), work, np.inf)))
        i, j = np.unravel_index(flat, work.shape)
        if not np.isfinite(work[i, j]):
            break
        result[int(i)] = int(j)
        work[i, :] = np.inf
    return result


# ---------------------------------------------------------------------------
# Label matrix construction
# ---------------------------------------------------------------------------


def build_label_matrix(
    trace,
    matched: pd.DataFrame,
    config: dict,
    rates: dict[str, float] | None = None,
    *,
    report: MatchReport | None = None,
) -> LabelSet:
    """Turn matched events into a per-second per-fixture flow matrix.

    Where one fixture is active, it receives the observed aggregate. Where
    several overlap, the aggregate is divided in proportion to their fitted
    steady-state rates -- an approximation that is close to exact when supply
    pressure holds up under concurrent draw, and which at least conserves the
    measured total when it does not.

    Parameters
    ----------
    rates:
        Fitted steady-state rates by fixture code. Falls back to the
        configured values, then to equal division, with a warning.
    """
    codes = [f["code"] for f in config["fixtures"]]
    index = trace.flow.index
    # Allocated directly rather than via DataFrame.to_numpy(), which may
    # return a read-only view depending on the pandas version.
    values = np.zeros((len(index), len(codes)), dtype=np.float32)
    matrix = pd.DataFrame(values.copy(), index=index, columns=codes)

    if matched.empty:
        log.warning("no matched events; label matrix is empty")
        return LabelSet(matrix, matched, [], report)

    configured = {f["code"]: f.get("rate_gpm") for f in config["fixtures"]}
    rates = dict(rates or {})
    for code in codes:
        if code not in rates or not np.isfinite(rates.get(code, np.nan)):
            fallback = configured.get(code)
            if fallback is not None:
                rates[code] = float(fallback)

    dead = trace.deadbanded()
    positions = {code: i for i, code in enumerate(codes)}

    # Occupancy: which fixtures are active at each second.
    occupancy = np.zeros((len(index), len(codes)), dtype=bool)
    starts = index.searchsorted(matched["start"].to_numpy(), side="left")
    ends = index.searchsorted(matched["end"].to_numpy(), side="left")
    for (start_i, end_i, code) in zip(starts, ends, matched["fixture"]):
        if code not in positions or end_i <= start_i:
            continue
        occupancy[start_i:end_i, positions[code]] = True

    n_active = occupancy.sum(axis=1)
    aggregate = dead.to_numpy(dtype=np.float32)

    # Single-fixture intervals take the observed aggregate directly, which
    # preserves real within-event variation.
    solo = n_active == 1
    if solo.any():
        values[solo] = occupancy[solo] * aggregate[solo, None]

    # Concurrent intervals divide the aggregate by fitted rate.
    concurrent = n_active > 1
    if concurrent.any():
        weights = np.array(
            [rates.get(code, np.nan) for code in codes], dtype=np.float32
        )
        if not np.isfinite(weights).all():
            log.warning(
                "no rate available for %s; dividing equally during "
                "concurrent use",
                [c for c, w in zip(codes, weights) if not np.isfinite(w)],
            )
            weights = np.where(np.isfinite(weights), weights, 1.0)
        masked = occupancy[concurrent] * weights[None, :]
        totals = masked.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        values[concurrent] = masked / totals * aggregate[concurrent, None]

        log.info(
            "%d samples (%.2f%% of active time) involve concurrent use",
            int(concurrent.sum()),
            100.0 * concurrent.sum() / max((n_active > 0).sum(), 1),
        )

    matrix = pd.DataFrame(values, index=index, columns=codes)
    labelled_days = _days_with_labels(matrix, config)

    label_set = LabelSet(
        matrix=matrix,
        events=matched.drop(columns=["candidate"], errors="ignore"),
        labelled_days=labelled_days,
        report=report,
    )
    log.info("built labels: %s", label_set.summary())
    return label_set
