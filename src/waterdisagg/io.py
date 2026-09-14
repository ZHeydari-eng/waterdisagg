"""Loading and conditioning of raw smart-meter traces.

The reader deals with three properties of real meter output that break naive
``pd.read_csv`` handling:

1. **Interleaved rows.** The study meter emits two rows per timestamp -- one
   carrying only the cumulative volume, the next carrying the instantaneous
   channels. Neither row is complete on its own.
2. **Irregular sampling.** A nominally 1 Hz logger drops occasional seconds
   and duplicates others. Gaps must be tracked rather than silently filled:
   zero-filling a sensor dropout is indistinguishable, downstream, from a
   genuine period of no flow.
3. **A non-zero idle baseline.** Flow oscillates around zero when nothing is
   running, including small negative excursions. A deadband is applied before
   any thresholding, but the raw values are preserved for the model.

The public entry point is :func:`read_trace`, which returns a
:class:`Trace` holding a regular 1 Hz series plus a boolean validity mask.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["Trace", "read_trace", "read_trace_dir"]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Trace:
    """A regularly-sampled aggregate flow trace.

    Attributes
    ----------
    flow:
        Flow rate in gpm on a gap-free :class:`~pandas.DatetimeIndex` at the
        configured sample period. Missing samples are filled with 0.0 so that
        array operations are well defined; consult ``valid`` to distinguish
        filled samples from observed ones.
    valid:
        ``True`` where the sample was actually observed. Loss masking during
        training and volume accounting both key off this.
    sample_period_s:
        Spacing of the index, in seconds.
    noise_floor_gpm:
        Deadband applied by :meth:`deadbanded`.
    """

    flow: pd.Series
    valid: pd.Series
    sample_period_s: int
    noise_floor_gpm: float

    def __post_init__(self) -> None:
        if not self.flow.index.equals(self.valid.index):
            raise ValueError("flow and valid must share an index")

    # -- derived views ---------------------------------------------------

    def deadbanded(self) -> pd.Series:
        """Flow with the idle noise band collapsed to exactly zero.

        Used for event segmentation and volume integration. The model is fed
        :attr:`flow` instead, so that it can learn the noise characteristics
        rather than inheriting our threshold choice.
        """
        out = self.flow.copy()
        out[out.abs() < self.noise_floor_gpm] = 0.0
        return out.clip(lower=0.0)

    def volume_gal(self) -> float:
        """Total volume over the trace, integrating only observed samples."""
        dead = self.deadbanded().where(self.valid, 0.0)
        return float(dead.sum() * self.sample_period_s / 60.0)

    def local_days(self) -> list[pd.Timestamp]:
        """Sorted list of calendar days present in the index."""
        return sorted({pd.Timestamp(d) for d in self.flow.index.normalize()})

    def day(self, day: pd.Timestamp | str) -> "Trace":
        """Return the sub-trace covering a single calendar day."""
        day = pd.Timestamp(day).normalize()
        end = day + pd.Timedelta(days=1)
        sel = (self.flow.index >= day) & (self.flow.index < end)
        return dataclasses.replace(
            self, flow=self.flow[sel], valid=self.valid[sel]
        )

    def summary(self) -> dict:
        """Diagnostic summary, for logging and the prepare script's report."""
        n = len(self.flow)
        return {
            "start": str(self.flow.index[0]) if n else None,
            "end": str(self.flow.index[-1]) if n else None,
            "samples": n,
            "observed": int(self.valid.sum()),
            "missing": int((~self.valid).sum()),
            "missing_frac": float((~self.valid).mean()) if n else 0.0,
            "days": len(self.local_days()),
            "volume_gal": round(self.volume_gal(), 2),
            "peak_gpm": round(float(self.flow.max()), 3) if n else None,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_paired_rows(
    path: Path,
    columns: Sequence[str],
    timestamp_format: str | None,
    flow_column: str,
    has_header: bool,
) -> pd.DataFrame:
    """Parse a CSV whose records are split across consecutive rows.

    Each timestamp appears on more than one row, with a disjoint set of
    populated fields on each. Grouping by timestamp and taking the mean of the
    populated values reassembles the record; the mean also collapses the
    duplicate samples the logger occasionally emits.
    """
    raw = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else list(columns),
        dtype=str,
        keep_default_na=True,
    )

    if "timestamp" not in raw.columns:
        raise ValueError(
            f"{path.name}: expected a 'timestamp' column; got {list(raw.columns)}"
        )
    if flow_column not in raw.columns:
        raise ValueError(
            f"{path.name}: flow column {flow_column!r} not present; "
            f"got {list(raw.columns)}"
        )

    ts = pd.to_datetime(
        raw["timestamp"].str.strip(),
        format=timestamp_format,
        errors="coerce",
    )
    bad = int(ts.isna().sum())
    if bad:
        log.warning("%s: dropped %d rows with unparseable timestamps", path.name, bad)

    numeric = raw.drop(columns=["timestamp"]).apply(
        pd.to_numeric, errors="coerce"
    )
    numeric["timestamp"] = ts
    numeric = numeric.dropna(subset=["timestamp"])

    # Rows carrying no flow reading contribute nothing to the flow channel.
    # They are retained here so that other channels (e.g. cumulative volume)
    # remain available to callers that want them.
    return numeric.groupby("timestamp", sort=True).mean(numeric_only=True)


def _apply_timezone_shift(frame: pd.DataFrame, shift_hours: float) -> pd.DataFrame:
    """Shift the index by a fixed offset.

    Some acquisition scripts record in a timezone other than the household's
    local time. The correction is a fixed offset rather than a proper timezone
    conversion, which is valid when the logging and local zones observe the
    same daylight-saving transitions.

    Applying this shift means the calendar-day boundaries in the source files
    no longer align with local midnight, so callers should reassemble days
    from the concatenated trace rather than trusting per-file extents.
    """
    if not shift_hours:
        return frame
    shifted = frame.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=shift_hours)
    return shifted


def _regularise(
    flow: pd.Series, sample_period_s: int
) -> tuple[pd.Series, pd.Series]:
    """Place a series on a gap-free index, reporting which samples are real."""
    if flow.empty:
        empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        return empty, empty.astype(bool)

    full = pd.date_range(
        flow.index.min(),
        flow.index.max(),
        freq=f"{sample_period_s}s",
    )
    reindexed = flow.reindex(full)
    valid = reindexed.notna()
    return reindexed.fillna(0.0), valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_trace(paths: Path | str | Iterable[Path | str], config: dict) -> Trace:
    """Read one or more raw trace files into a single conditioned ``Trace``.

    Parameters
    ----------
    paths:
        A file, or an iterable of files, to concatenate. Files may overlap or
        abut; duplicate timestamps are averaged.
    config:
        Parsed household configuration. The ``meter`` block is consulted for
        the raw layout, sample period, noise floor and timezone correction.

    Notes
    -----
    Concatenating before regularising is deliberate. Source files are often
    split by calendar day in the *logger's* timezone, and may carry a few
    samples belonging to the adjacent day. Assembling the full record first
    and then re-deriving days avoids inheriting those boundaries.
    """
    meter = config["meter"]
    fmt = meter["raw_format"]

    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no trace files given")

    layout = fmt.get("layout", "paired_rows")
    frames = []
    for path in paths:
        if layout == "paired_rows":
            frame = _parse_paired_rows(
                path,
                columns=fmt["columns"],
                timestamp_format=fmt.get("timestamp_format"),
                flow_column=fmt["flow_column"],
                has_header=fmt.get("has_header", False),
            )
        elif layout == "flat":
            frame = pd.read_csv(
                path,
                header=0 if fmt.get("has_header", True) else None,
                names=None if fmt.get("has_header", True) else list(fmt["columns"]),
            )
            frame["timestamp"] = pd.to_datetime(
                frame["timestamp"], format=fmt.get("timestamp_format"),
                errors="coerce",
            )
            frame = (
                frame.dropna(subset=["timestamp"])
                .groupby("timestamp", sort=True)
                .mean(numeric_only=True)
            )
        else:
            raise ValueError(f"unknown raw layout {layout!r}")
        frames.append(frame)
        log.debug("%s: %d timestamps", path.name, len(frame))

    combined = pd.concat(frames).groupby(level=0).mean()

    tz = meter.get("timezone", {}) or {}
    combined = _apply_timezone_shift(combined, tz.get("shift_hours", 0) or 0)

    flow = combined[fmt["flow_column"]].dropna()
    flow, valid = _regularise(flow, meter["sample_period_s"])

    trace = Trace(
        flow=flow,
        valid=valid,
        sample_period_s=meter["sample_period_s"],
        noise_floor_gpm=meter["noise_floor_gpm"],
    )
    log.info("loaded trace: %s", trace.summary())
    return trace


def read_trace_dir(directory: Path | str, config: dict, pattern: str = "*.csv") -> Trace:
    """Read every matching file in a directory as one trace."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {directory}")
    return read_trace(files, config)
