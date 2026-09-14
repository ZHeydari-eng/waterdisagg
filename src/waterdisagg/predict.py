"""Inference over a continuous record.

Training operates on short windows; analysis needs a per-fixture estimate for
every second of a multi-year trace. This module bridges the two.

The reconciliation matters. Windows overlap -- at a 60-second length and
15-second stride each second falls inside four of them -- so every timestep
receives several estimates. Averaging them is possible but blurs event edges,
because a window that barely contains an event's onset sees only a rising
partial signal and predicts accordingly. Instead, each window contributes
only the slice at its centre, and the slices tile the timeline exactly. Every
second is then predicted by the window that has the most context on both
sides of it.

For a causal (unidirectional) model the corresponding choice is the trailing
slice, since a causal model has no forward context and its best-informed
position is its last timestep.

Memory is the other constraint. Seven years at 1 Hz is upwards of two hundred
million samples, so inference runs in chunks over the timeline and writes
into a preallocated output array rather than concatenating batch results.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["PredictionSet", "predict_trace", "predict_windows"]


@dataclasses.dataclass
class PredictionSet:
    """Per-second per-fixture predictions aligned to a trace."""

    flow: pd.DataFrame
    activation: pd.DataFrame | None
    unattributed: pd.Series | None
    covered: pd.Series

    @property
    def fixtures(self) -> list[str]:
        return list(self.flow.columns)

    def summary(self) -> dict:
        total = self.flow.to_numpy().sum() / 60.0
        row = {
            "samples": len(self.flow),
            "covered": int(self.covered.sum()),
            "predicted_volume_gal": round(float(total), 2),
        }
        if self.unattributed is not None:
            row["unattributed_gal"] = round(
                float(self.unattributed.sum() / 60.0), 2
            )
            denominator = total + float(self.unattributed.sum() / 60.0)
            row["unattributed_frac"] = (
                round(float(self.unattributed.sum() / 60.0) / denominator, 4)
                if denominator > 0 else 0.0
            )
        return row

    def volume_by_fixture(self) -> pd.Series:
        """Total predicted volume per fixture, in gallons."""
        return (self.flow.sum() / 60.0).sort_values(ascending=False)

    def daily_volume(self) -> pd.DataFrame:
        """Predicted volume per fixture per calendar day."""
        return self.flow.groupby(self.flow.index.normalize()).sum() / 60.0

    def reconciliation(self, trace) -> pd.Series:
        """How much of the measured flow the fixtures account for.

        Should sit close to one wherever the unattributed channel is quiet.
        A persistent shortfall points to flow the model cannot place, which
        is worth inspecting before trusting downstream event counts.
        """
        measured = trace.deadbanded().reindex(self.flow.index)
        predicted = self.flow.sum(axis=1)
        active = measured > 0.05
        if not active.any():
            return pd.Series(dtype="float64")
        return (predicted[active] / measured[active]).describe()


def _slice_bounds(length: int, stride: int, emit: str) -> tuple[int, int]:
    """Which part of each window to keep so that slices tile exactly.

    Returns the offset of the retained slice within the window and its width.
    The width is the stride, so consecutive windows abut without gap or
    overlap.
    """
    if stride > length:
        raise ValueError(
            f"stride {stride} exceeds window length {length}; "
            "the timeline cannot be tiled"
        )
    if emit == "center":
        offset = (length - stride) // 2
    elif emit in ("tail", "last"):
        offset = length - stride
    elif emit in ("head", "first"):
        offset = 0
    else:
        raise ValueError(f"unknown emit mode {emit!r}")
    return offset, stride


def predict_windows(model, x: np.ndarray, batch_size: int = 256, device=None):
    """Run the model over a stack of windows, returning raw outputs."""
    import torch

    model.eval()
    flows, logits, allocations = [], [], []
    with torch.no_grad():
        for begin in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[begin: begin + batch_size]).unsqueeze(-1)
            if device is not None:
                batch = batch.to(device)
            out = model(batch)
            flows.append(out["flow"].cpu().numpy())
            logits.append(torch.sigmoid(out["logits"]).cpu().numpy())
            allocations.append(out["alloc"][..., -1].cpu().numpy())
    return (
        np.concatenate(flows),
        np.concatenate(logits),
        np.concatenate(allocations),
    )


def predict_trace(
    model,
    trace,
    config: dict,
    fixtures: Sequence[str],
    *,
    batch_size: int = 256,
    chunk_days: int = 7,
    device: str | None = None,
    skip_idle: bool = True,
) -> PredictionSet:
    """Predict per-fixture flow for every second of a trace.

    Parameters
    ----------
    fixtures:
        Fixture codes in the order the model's outputs correspond to. Taken
        from the label matrix used in training, not re-derived, so that a
        change to the configuration cannot silently reorder columns.
    chunk_days:
        Days processed per pass. Bounds peak memory on long records.
    skip_idle:
        Skip windows whose aggregate never exceeds the noise floor. The
        allocation is multiplied by the observed flow, so a window with no
        flow predicts zero for every fixture regardless; evaluating it wastes
        the majority of the compute on a household that is idle most of the
        time. Idle stretches are left at zero and marked as covered.

    Notes
    -----
    Predictions are written into a preallocated array rather than
    accumulated, because concatenating per-batch results over a multi-year
    trace would exceed available memory.
    """
    import torch

    period = config["meter"]["sample_period_s"]
    length = int(config["windows"]["length_s"] / period)
    stride = int(config["windows"]["stride_s"] / period)
    emit = config["windows"].get("emit", "center")
    floor = config["meter"]["noise_floor_gpm"]

    offset, width = _slice_bounds(length, stride, emit)

    index = trace.flow.index
    flow_values = trace.flow.to_numpy(dtype=np.float32)
    n = len(index)
    n_fixtures = len(fixtures)

    predicted = np.zeros((n, n_fixtures), dtype=np.float32)
    activation = np.zeros((n, n_fixtures), dtype=np.float32)
    unattributed = np.zeros(n, dtype=np.float32)
    covered = np.zeros(n, dtype=bool)

    if n < length:
        log.warning("trace shorter than one window; nothing predicted")
        return PredictionSet(
            pd.DataFrame(predicted, index=index, columns=list(fixtures)),
            pd.DataFrame(activation, index=index, columns=list(fixtures)),
            pd.Series(unattributed, index=index),
            pd.Series(covered, index=index),
        )

    resolved = device
    if resolved is None:
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(resolved)

    # Prefix sum of activity, so a window's idleness is an O(1) test.
    active_any = (np.abs(flow_values) > floor).astype(np.int32)
    cumulative = np.concatenate([[0], np.cumsum(active_any)])

    chunk_samples = int(chunk_days * 86400 / period)
    total_windows = 0

    for chunk_start in range(0, n, chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, n)
        # Extend backwards so windows near the chunk edge keep their context.
        first = max(chunk_start - offset, 0)
        last = min(chunk_end + (length - offset), n)

        starts = np.arange(first, last - length + 1, stride)
        if starts.size == 0:
            continue
        # Retain only windows whose emitted slice lands inside this chunk,
        # so no timestep is written twice.
        slice_start = starts + offset
        in_chunk = (slice_start >= chunk_start) & (slice_start < chunk_end)
        starts = starts[in_chunk]
        if starts.size == 0:
            continue

        if skip_idle:
            has_flow = (
                cumulative[starts + length] - cumulative[starts]
            ) > 0
            idle_starts = starts[~has_flow]
            for begin in idle_starts:
                covered[begin + offset: begin + offset + width] = True
            starts = starts[has_flow]

        if starts.size == 0:
            continue

        rows = starts[:, None] + np.arange(length)[None, :]
        windows = flow_values[rows]

        flows, probabilities, escape = predict_windows(
            model, windows, batch_size=batch_size, device=resolved
        )

        for i, begin in enumerate(starts):
            write_from = begin + offset
            write_to = min(write_from + width, n)
            take = write_to - write_from
            predicted[write_from:write_to] = flows[i, offset: offset + take]
            activation[write_from:write_to] = probabilities[
                i, offset: offset + take
            ]
            unattributed[write_from:write_to] = escape[
                i, offset: offset + take
            ]
            covered[write_from:write_to] = True

        total_windows += len(starts)
        log.debug(
            "chunk %s..%s: %d windows",
            index[chunk_start], index[chunk_end - 1], len(starts),
        )

    gaps = int((~covered).sum())
    if gaps:
        log.info(
            "%d samples (%.3f%%) fall outside any emitted slice, at the very "
            "start and end of the record",
            gaps, 100.0 * gaps / n,
        )

    prediction_set = PredictionSet(
        flow=pd.DataFrame(predicted, index=index, columns=list(fixtures)),
        activation=pd.DataFrame(
            activation, index=index, columns=list(fixtures)
        ),
        unattributed=pd.Series(unattributed, index=index, name="unattributed"),
        covered=pd.Series(covered, index=index, name="covered"),
    )
    log.info(
        "predicted %d windows over %d samples: %s",
        total_windows, n, prediction_set.summary(),
    )
    return prediction_set
