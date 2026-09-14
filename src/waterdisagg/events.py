"""Assembly of discrete events from per-second predictions.

Every behavioural analysis in this work operates on events -- a shower with a
start, an end and a volume -- not on individual samples. This module performs
that conversion, and it is where most of the practical accuracy is won or
lost: a model with good per-sample error can still produce a useless event
table if the assembly rules are wrong.

Three mechanisms are applied, each addressing a distinct failure:

**Hysteresis.** A single threshold on a signal that hovers near it chops one
physical event into a run of fragments. Separate rising and falling
thresholds eliminate that.

**Gap bridging.** Genuine brief interruptions occur within a single use, and
appliances draw water in several fills per cycle. The tolerated gap differs
by an order of magnitude across end uses, so it is set per fixture.

**Minimum duration.** Short excursions are usually misattribution rather than
use, particularly for fixtures that resemble a louder neighbour.

Thresholds are fitted rather than assumed. :func:`fit_thresholds` selects
each fixture's threshold by maximising event-level agreement on a validation
partition, which is the quantity that matters downstream, rather than
per-sample error.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from .rates import Event, events_to_frame, segment

log = logging.getLogger(__name__)

__all__ = [
    "assemble_events",
    "fit_thresholds",
    "ThresholdFit",
    "match_events",
    "merge_close_events",
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_events(
    predictions: pd.DataFrame,
    config: dict,
    thresholds: dict[str, dict] | None = None,
    *,
    valid: pd.Series | None = None,
    aggregate: pd.Series | None = None,
) -> pd.DataFrame:
    """Convert per-second per-fixture predictions into an event table.

    Parameters
    ----------
    predictions:
        ``(T, F)`` frame of predicted flow in gpm, indexed by timestamp. May
        be predicted flow, or the activation probabilities from the auxiliary
        head -- in the latter case supply ``thresholds`` in probability units.
    thresholds:
        Per-fixture parameters as produced by
        :func:`~waterdisagg.rates.fixture_thresholds` or
        :func:`fit_thresholds`. Derived from the configuration when omitted.
    valid:
        Sample-validity mask. Events are not permitted to span a stretch of
        missing data, since flow there is unknown.
    aggregate:
        Observed aggregate flow. When supplied, each event's volume is
        recomputed from the predicted flow but additionally reported as a
        share of the measured total, which is useful for auditing.
    """
    if thresholds is None:
        from .rates import RateFit, fixture_thresholds

        configured = {
            f["code"]: float(f["rate_gpm"])
            for f in config["fixtures"]
            if f.get("rate_gpm") is not None
        }
        thresholds = fixture_thresholds(
            RateFit(configured, {}, {}, {}), config
        )

    period = config["meter"]["sample_period_s"]
    events: list[Event] = []

    for code in predictions.columns:
        params = thresholds.get(code)
        if params is None:
            log.warning("no thresholds for %s; skipping", code)
            continue
        events.extend(
            segment(
                predictions[code],
                on_threshold=params["on_threshold"],
                off_threshold=params["off_threshold"],
                min_duration_s=params["min_duration_s"],
                bridge_gap_s=params["bridge_gap_s"],
                sample_period_s=period,
                fixture=code,
                valid=valid,
            )
        )

    frame = events_to_frame(events)
    if frame.empty:
        log.warning("no events assembled from predictions")
        return frame

    frame["date"] = frame["start"].dt.normalize()
    frame["time_of_day_h"] = (
        frame["start"].dt.hour
        + frame["start"].dt.minute / 60.0
        + frame["start"].dt.second / 3600.0
    )
    frame["day_type"] = np.where(
        frame["start"].dt.dayofweek >= 5, "weekend", "weekday"
    )
    frame["duration_min"] = frame["duration_s"] / 60.0

    if aggregate is not None:
        shares = []
        for _, row in frame.iterrows():
            window = aggregate.loc[row["start"]: row["end"]]
            measured = float(window.clip(lower=0).sum() * period / 60.0)
            shares.append(
                row["volume_gal"] / measured if measured > 0 else np.nan
            )
        frame["share_of_measured"] = shares

    log.info(
        "assembled %d events across %d fixtures",
        len(frame), frame["fixture"].nunique(),
    )
    return frame


def merge_close_events(
    events: pd.DataFrame, min_gap_s: float, *, by: str = "fixture"
) -> pd.DataFrame:
    """Merge same-fixture events separated by less than ``min_gap_s``.

    Provided separately from assembly because it addresses an interpretive
    question rather than a signal-processing one. Two uses of a fixture a few
    minutes apart may be two people in succession or one person pausing, and
    the distinction matters when short inter-event intervals are treated as
    evidence about the number of users. Running the analysis with and without
    a merge threshold shows how far a conclusion depends on that reading.

    A threshold of zero returns the input unchanged.
    """
    if events.empty or min_gap_s <= 0:
        return events

    out = []
    for code, group in events.groupby(by, sort=False):
        group = group.sort_values("start")
        current = None
        for _, row in group.iterrows():
            if current is None:
                current = row.copy()
                continue
            gap = (row["start"] - current["end"]).total_seconds()
            if gap < min_gap_s:
                # Combine: the merged event spans both, and volumes add.
                current["end"] = row["end"]
                current["duration_s"] = (
                    current["end"] - current["start"]
                ).total_seconds()
                current["duration_min"] = current["duration_s"] / 60.0
                current["volume_gal"] = (
                    current["volume_gal"] + row["volume_gal"]
                )
                current["peak_gpm"] = max(
                    current["peak_gpm"], row["peak_gpm"]
                )
            else:
                out.append(current)
                current = row.copy()
        if current is not None:
            out.append(current)

    merged = pd.DataFrame(out).sort_values("start").reset_index(drop=True)
    log.info(
        "merged events with gaps under %.0f s: %d -> %d",
        min_gap_s, len(events), len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------


def match_events(
    predicted: pd.DataFrame,
    labelled: pd.DataFrame,
    *,
    min_iou: float = 0.5,
    per_fixture: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Match predicted events to labelled events by temporal overlap.

    A predicted event counts as correct when it overlaps a labelled event of
    the same fixture by at least ``min_iou`` in intersection-over-union
    terms. Requiring overlap rather than exact boundary agreement is
    appropriate given that labelled boundaries themselves derive from a
    hand-recorded diary.

    Parameters
    ----------
    per_fixture:
        When false, matching ignores fixture identity, which measures
        detection independently of attribution. Comparing the two settings
        separates "did it find the event" from "did it name the right
        fixture" -- a distinction that matters when fixtures of the same
        class flow at similar rates.

    Returns
    -------
    pairs:
        One row per matched pair with the achieved IoU.
    counts:
        True positives, false positives and false negatives, overall and per
        fixture.
    """
    if predicted.empty or labelled.empty:
        return pd.DataFrame(), {
            "tp": 0,
            "fp": len(predicted),
            "fn": len(labelled),
            "per_fixture": {},
        }

    pred = predicted.sort_values("start").reset_index(drop=True)
    true = labelled.sort_values("start").reset_index(drop=True)

    p_start = pred["start"].to_numpy().astype("datetime64[s]").astype(np.int64)
    p_end = pred["end"].to_numpy().astype("datetime64[s]").astype(np.int64)
    t_start = true["start"].to_numpy().astype("datetime64[s]").astype(np.int64)
    t_end = true["end"].to_numpy().astype("datetime64[s]").astype(np.int64)

    intersection = np.minimum(p_end[:, None], t_end[None, :]) - np.maximum(
        p_start[:, None], t_start[None, :]
    )
    intersection = np.clip(intersection, 0, None)
    union = (
        (p_end - p_start)[:, None] + (t_end - t_start)[None, :] - intersection
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)

    if per_fixture:
        same = (
            pred["fixture"].to_numpy()[:, None]
            == true["fixture"].to_numpy()[None, :]
        )
        iou = np.where(same, iou, 0.0)

    iou[iou < min_iou] = 0.0

    # Greedy matching on descending IoU. Optimal assignment would differ only
    # in pathological cases, since events of one fixture rarely overlap.
    pairs = []
    used_pred, used_true = set(), set()
    order = np.argsort(iou.ravel())[::-1]
    for flat in order:
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] <= 0:
            break
        if i in used_pred or j in used_true:
            continue
        used_pred.add(int(i))
        used_true.add(int(j))
        pairs.append(
            {
                "fixture": true.loc[j, "fixture"],
                "predicted_fixture": pred.loc[i, "fixture"],
                "predicted_start": pred.loc[i, "start"],
                "labelled_start": true.loc[j, "start"],
                "iou": float(iou[i, j]),
                "predicted_volume_gal": pred.loc[i, "volume_gal"],
                "labelled_volume_gal": true.loc[j, "volume_gal"],
            }
        )

    counts = {
        "tp": len(pairs),
        "fp": len(pred) - len(used_pred),
        "fn": len(true) - len(used_true),
        "per_fixture": {},
    }
    for code in sorted(set(true["fixture"]) | set(pred["fixture"])):
        matched = sum(1 for p in pairs if p["fixture"] == code)
        counts["per_fixture"][code] = {
            "tp": matched,
            "fp": int((pred["fixture"] == code).sum())
            - sum(1 for p in pairs if p["predicted_fixture"] == code),
            "fn": int((true["fixture"] == code).sum()) - matched,
            "n_labelled": int((true["fixture"] == code).sum()),
        }

    return pd.DataFrame(pairs), counts


# ---------------------------------------------------------------------------
# Threshold fitting
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ThresholdFit:
    """Fitted per-fixture thresholds and the scores behind them."""

    thresholds: dict[str, dict]
    scores: dict[str, float]
    searched: dict[str, pd.DataFrame]
    fallback: dict[str, str] = dataclasses.field(default_factory=dict)

    def report(self) -> pd.DataFrame:
        rows = []
        for code, params in sorted(self.thresholds.items()):
            rows.append(
                {
                    "fixture": code,
                    "rate_gpm": round(params.get("rate_gpm", float("nan")), 3),
                    "on_threshold": round(params["on_threshold"], 3),
                    "on_frac_of_rate": round(
                        params["on_threshold"]
                        / max(params.get("rate_gpm", np.nan), 1e-9),
                        3,
                    ),
                    "f1": round(self.scores.get(code, float("nan")), 3),
                    "note": self.fallback.get(code, ""),
                }
            )
        return pd.DataFrame(rows)


def fit_thresholds(
    predictions: pd.DataFrame,
    labelled_events: pd.DataFrame,
    config: dict,
    rate_fit,
    *,
    fractions: Sequence[float] = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6),
    min_iou: float = 0.5,
    valid: pd.Series | None = None,
) -> ThresholdFit:
    """Select each fixture's threshold by maximising validation event $F_1$.

    Fitting is per fixture and independent, which is a simplification --
    fixtures interact through the allocation, so the joint optimum may differ
    slightly. In practice the objective is flat near its maximum and the
    independent choice is adequate; the full search is retained in
    ``searched`` so the flatness can be inspected.

    Thresholds are searched as fractions of each fixture's fitted rate, so
    the resulting configuration transfers to a household whose fixtures
    deliver different flows.
    """
    defaults = config["segmentation"]
    by_code = {f["code"]: f for f in config["fixtures"]}

    thresholds: dict[str, dict] = {}
    scores: dict[str, float] = {}
    searched: dict[str, pd.DataFrame] = {}
    fallback: dict[str, str] = {}

    for code in predictions.columns:
        rate = rate_fit.rates.get(code)
        if rate is None or not np.isfinite(rate):
            fallback[code] = "no fitted rate; threshold not searched"
            continue

        local = (by_code.get(code, {}) or {}).get("segmentation", {}) or {}
        min_duration = local.get("min_duration_s", defaults["min_duration_s"])
        bridge = local.get("bridge_gap_s", defaults["bridge_gap_s"])
        off_ratio = (
            local.get("off_threshold_frac", defaults["off_threshold_frac"])
            / local.get("on_threshold_frac", defaults["on_threshold_frac"])
        )

        truth = labelled_events[labelled_events["fixture"] == code]
        rows = []
        best = (-1.0, None)

        for fraction in fractions:
            on = rate * fraction
            candidates = segment(
                predictions[code],
                on_threshold=on,
                off_threshold=on * off_ratio,
                min_duration_s=min_duration,
                bridge_gap_s=bridge,
                sample_period_s=config["meter"]["sample_period_s"],
                fixture=code,
                valid=valid,
            )
            found = events_to_frame(candidates)
            if truth.empty:
                # Nothing to score against; prefer the configured default.
                rows.append(
                    {"fraction": fraction, "n_predicted": len(found),
                     "precision": np.nan, "recall": np.nan, "f1": np.nan}
                )
                continue

            _, counts = match_events(found, truth, min_iou=min_iou)
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            rows.append(
                {
                    "fraction": fraction,
                    "n_predicted": len(found),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )
            if f1 > best[0]:
                best = (f1, fraction)

        searched[code] = pd.DataFrame(rows)

        if best[1] is None:
            fraction = local.get(
                "on_threshold_frac", defaults["on_threshold_frac"]
            )
            fallback[code] = (
                "no labelled validation events; using configured fraction"
            )
        else:
            fraction = best[1]
            scores[code] = best[0]

        thresholds[code] = {
            "rate_gpm": rate,
            "on_threshold": rate * fraction,
            "off_threshold": rate * fraction * off_ratio,
            "min_duration_s": min_duration,
            "bridge_gap_s": bridge,
        }

    fit = ThresholdFit(thresholds, scores, searched, fallback)
    log.info("fitted thresholds:\n%s", fit.report().to_string(index=False))
    return fit
