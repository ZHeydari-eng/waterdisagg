"""Evaluation of disaggregation performance.

Three families of measure are reported, and the order reflects their
importance for this work.

**Event-level agreement** -- precision, recall and $F_1$ over matched events
-- comes first, because every downstream analysis operates on events. A model
with respectable per-sample error can still produce an event table that
fragments showers or invents flushes, and per-sample error will not reveal it.

**Volume error** per fixture per day speaks to the end-use apportionment
claims. It is reported in gallons rather than as a percentage so that a large
relative error on a fixture that uses very little water is not mistaken for a
serious one.

**Per-sample error** comes last. It is the quantity the model was trained on,
so it is useful for diagnosis, but it is a poor summary: the household is
idle most of the time, and an average over all samples is dominated by
periods where there is nothing to disaggregate.

Alongside every score, the number of held-out events is reported. A per-fixture
$F_1$ computed from two events is not a measurement, and presenting it without
its support invites over-reading.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .events import match_events

log = logging.getLogger(__name__)

__all__ = [
    "EvaluationResult",
    "evaluate",
    "event_metrics",
    "volume_metrics",
    "sample_metrics",
    "confusion_by_fixture",
    "to_latex_table",
]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvaluationResult:
    """All computed metrics, per fixture and aggregated by class."""

    per_fixture: pd.DataFrame
    by_class: pd.DataFrame
    overall: dict
    confusion: pd.DataFrame | None = None
    detection_only: dict | None = None

    def report(self) -> str:
        lines = [
            "Per-fixture performance on held-out days",
            self.per_fixture.to_string(index=False),
            "",
            "By end-use class",
            self.by_class.to_string(index=False),
            "",
            "Overall: " + ", ".join(
                f"{k}={v}" for k, v in self.overall.items()
            ),
        ]
        if self.detection_only:
            lines += [
                "",
                "Detection ignoring fixture identity: "
                + ", ".join(f"{k}={v}" for k, v in self.detection_only.items()),
            ]
        return "\n".join(lines)

    def save(self, directory: Path | str) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, frame in (
            ("metrics_per_fixture.csv", self.per_fixture),
            ("metrics_by_class.csv", self.by_class),
        ):
            path = directory / name
            frame.to_csv(path, index=False)
            written[name] = path
        if self.confusion is not None:
            path = directory / "confusion.csv"
            self.confusion.to_csv(path)
            written["confusion.csv"] = path
        return written


# ---------------------------------------------------------------------------
# Component metrics
# ---------------------------------------------------------------------------


def event_metrics(
    predicted: pd.DataFrame,
    labelled: pd.DataFrame,
    *,
    min_iou: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Precision, recall and $F_1$ over matched events, per fixture.

    An event counts as correct when it overlaps a labelled event of the same
    fixture by at least ``min_iou``. Overlap rather than exact agreement is
    the right criterion here, since the labelled boundaries themselves come
    from a hand-recorded diary and carry their own uncertainty.
    """
    pairs, counts = match_events(predicted, labelled, min_iou=min_iou)

    rows = []
    for code, stats in counts["per_fixture"].items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        if precision and recall and np.isfinite(precision + recall):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if (tp + fp + fn) else np.nan
        matched = pairs[pairs["fixture"] == code] if not pairs.empty else pairs
        rows.append(
            {
                "fixture": code,
                "n_labelled_events": stats["n_labelled"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "median_iou": (
                    float(matched["iou"].median()) if len(matched) else np.nan
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values("fixture").reset_index(drop=True)

    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    overall = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
            4,
        ),
    }
    return frame, overall


def volume_metrics(
    predicted_flow: pd.DataFrame,
    labelled_flow: pd.DataFrame,
    *,
    sample_period_s: int = 1,
    days: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Mean absolute error in daily volume, per fixture.

    Daily rather than total, because a total can hide compensating errors --
    a fixture over-attributed on one day and under-attributed on another may
    show almost no error in aggregate while getting both days wrong.
    """
    common = predicted_flow.index.intersection(labelled_flow.index)
    predicted = predicted_flow.loc[common]
    labelled = labelled_flow.loc[common]

    if days is not None:
        wanted = {pd.Timestamp(d).normalize() for d in days}
        keep = pd.Index([t for t in common if t.normalize() in wanted])
        predicted, labelled = predicted.loc[keep], labelled.loc[keep]

    if predicted.empty:
        return pd.DataFrame(columns=["fixture", "volume_mae_gal_day",
                                     "volume_bias_gal_day", "n_days"])

    scale = sample_period_s / 60.0
    predicted_daily = predicted.groupby(predicted.index.normalize()).sum() * scale
    labelled_daily = labelled.groupby(labelled.index.normalize()).sum() * scale

    difference = predicted_daily - labelled_daily
    return pd.DataFrame(
        {
            "fixture": difference.columns,
            "volume_mae_gal_day": difference.abs().mean().to_numpy(),
            "volume_bias_gal_day": difference.mean().to_numpy(),
            "n_days": len(difference),
        }
    ).reset_index(drop=True)


def sample_metrics(
    predicted_flow: pd.DataFrame,
    labelled_flow: pd.DataFrame,
    *,
    noise_floor: float = 0.05,
    valid: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-sample error, restricted to samples where flow is present.

    Averaging over all samples would be dominated by idle periods, where
    predicting zero is both correct and uninformative. The restriction makes
    the figure comparable across fixtures of very different frequency.
    """
    common = predicted_flow.index.intersection(labelled_flow.index)
    predicted = predicted_flow.loc[common]
    labelled = labelled_flow.loc[common]

    aggregate = labelled.sum(axis=1)
    mask = aggregate > noise_floor
    if valid is not None:
        mask &= valid.reindex(common).fillna(False)

    rows = []
    for code in labelled.columns:
        if code not in predicted.columns:
            continue
        error = (predicted[code] - labelled[code])[mask]
        active = labelled[code][mask] > noise_floor
        rows.append(
            {
                "fixture": code,
                "mae_gpm_active_household": (
                    float(error.abs().mean()) if len(error) else np.nan
                ),
                "mae_gpm_when_running": (
                    float(error[active].abs().mean())
                    if active.any() else np.nan
                ),
                "n_samples_running": int(active.sum()),
            }
        )
    return pd.DataFrame(rows)


def confusion_by_fixture(
    predicted_flow: pd.DataFrame,
    labelled_flow: pd.DataFrame,
    *,
    noise_floor: float = 0.05,
) -> pd.DataFrame:
    """Where attributed flow actually went, at timesteps with a single label.

    Restricted to unambiguous timesteps -- exactly one fixture labelled
    active -- because a confusion matrix has no clear meaning when two
    fixtures genuinely run at once. Rows are normalised, so each gives the
    distribution of predicted attribution for one true fixture.

    This is the table that reveals same-class confusion: two toilets with
    similar flow rates appear as mass off the diagonal between them, which an
    averaged $F_1$ would obscure.
    """
    common = predicted_flow.index.intersection(labelled_flow.index)
    predicted = predicted_flow.loc[common]
    labelled = labelled_flow.loc[common]

    active = (labelled > noise_floor)
    single = active.sum(axis=1) == 1
    if not single.any():
        return pd.DataFrame()

    codes = list(labelled.columns)
    true_index = active[single].to_numpy().argmax(axis=1)
    predicted_index = predicted[single].to_numpy().argmax(axis=1)

    matrix = np.zeros((len(codes), len(codes)), dtype=float)
    for t, p in zip(true_index, predicted_index):
        matrix[t, p] += 1
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0

    return pd.DataFrame(
        matrix / totals,
        index=pd.Index(codes, name="true"),
        columns=pd.Index(codes, name="predicted"),
    )


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate(
    predicted_events: pd.DataFrame,
    labelled_events: pd.DataFrame,
    predicted_flow: pd.DataFrame,
    labelled_flow: pd.DataFrame,
    config: dict,
    *,
    days: Sequence[pd.Timestamp] | None = None,
    valid: pd.Series | None = None,
    min_iou: float = 0.5,
) -> EvaluationResult:
    """Compute all metrics and assemble them into one result.

    Parameters
    ----------
    days:
        Restrict evaluation to these days -- normally the test partition.
        Passing them explicitly, rather than inferring from the extent of the
        prediction, avoids accidentally scoring on days used for training.
    """
    floor = config["meter"]["noise_floor_gpm"]
    period = config["meter"]["sample_period_s"]

    if days is not None:
        wanted = {pd.Timestamp(d).normalize() for d in days}
        if not predicted_events.empty:
            predicted_events = predicted_events[
                predicted_events["start"].dt.normalize().isin(wanted)
            ]
        if not labelled_events.empty:
            labelled_events = labelled_events[
                labelled_events["start"].dt.normalize().isin(wanted)
            ]

    events, overall = event_metrics(
        predicted_events, labelled_events, min_iou=min_iou
    )
    volumes = volume_metrics(
        predicted_flow, labelled_flow, sample_period_s=period, days=days
    )
    samples = sample_metrics(
        predicted_flow, labelled_flow, noise_floor=floor, valid=valid
    )

    per_fixture = (
        events.merge(volumes, on="fixture", how="outer")
        .merge(samples, on="fixture", how="outer")
    )

    # Label and class come from the configuration, so the table reads in the
    # paper's terms rather than in internal codes.
    meta = {
        f["code"]: (f.get("label", f["code"]), f.get("category", "other"))
        for f in config["fixtures"]
    }
    per_fixture["label"] = per_fixture["fixture"].map(
        lambda c: meta.get(c, (c, "other"))[0]
    )
    per_fixture["category"] = per_fixture["fixture"].map(
        lambda c: meta.get(c, (c, "other"))[1]
    )

    threshold = config["split"].get("min_test_events_warn", 3)
    per_fixture["sufficient_support"] = (
        per_fixture["n_labelled_events"].fillna(0) >= threshold
    )

    by_class = _aggregate_by_class(
        predicted_events, labelled_events, config, min_iou=min_iou
    )

    # Detection independent of attribution: did the model find the event at
    # all, regardless of which fixture it named? The gap against `overall`
    # isolates confusion between fixtures from failure to detect.
    _, loose = match_events(
        predicted_events, labelled_events, min_iou=min_iou, per_fixture=False
    )
    tp, fp, fn = loose["tp"], loose["fp"], loose["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    detection = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
            4,
        ),
    }

    confusion = confusion_by_fixture(
        predicted_flow, labelled_flow, noise_floor=floor
    )

    result = EvaluationResult(
        per_fixture=per_fixture,
        by_class=by_class,
        overall=overall,
        confusion=confusion if not confusion.empty else None,
        detection_only=detection,
    )

    thin = per_fixture.loc[~per_fixture["sufficient_support"], "fixture"]
    if len(thin):
        log.warning(
            "fixtures with fewer than %d held-out events: %s -- scores "
            "reported but not interpretable",
            threshold, list(thin),
        )
    return result


def _aggregate_by_class(
    predicted_events: pd.DataFrame,
    labelled_events: pd.DataFrame,
    config: dict,
    *,
    min_iou: float = 0.5,
) -> pd.DataFrame:
    """Recompute metrics with fixtures collapsed to their end-use class.

    Not an average of the per-fixture scores. Events are relabelled by class
    and rematched, so a shower attributed to the wrong shower counts as
    correct at this level. The gap between the two tables is precisely the
    cost of fixture-level ambition, which is the quantity a reader wants when
    judging whether appliance-level claims are supported.
    """
    classes = {f["code"]: f.get("category", "other") for f in config["fixtures"]}
    if predicted_events.empty and labelled_events.empty:
        return pd.DataFrame()

    predicted = predicted_events.copy()
    labelled = labelled_events.copy()
    for frame in (predicted, labelled):
        if not frame.empty:
            frame["fixture"] = frame["fixture"].map(
                lambda c: classes.get(c, "other")
            )

    frame, _ = event_metrics(predicted, labelled, min_iou=min_iou)
    return frame.rename(columns={"fixture": "class"})


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def _escape_latex(text: str) -> str:
    """Escape characters LaTeX treats specially.

    Fixture labels routinely contain ``#`` (as in "Shower #1"), which begins
    a macro parameter in LaTeX and produces a compile error if passed
    through unescaped.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for character, replacement in replacements.items():
        text = text.replace(character, replacement)
    return text


def to_latex_table(
    result: EvaluationResult,
    *,
    caption: str | None = None,
    label: str = "tab:performance",
    include_classes: bool = True,
    order: Sequence[str] | None = None,
) -> str:
    """Render the performance table for the supporting information.

    Produces the table described in the revised SI: held-out event count,
    precision, recall, $F_1$ and daily volume error per fixture, with class
    level rows appended. Fixtures whose support falls below the configured
    threshold are marked, so the reader is not left to infer that an $F_1$ of
    1.000 rests on a single event.
    """
    frame = result.per_fixture.copy()
    if order is not None:
        rank = {code: i for i, code in enumerate(order)}
        frame["_rank"] = frame["fixture"].map(
            lambda c: rank.get(c, len(rank))
        )
        frame = frame.sort_values(["_rank", "fixture"])
    else:
        category_rank = {"shower": 0, "toilet": 1, "faucet": 2, "appliance": 3}
        frame["_rank"] = frame["category"].map(
            lambda c: category_rank.get(c, 9)
        )
        frame = frame.sort_values(["_rank", "fixture"])

    def fmt(value, digits=3):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "---"
        return f"{value:.{digits}f}"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
    ]
    if caption is None:
        caption = (
            "Disaggregation performance on held-out days. Events are matched "
            "to labelled events of the same fixture at a minimum temporal "
            "overlap of 50\\%. Daily volume error is the mean absolute "
            "difference between predicted and labelled daily volume. "
            "Fixtures marked $\\dagger$ have too few held-out events for "
            "their scores to be interpretable."
        )
    lines += [
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Fixture/Appliance & Held-out events & Precision & Recall & $F_1$ "
        r"& Volume MAE (gal/day) \\",
        r"\hline",
    ]

    for _, row in frame.iterrows():
        n = row.get("n_labelled_events")
        n_text = "0" if pd.isna(n) else f"{int(n)}"
        marker = "" if row.get("sufficient_support", False) else r"$\dagger$"
        lines.append(
            f"{_escape_latex(str(row['label']))}{marker} & {n_text} & "
            f"{fmt(row.get('precision'))} & {fmt(row.get('recall'))} & "
            f"{fmt(row.get('f1'))} & "
            f"{fmt(row.get('volume_mae_gal_day'), 2)} \\\\"
        )

    if include_classes and not result.by_class.empty:
        lines.append(r"\hline")
        for _, row in result.by_class.iterrows():
            name = _escape_latex(str(row["class"]).capitalize())
            lines.append(
                f"{name} class (all) & {int(row['n_labelled_events'])} & "
                f"{fmt(row['precision'])} & {fmt(row['recall'])} & "
                f"{fmt(row['f1'])} & --- \\\\"
            )

    lines += [
        r"\hline",
        r"\end{tabular}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
