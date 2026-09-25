"""
===============================================================================
 standalone.py -- the whole method in one file, no installation required
===============================================================================

A self-contained implementation of the disaggregation method: paste it into a
Colab notebook or run it directly. It depends only on numpy, pandas and
torch, and imports nothing from the `waterdisagg` package.

    python standalone.py              # runs on a simulated household
    python standalone.py --trace data/trace.csv --labels data/labels.csv

WHY THIS FILE EXISTS

The `waterdisagg` package is the maintained implementation: it is tested,
configurable, and handles the awkward cases that arise with real data. This
file is for reading and for quick experimentation. It condenses the same
method into one place so it can be followed end to end without navigating a
package, and so it can be run somewhere with no setup.

Being a condensation, it omits things the package does: reading meter files
whose records span several rows, matching a hand-written diary to the trace,
per-fixture event-assembly rules, and the full evaluation suite. Where the
two differ, the package is authoritative.

WHAT THE METHOD IS

A single meter on the main supply records total household flow once per
second. The task is to divide that one number, at each second, among the
fixtures that produced it.

    aggregate flow  ->  per-fixture flow  ->  events  ->  behaviour
       (1 number)        (14 numbers)        (start,      (stagnation
                                              end,          time)
                                              volume)

The model reads a 60-second window and, for each second in it, predicts how
the measured flow divides across fixtures. Two ideas make this work better
than regressing each fixture independently:

  1. The model predicts a *share* of the observed flow rather than an
     absolute rate. Shares are non-negative and sum to one, so predicted
     flows are automatically non-negative and add up to what the meter
     actually measured.

  2. There is one extra share for flow belonging to no known fixture.
     Without it, anything unfamiliar -- an unlogged use, a leak -- would be
     forced onto a real fixture, inventing events.

===============================================================================
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

FIXTURES = [
    "Ad_S", "Kid_S", "Dwn_S",          # showers
    "Ad_T", "Kid_T", "Dwn_T",          # toilets
    "Ad_F", "Kid_F", "Dwn_F",          # bathroom faucets
    "Kitch_F", "Kitch_r", "Bsmnt_F",   # kitchen, refrigerator, laundry faucets
    "Dish_W", "Wash_M",                # appliances
]

WINDOW = 60        # seconds the model reads at once
STRIDE = 15        # how far the window advances between examples
NOISE_FLOOR = 0.05  # gpm; below this the meter is reading noise, not flow

# Minimum event duration and the largest gap bridged within one event, per
# fixture class. These differ by an order of magnitude: a toilet flush is
# continuous, a shower may pause briefly, and a dishwasher draws in several
# fills separated by genuine inactivity.
EVENT_RULES = {
    "shower":    {"min_duration_s": 120, "bridge_gap_s": 45},
    "toilet":    {"min_duration_s": 10,  "bridge_gap_s": 5},
    "faucet":    {"min_duration_s": 4,   "bridge_gap_s": 3},
    "appliance": {"min_duration_s": 20,  "bridge_gap_s": 420},
}


def fixture_class(code: str) -> str:
    if code.endswith("_S"):
        return "shower"
    if code.endswith("_T"):
        return "toilet"
    if code in ("Dish_W", "Wash_M"):
        return "appliance"
    return "faucet"


# =============================================================================
# 1. SIMULATED HOUSEHOLD
#
# Generates a flow trace and a matching per-fixture truth table, so the whole
# thing runs with no data. Real meter output is reproduced closely enough to
# be a fair test: the signal is smoothed by the instrument, the idle baseline
# is noisy rather than exactly zero, and fixtures sometimes run at once.
# =============================================================================


def simulate(n_days: int = 14, seed: int = 7):
    """Return (flow_series, truth_frame) for a synthetic household."""
    rng = np.random.default_rng(seed)
    n = n_days * 86400
    index = pd.date_range("2024-06-06", periods=n, freq="1s")
    truth = np.zeros((n, len(FIXTURES)), dtype=np.float32)

    # rate (gpm), duration range (s), uses per day, preferred hours
    profile = {
        "Ad_S":    (2.26, (300, 1200), 1.1, [(7, 1.2), (21, 1.5)]),
        "Kid_S":   (2.44, (240, 1100), 1.0, [(7.5, 1.3), (21, 1.8)]),
        "Dwn_S":   (2.58, (300, 900), 0.05, [(9, 3)]),
        "Ad_T":    (2.84, (30, 70), 4.5, [(9, 3), (7, 1.2)]),
        "Kid_T":   (2.99, (30, 70), 3.0, [(9, 3), (18, 3)]),
        "Dwn_T":   (1.24, (45, 90), 3.5, [(9, 3), (18, 3)]),
        "Ad_F":    (1.41, (5, 60), 4.0, [(7.5, 1.3), (21, 1.8)]),
        "Kid_F":   (0.90, (5, 50), 3.5, [(7.5, 1.3), (21, 1.8)]),
        "Dwn_F":   (0.54, (4, 40), 3.5, [(9, 3), (18, 3)]),
        "Kitch_F": (1.20, (4, 180), 20.0, [(9, 3), (18, 3)]),
        "Kitch_r": (0.40, (3, 20), 5.0, [(9, 3), (18, 3)]),
        "Bsmnt_F": (1.55, (10, 120), 0.2, [(9, 3)]),
        "Dish_W":  (1.24, (60, 120), 0.8, [(21, 1.5)]),
        "Wash_M":  (2.99, (90, 150), 1.0, [(9, 3), (18, 3)]),
    }

    for col, code in enumerate(FIXTURES):
        rate, (lo, hi), per_day, hours = profile[code]
        for day in range(n_days):
            for _ in range(rng.poisson(per_day)):
                centre, spread = hours[rng.integers(len(hours))]
                hour = float(np.clip(rng.normal(centre, spread), 0, 23.99))
                start = day * 86400 + int(hour * 3600) + int(rng.integers(60))
                duration = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                end = min(start + duration, n)
                if start < n:
                    truth[start:end, col] += rate * (1 + rng.normal(0, 0.03))

    # The meter smooths the signal heavily: a valve that opens instantly
    # produces a reading that takes ten seconds or more to reach its plateau.
    # Four cascaded first-order lags reproduce that S-shaped ramp.
    flow = truth.sum(axis=1).astype(float)
    alpha = 1.0 / (1.5 + 1.0)
    for _ in range(4):
        smoothed = np.empty_like(flow)
        state = flow[0]
        for i, value in enumerate(flow):
            state += alpha * (value - state)
            smoothed[i] = state
        flow = smoothed

    # Idle flow oscillates around zero, including negative excursions.
    flow = flow + rng.normal(0, 0.012, n)

    return (
        pd.Series(flow.astype(np.float32), index=index, name="flow_gpm"),
        pd.DataFrame(truth, index=index, columns=FIXTURES),
    )


# =============================================================================
# 2. WINDOWING AND THE DAY-BASED SPLIT
#
# THIS IS THE STEP MOST EASILY GOT WRONG, AND GETTING IT WRONG DOES NOT RAISE
# AN ERROR -- IT PRODUCES BETTER-LOOKING NUMBERS.
#
# Windows are taken every 15 seconds but are 60 seconds long, so neighbouring
# windows share three quarters of their samples. They are near-duplicates.
#
# Split those windows at random into training and test sets and near-copies
# land on both sides. The model is then tested on material it has effectively
# already seen, and the reported accuracy measures memorisation.
#
# So the split is over whole CALENDAR DAYS, and windows are cut afterwards,
# separately for each set. A window cannot belong to a set its day does not,
# because the function that made it only saw that day. Windows crossing
# midnight are dropped, since they would belong to both.
# =============================================================================


def make_windows(flow: pd.Series, truth: pd.DataFrame, days: set):
    """Cut windows that lie entirely within the given days and contain flow."""
    values = flow.to_numpy(dtype=np.float32)
    targets = truth.to_numpy(dtype=np.float32)
    day_of = flow.index.normalize()
    unique_days = np.array(sorted(set(day_of)))
    code_of = np.searchsorted(unique_days, day_of.to_numpy())
    permitted = np.array([d in days for d in unique_days])

    active = (targets > NOISE_FLOOR).any(axis=1).astype(np.int32)
    running = np.concatenate([[0], np.cumsum(active)])

    starts = np.arange(0, len(values) - WINDOW + 1, STRIDE)
    first, last = code_of[starts], code_of[starts + WINDOW - 1]
    same_day = first == last                      # does not cross midnight
    keep = np.zeros(len(starts), dtype=bool)
    keep[same_day] = permitted[first[same_day]]
    keep &= (running[starts + WINDOW] - running[starts]) > 0   # has activity
    starts = starts[keep]

    if starts.size == 0:
        return None
    rows = starts[:, None] + np.arange(WINDOW)[None, :]
    return (
        torch.from_numpy(values[rows]).unsqueeze(-1),
        torch.from_numpy(targets[rows]),
    )


def split_days(flow: pd.Series, test_frac: float = 0.2, val_frac: float = 0.15,
               seed: int = 0):
    """Assign whole days to train, validation and test.

    Weekdays and weekend days are drawn separately, so that a small held-out
    set is not accidentally all weekdays -- weekday/weekend differences are
    among the things being studied.
    """
    days = sorted(set(flow.index.normalize()))
    rng = np.random.default_rng(seed)
    weekday = [d for d in days if d.dayofweek < 5]
    weekend = [d for d in days if d.dayofweek >= 5]

    def take(pool, k):
        k = min(k, len(pool))
        return [] if k == 0 else list(rng.choice(pool, k, replace=False))

    n_test, n_val = max(1, int(len(days) * test_frac)), max(1, int(len(days) * val_frac))
    share = len(weekend) / max(len(days), 1)
    test = take(weekday, round(n_test * (1 - share))) + take(weekend, round(n_test * share))
    remaining_wd = [d for d in weekday if d not in test]
    remaining_we = [d for d in weekend if d not in test]
    val = take(remaining_wd, round(n_val * (1 - share))) + take(remaining_we, round(n_val * share))
    train = [d for d in days if d not in set(test) | set(val)]
    return set(train), set(val), set(test)


# =============================================================================
# 3. THE MODEL
#
# Two stacked LSTM layers read the window, then a linear layer produces one
# number per fixture plus one spare. A softmax turns those into shares that
# sum to one, and multiplying by the measured flow gives predicted rates.
#
# Because the shares are non-negative and sum to one, the predictions are
# guaranteed to be non-negative and to add up to the meter reading. Neither
# has to be learned.
#
# The spare channel holds flow belonging to no listed fixture. Real records
# contain uses nobody wrote down, and a leak is by definition something the
# model never trained on. Without a place to put such flow, it would have to
# be blamed on a real fixture.
# =============================================================================


class Disaggregator(nn.Module):
    def __init__(self, n_fixtures: int, hidden: int = 128, layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.n_fixtures = n_fixtures
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=0.15 if layers > 1 else 0.0)
        width = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(width, n_fixtures + 1)   # +1 = unattributed

    def forward(self, x):
        # x: (batch, time, 1) flow in gpm. Scaled only for the network's
        # benefit; the multiplication below uses the unscaled value, so the
        # output stays in gpm.
        hidden, _ = self.lstm(x / 5.0)
        shares = F.softmax(self.head(hidden), dim=-1)
        flow = shares[..., :self.n_fixtures] * x
        spare = shares[..., -1:] * x          # flow attributed to nothing
        return flow, spare


def loss_fn(predicted, spare, target, aggregate):
    """Error in predicted flow, ignoring seconds with nothing to divide.

    Idle seconds are excluded: dividing zero flow among fixtures has no right
    answer, and since the household is idle most of the time, including them
    would let the model score well by predicting nothing.

    Absolute error rather than squared: squared error is dominated by the
    highest-flow fixtures and by the middles of long events, and reacts badly
    to the small timing errors a hand-written diary inevitably contains.

    The spare channel is penalised. Without this the model learns to put
    everything there: most fixtures are off at any moment, so predicting
    "none of them" is the cheapest way to cut the error early in training,
    and it never recovers. The penalty makes the spare channel a last resort
    rather than a default, which is what it is meant to be.
    """
    mask = (aggregate.squeeze(-1) > NOISE_FLOOR).unsqueeze(-1).float()
    denominator = mask.sum().clamp_min(1.0)
    error = F.smooth_l1_loss(predicted, target, beta=0.25, reduction="none")
    flow_term = (error * mask).sum() / denominator / predicted.shape[-1]
    spare_term = (spare * mask).sum() / denominator
    return flow_term + 0.3 * spare_term


# =============================================================================
# 4. PREDICTION ACROSS A LONG RECORD
#
# Windows overlap, so each second sits inside four of them and gets four
# predictions. Averaging blurs the edges of events: a window that only just
# catches an event's start sees a partial signal and predicts accordingly.
#
# Instead each window contributes only its middle slice, and those slices tile
# the record exactly. Every second is predicted by the window that has the
# most context on both sides of it.
# =============================================================================


@torch.no_grad()
def predict(model, flow: pd.Series, batch_size: int = 256):
    model.eval()
    values = flow.to_numpy(dtype=np.float32)
    n = len(values)
    output = np.zeros((n, len(FIXTURES)), dtype=np.float32)
    offset = (WINDOW - STRIDE) // 2      # keep the middle STRIDE seconds

    starts = np.arange(0, n - WINDOW + 1, STRIDE)
    # Skip windows with no flow: the prediction is zero either way, and the
    # household is idle for most of the record.
    active = (np.abs(values) > NOISE_FLOOR).astype(np.int32)
    running = np.concatenate([[0], np.cumsum(active)])
    starts = starts[(running[starts + WINDOW] - running[starts]) > 0]

    for begin in range(0, len(starts), batch_size):
        chunk = starts[begin:begin + batch_size]
        rows = chunk[:, None] + np.arange(WINDOW)[None, :]
        batch = torch.from_numpy(values[rows]).unsqueeze(-1)
        predicted, _ = model(batch)
        predicted = predicted.numpy()
        for i, start in enumerate(chunk):
            a = start + offset
            b = min(a + STRIDE, n)
            output[a:b] = predicted[i, offset:offset + (b - a)]

    return pd.DataFrame(output, index=flow.index, columns=FIXTURES)


# =============================================================================
# 5. EVENTS
#
# Every behavioural result rests on events, not on individual seconds. Three
# mechanisms turn a per-second signal into a sensible event list:
#
#   Hysteresis    -- a single threshold on a signal hovering near it chops one
#                    event into fragments. Turning on at a higher level than
#                    turning off prevents that.
#   Gap bridging  -- brief interruptions happen within one use, and appliances
#                    draw in several fills per cycle.
#   Minimum length -- very short blips are usually misattribution, not use.
# =============================================================================


def extract_events(predictions: pd.DataFrame, rates: dict) -> pd.DataFrame:
    rows = []
    for code in predictions.columns:
        rule = EVENT_RULES[fixture_class(code)]
        series = predictions[code].to_numpy()
        on, off = 0.5 * rates[code], 0.25 * rates[code]

        state = np.zeros(len(series), dtype=bool)
        running = False
        for i, value in enumerate(series):
            if not running and value > on:
                running = True
            elif running and value < off:
                running = False
            state[i] = running

        edges = np.diff(state.astype(np.int8), prepend=0, append=0)
        spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

        merged = []
        for start, end in spans:
            if merged and start - merged[-1][1] <= rule["bridge_gap_s"]:
                merged[-1][1] = end
            else:
                merged.append([start, end])

        for start, end in merged:
            if end - start < rule["min_duration_s"]:
                continue
            segment = np.clip(series[start:end], 0, None)
            rows.append({
                "fixture": code,
                "start": predictions.index[start],
                "end": predictions.index[end - 1] + pd.Timedelta(seconds=1),
                "duration_min": (end - start) / 60.0,
                "volume_gal": segment.sum() / 60.0,
                "time_of_day_h": predictions.index[start].hour
                                 + predictions.index[start].minute / 60.0,
                "day_type": "weekend" if predictions.index[start].dayofweek >= 5
                            else "weekday",
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)


# =============================================================================
# 6. STAGNATION TIME
#
# The paper's central metric: how long water sat still in the pipe serving a
# fixture. Measured from the END of one use to the START of the next, so it
# captures the idle interval itself and does not change with how long the
# previous use lasted.
# =============================================================================


def stagnation_time(events: pd.DataFrame, fixture: str) -> pd.Series:
    if events.empty or "fixture" not in events.columns:
        return pd.Series(dtype="float64")
    subset = events[events["fixture"] == fixture].sort_values("start")
    if len(subset) < 2:
        return pd.Series(dtype="float64")
    gaps = (subset["start"].iloc[1:].to_numpy()
            - subset["end"].iloc[:-1].to_numpy())
    return pd.Series(gaps.astype("timedelta64[s]").astype(float) / 3600.0,
                     name=f"{fixture}_stagnation_h")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", help="CSV with a timestamp and flow column")
    parser.add_argument("--labels", help="CSV with per-second fixture columns")
    parser.add_argument("--days", type=int, default=14, help="days to simulate")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- data ---------------------------------------------------------------
    if args.trace and args.labels:
        print(f"Reading {args.trace} and {args.labels}")
        raw = pd.read_csv(args.trace, parse_dates=[0], index_col=0)
        flow = raw.iloc[:, 0].astype(np.float32)
        labels = pd.read_csv(args.labels, parse_dates=[0], index_col=0)
        truth = labels.reindex(columns=FIXTURES).fillna(0.0).astype(np.float32)
        truth = truth.reindex(flow.index).fillna(0.0)
    else:
        print(f"No data given; simulating {args.days} days")
        flow, truth = simulate(args.days, args.seed)

    print(f"{len(flow):,} seconds, {len(set(flow.index.normalize()))} days, "
          f"{flow.clip(lower=0).sum() / 60:,.0f} gallons")

    # -- split and window ---------------------------------------------------
    train_days, val_days, test_days = split_days(flow, seed=args.seed)
    print(f"\nDays: {len(train_days)} train, {len(val_days)} validation, "
          f"{len(test_days)} test")

    train = make_windows(flow, truth, train_days)
    val = make_windows(flow, truth, val_days)
    if train is None:
        sys.exit("No training windows produced; check the labels.")
    print(f"Windows: {len(train[0]):,} train"
          + (f", {len(val[0]):,} validation" if val else ""))

    # -- train --------------------------------------------------------------
    model = Disaggregator(len(FIXTURES), args.hidden, args.layers,
                          args.bidirectional)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train, y_train = train
    print(f"\nTraining {sum(p.numel() for p in model.parameters()):,} parameters")

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for begin in range(0, len(x_train), 64):
            idx = order[begin:begin + 64]
            xb, yb = x_train[idx], y_train[idx]
            predicted, spare = model(xb)
            loss = loss_fn(predicted, spare, yb, xb)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss)

        if epoch % 5 == 0 or epoch == 1:
            note = ""
            if val:
                model.eval()
                with torch.no_grad():
                    vp, vs = model(val[0])
                    note = f"  validation {float(loss_fn(vp, vs, val[1], val[0])):.5f}"
            print(f"  epoch {epoch:3d}  training {total / max(len(x_train) // 64, 1):.5f}{note}")

    # -- predict and extract events -----------------------------------------
    print("\nPredicting across the full record")
    predictions = predict(model, flow)

    # Sanity check: the fixture flows should add up to the meter reading.
    measured = flow.clip(lower=0)
    busy = measured > NOISE_FLOOR
    ratio = (predictions.sum(axis=1)[busy] / measured[busy]).median()
    print(f"  predicted total / measured total: {ratio:.4f}  (1.0 is exact)")

    rates = {c: max(truth[c][truth[c] > NOISE_FLOOR].median(), 0.1)
             if (truth[c] > NOISE_FLOOR).any() else 1.0
             for c in FIXTURES}
    events = extract_events(predictions, rates)
    print(f"\n{len(events):,} events extracted")
    if not events.empty:
        print(events["fixture"].value_counts().to_string())

    # -- stagnation time ----------------------------------------------------
    print("\nStagnation time (hours between uses of the same fixture)")
    for code in ("Ad_S", "Kid_S", "Ad_T"):
        gaps = stagnation_time(events, code)
        if len(gaps):
            print(f"  {code:8} n={len(gaps):4d}  median {gaps.median():6.2f}  "
                  f"under 1 h: {(gaps < 1).mean():.1%}")
        else:
            print(f"  {code:8} too few events")

    if not events.empty:
        events.to_csv("events.csv", index=False)
        print("\nEvents written to events.csv")


if __name__ == "__main__":
    main()
