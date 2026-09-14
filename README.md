# waterdisagg

> **Status: in development.** The data handling, labelling, windowing, model,
> inference and evaluation modules are complete and tested. The command-line
> scripts and the behavioural analysis modules (`analysis/`) are still being
> written, so the Quick Start below does not yet run end to end. The library
> API works; see `tests/` for usage.

Non-intrusive disaggregation of residential water use from a single-point
smart meter, and analysis of end-use behaviour through **stagnation time** —
the interval during which no water moves through a given fixture.

Supporting code for:

> Heydari, Z. and Stillwell, A. S. *From Data to Insight: Smart Monitoring
> Enables Residential Water Conservation Strategies.*

## What this does

A single flow meter on the main supply pipe records total household flow at
1-second resolution. This package divides that aggregate signal into
per-fixture contributions, converts them into discrete water-use events, and
derives behavioural metrics from those events.

```
aggregate flow  ->  per-fixture flow  ->  events  ->  stagnation time
   (1 Hz)            (14 fixtures)      (start, end,      (behavioural
                                         duration,          analysis,
                                         volume)         leak detection)
```

The disaggregation model is a bidirectional LSTM trained on short sliding
windows of aggregate flow, supervised by hand-recorded water diaries. Rather
than regressing each fixture's flow independently, the network predicts an
allocation over fixtures and multiplies it by the observed total, which
guarantees that predicted flows are non-negative and sum to what the meter
actually measured.

## Install

```bash
git clone https://github.com/<your-username>/waterdisagg.git
cd waterdisagg
pip install -e ".[dev]"
```

Python 3.10 or later. A GPU is optional; the model is small enough to train on
CPU, though a GPU is considerably faster.

## Quick start — no data required

The package includes a household simulator that generates a synthetic trace,
a matching hand-style water diary, and per-second ground truth. This makes the
whole pipeline runnable without access to any real household's data:

```bash
python scripts/01_prepare.py --demo --outdir runs/demo
python scripts/02_train.py   --run runs/demo
python scripts/03_predict.py --run runs/demo
python scripts/04_analyze.py --run runs/demo
```

The simulator reproduces the properties that make real data awkward:
instrument smoothing, a noisy non-zero idle baseline, dropped samples,
minute-resolution diary entries with no recorded duration for short events,
and concurrent fixture use.

## Using your own data

Two inputs are needed.

**An aggregate flow trace.** One or more CSV files of timestamped flow
readings. The reader handles two layouts, selected in the config: `flat`
(one row per reading) and `paired_rows` (records split across consecutive
rows, as some data-acquisition scripts produce).

**A water diary.** One row per observed fixture use, giving a start time, an
end time, and an identifier for the fixture. Diary times may be recorded to
the minute; event boundaries are taken from the flow trace, and only fixture
identity is taken from the diary.

Then copy and edit the configuration:

```bash
cp config/household.yaml config/my_house.yaml
```

Everything site-specific lives in that file — the fixture list, the mapping
from diary codes to fixtures, the meter's sampling period and noise floor,
window and threshold settings. No code changes are needed to apply the
pipeline to a different home. Fixture flow rates are fitted from your own
labelled events rather than assumed.

```bash
python scripts/01_prepare.py --config config/my_house.yaml \
    --trace data/trace/ --diary data/diary.csv --outdir runs/my_house
```

## Repository layout

```
config/household.yaml        Site configuration: fixtures, diary codes, thresholds
src/waterdisagg/
    io.py                    Trace parsing, resampling, validity masking
    rates.py                 Event segmentation; per-fixture rate fitting
    labels.py                Diary-to-trace matching; label matrix construction
    signatures.py            Template library extracted from training days
    synth.py                 Synthetic event augmentation
    windows.py               Windowing and day-based partitioning
    model.py                 Bidirectional LSTM with allocation head
    predict.py               Tiled inference over a continuous record
    events.py                Thresholding and event assembly
    metrics.py               Per-sample and event-level evaluation
    simulate.py              Synthetic household generator
    analysis/
        stagnation.py        Stagnation time; counterfactual user assignment
        clustering.py        k-means over shower events
        leak.py              Median shift and distribution tests
scripts/                     Four ordered entry points
tests/                       Test suite
```

The analysis modules import nothing from PyTorch, so behavioural analyses can
be re-run on saved predictions without a training environment.

## Notes on method

A few choices are worth flagging for anyone adapting this.

**Splits are by calendar day, never by window.** At the default 60-second
window and 15-second stride, adjacent windows share three quarters of their
samples. If windows are assigned to partitions at random, near-duplicates end
up on both sides of the split, and every reported metric is inflated.

**Sensor dropouts are masked, not zero-filled.** A gap in the record is not an
observation of zero flow, and treating it as one teaches the model that
missing data means idleness.

**Augmentation is applied to the training partition only,** using signatures
extracted only from training days. Evaluation uses observed events
exclusively.

**Fixture-level separation depends on fixtures differing in flow rate.**
Two fixtures of the same class flowing within a few percent of each other are
difficult to distinguish from a single aggregate signal. `rates.py` reports
the relative gap between every pair of fitted rates and flags those at risk,
so this limitation is visible rather than buried in an average. Where
fixture-level attribution is unreliable, aggregating to the end-use class
recovers usable performance.

## Data availability

The household trace and water diaries analysed in the paper are not included
here, to protect the privacy of the study occupants: fine-resolution water use
data reveal detailed patterns of daily activity. The simulator is provided so
that the pipeline can be run and inspected end to end without them. Requests
for access to the underlying data may be directed to the authors.

## Tests

```bash
pytest
```

## Citation

```bibtex
@article{heydari2026data,
  title   = {From Data to Insight: Smart Monitoring Enables Residential
             Water Conservation Strategies},
  author  = {Heydari, Zahra and Stillwell, Ashlynn S.},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2026}
}
```

## Acknowledgements

Supported by the National Science Foundation under Grant CBET-1847404, the
Taylor Geospatial Institute, and the Department of Civil and Environmental
Engineering in the Grainger College of Engineering at the University of
Illinois Urbana-Champaign. The custom ally® water meter was provided by
Sensus.

## License

MIT — see [LICENSE](LICENSE).
