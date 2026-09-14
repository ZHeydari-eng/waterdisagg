"""Tests for the disaggregation model.

The constraint tests matter more than they might appear. Non-negativity and
sum-to-total are the properties that distinguish this formulation from a bank
of linear regression outputs, and they are enforced by construction rather
than learned -- so they should hold at initialisation, not merely after
training.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from waterdisagg.model import (  # noqa: E402
    DisaggLoss,
    DisaggLSTM,
    ModelConfig,
    load_checkpoint,
    save_checkpoint,
    train,
)


@pytest.fixture
def config():
    return ModelConfig(n_fixtures=5, hidden_size=16, n_layers=1, max_epochs=2)


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return DisaggLSTM(config)


# ---------------------------------------------------------------------------
# Shapes and constraints
# ---------------------------------------------------------------------------


def test_output_shapes(model, config):
    x = torch.rand(3, 40, 1) * 4
    out = model(x)
    assert out["flow"].shape == (3, 40, config.n_fixtures)
    assert out["logits"].shape == (3, 40, config.n_fixtures)
    assert out["alloc"].shape == (3, 40, config.n_fixtures + 1)


def test_rejects_wrong_input_shape(model):
    with pytest.raises(ValueError):
        model(torch.rand(3, 40))
    with pytest.raises(ValueError):
        model(torch.rand(3, 40, 2))


def test_predictions_are_non_negative(model):
    """Negative flow is physically meaningless and must be impossible."""
    x = torch.rand(4, 50, 1) * 6
    assert model(x)["flow"].min() >= 0.0


def test_predictions_sum_to_observed_total(model, config):
    """Fixture flows plus the unattributed channel must equal the input.

    This is the one hard physical constraint available in disaggregation, and
    the softmax allocation enforces it identically rather than approximately.
    """
    x = torch.rand(4, 50, 1) * 6
    out = model(x)
    reconstructed = (
        out["flow"].sum(-1) + out["alloc"][..., -1] * x.squeeze(-1)
    )
    torch.testing.assert_close(
        reconstructed, x.squeeze(-1), atol=1e-5, rtol=1e-5
    )


def test_zero_input_gives_zero_output(model):
    """No flow in means no flow attributed to any fixture."""
    out = model(torch.zeros(2, 30, 1))
    assert out["flow"].abs().max() == 0.0


def test_allocation_is_a_distribution(model):
    x = torch.rand(2, 30, 1) * 3
    alloc = model(x)["alloc"]
    torch.testing.assert_close(
        alloc.sum(-1), torch.ones(2, 30), atol=1e-5, rtol=1e-5
    )
    assert alloc.min() >= 0.0


def test_bidirectional_uses_future_context():
    """A bidirectional encoder must respond to later timesteps.

    Establishes that the flag has the intended effect: with a causal model,
    perturbing the end of a sequence cannot change predictions at the start.
    """
    torch.manual_seed(1)
    x = torch.rand(1, 40, 1) * 3

    causal = DisaggLSTM(
        ModelConfig(n_fixtures=4, hidden_size=16, n_layers=1, bidirectional=False)
    ).eval()
    bidir = DisaggLSTM(
        ModelConfig(n_fixtures=4, hidden_size=16, n_layers=1, bidirectional=True)
    ).eval()

    perturbed = x.clone()
    perturbed[0, 35:, 0] += 2.0

    with torch.no_grad():
        causal_delta = (
            causal(x)["alloc"][0, :10] - causal(perturbed)["alloc"][0, :10]
        ).abs().max()
        bidir_delta = (
            bidir(x)["alloc"][0, :10] - bidir(perturbed)["alloc"][0, :10]
        ).abs().max()

    assert causal_delta < 1e-6
    assert bidir_delta > 1e-6


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_loss_components_are_finite_and_positive(model, config):
    x = torch.rand(3, 40, 1) * 4
    y = torch.rand(3, 40, config.n_fixtures) * 0.5
    losses = DisaggLoss(config)(model(x), y, aggregate=x)
    for key in ("loss", "flow", "activation", "unattributed"):
        assert torch.isfinite(losses[key]).all()
        assert float(losses[key]) >= 0.0


def test_loss_normalisation_is_independent_of_fixture_count():
    """Per-fixture loss terms must be means, not sums over channels.

    Dividing a sum over F channels by a count of one channel inflates the
    term by F, which silently rebalances the objective as the fixture list
    grows. Two models differing only in fixture count should report losses of
    comparable magnitude on equivalent data.
    """
    losses = []
    for n_fixtures in (3, 30):
        cfg = ModelConfig(n_fixtures=n_fixtures, hidden_size=8, n_layers=1)
        torch.manual_seed(0)
        net = DisaggLSTM(cfg)
        x = torch.rand(2, 30, 1) * 3
        # One fixture carries the whole flow; the rest are idle.
        y = torch.zeros(2, 30, n_fixtures)
        y[..., 0] = x.squeeze(-1)
        losses.append(float(DisaggLoss(cfg)(net(x), y, aggregate=x)["flow"]))
    assert losses[1] < losses[0] * 3


def test_loss_ignores_invalid_samples(model, config):
    """Samples flagged invalid must not contribute.

    Zero-filled sensor dropouts are not observations of zero flow; including
    them teaches the model that gaps mean idleness.
    """
    x = torch.rand(2, 40, 1) * 4
    y = torch.rand(2, 40, config.n_fixtures) * 0.5
    criterion = DisaggLoss(config)
    out = model(x)

    all_valid = torch.ones(2, 40, dtype=torch.bool)
    half_valid = all_valid.clone()
    half_valid[:, 20:] = False

    full = criterion(out, y, valid=all_valid, aggregate=x)
    partial = criterion(out, y, valid=half_valid, aggregate=x)
    assert float(partial["n_active_samples"]) < float(full["n_active_samples"])


def test_loss_ignores_idle_samples(config, model):
    """Allocation of near-zero flow is arbitrary and must be masked out."""
    x = torch.zeros(2, 40, 1)
    y = torch.zeros(2, 40, config.n_fixtures)
    losses = DisaggLoss(config, noise_floor_gpm=0.05)(model(x), y, aggregate=x)
    # No sample exceeds the noise floor, so nothing contributes.
    assert float(losses["n_active_samples"]) == 1.0  # clamped floor
    assert float(losses["flow"]) == 0.0


def test_loss_rejects_mismatched_shapes(model, config):
    x = torch.rand(2, 40, 1)
    y = torch.rand(2, 40, config.n_fixtures + 1)
    with pytest.raises(ValueError):
        DisaggLoss(config)(model(x), y, aggregate=x)


def test_fixture_weights_validated():
    cfg = ModelConfig(n_fixtures=5, fixture_weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        DisaggLoss(cfg)


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


def _separable_task(n=256, n_fixtures=3, length=60, seed=0):
    """Windows in which exactly one fixture runs, at a distinct rate.

    Deliberately easy: if the model cannot solve this, the fault is in the
    implementation rather than in the difficulty of real disaggregation.
    """
    rng = np.random.default_rng(seed)
    rates = [1.0, 2.0, 3.0][:n_fixtures]
    x = torch.zeros(n, length, 1)
    y = torch.zeros(n, length, n_fixtures)
    for i in range(n):
        f = int(rng.integers(n_fixtures))
        start = int(rng.integers(0, length // 2))
        end = start + int(rng.integers(15, 30))
        y[i, start:end, f] = rates[f]
        x[i, start:end, 0] = rates[f]
    return x, y


def test_model_learns_separable_task():
    torch.manual_seed(0)
    x, y = _separable_task()
    cfg = ModelConfig(n_fixtures=3, hidden_size=32, n_layers=1)
    model = DisaggLSTM(cfg)
    criterion = DisaggLoss(cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)

    initial = float(criterion(model(x[:64]), y[:64], aggregate=x[:64])["loss"])
    for _ in range(40):
        for b in range(0, len(x), 64):
            out = model(x[b: b + 64])
            loss = criterion(out, y[b: b + 64], aggregate=x[b: b + 64])["loss"]
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    final = float(criterion(model(x[:64]), y[:64], aggregate=x[:64])["loss"])

    assert final < initial * 0.2, f"loss did not fall: {initial} -> {final}"

    # Attribution should be correct wherever there is flow to attribute.
    with torch.no_grad():
        out = model(x[:64])
    active = x[:64, :, 0] > 0.05
    predicted = out["flow"][:64].argmax(-1)[active]
    truth = y[:64].argmax(-1)[active]
    assert (predicted == truth).float().mean() > 0.95


# ---------------------------------------------------------------------------
# Training loop and persistence
# ---------------------------------------------------------------------------


class _Loader:
    """Minimal batch iterator, avoiding a DataLoader dependency in tests."""

    def __init__(self, x, y, batch_size=32):
        self.x, self.y, self.batch_size = x, y, batch_size

    def __iter__(self):
        for b in range(0, len(self.x), self.batch_size):
            yield {
                "x": self.x[b: b + self.batch_size],
                "y": self.y[b: b + self.batch_size],
            }


def test_train_reduces_loss_and_records_history(tmp_path):
    torch.manual_seed(0)
    x, y = _separable_task(n=128)
    cfg = ModelConfig(
        n_fixtures=3, hidden_size=16, n_layers=1, max_epochs=5, patience=5
    )
    model = DisaggLSTM(cfg)
    history = train(
        model,
        _Loader(x[:96], y[:96]),
        _Loader(x[96:], y[96:]),
        checkpoint_path=tmp_path / "ckpt.pt",
        verbose=False,
    )
    assert len(history.train_loss) >= 1
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.best_epoch >= 1
    assert (tmp_path / "ckpt.pt").exists()


def test_train_without_validation_loader():
    torch.manual_seed(0)
    x, y = _separable_task(n=64)
    cfg = ModelConfig(n_fixtures=3, hidden_size=8, n_layers=1, max_epochs=2)
    history = train(DisaggLSTM(cfg), _Loader(x, y), verbose=False)
    assert len(history.train_loss) == 2
    assert history.val_loss == []


def test_train_raises_on_empty_loader():
    cfg = ModelConfig(n_fixtures=3, hidden_size=8, n_layers=1, max_epochs=1)
    with pytest.raises(ValueError):
        train(DisaggLSTM(cfg), _Loader(torch.zeros(0, 10, 1), torch.zeros(0, 10, 3)),
              verbose=False)


def test_checkpoint_round_trip(tmp_path, model, config):
    x = torch.rand(2, 30, 1) * 3
    with torch.no_grad():
        before = model(x)["flow"]

    path = tmp_path / "model.pt"
    save_checkpoint(model, path, epoch=7, note="test")

    restored = load_checkpoint(None, path)
    with torch.no_grad():
        after = restored(x)["flow"]

    torch.testing.assert_close(before, after)
    assert restored.config.n_fixtures == config.n_fixtures
