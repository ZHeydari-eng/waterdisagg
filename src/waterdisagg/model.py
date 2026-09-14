"""The disaggregation model.

Takes windows of aggregate household flow and predicts how that flow divides
among fixtures at every second.

Architecture
------------
Stacked LSTM encoder, then two heads:

**Allocation head.** Rather than regressing 14 flow rates directly, the model
predicts a distribution over ``F + 1`` channels at each timestep and
multiplies it by the observed aggregate::

    y_hat[t, f] = p[t, f] * x[t],    p[t] = softmax(W h[t])

This buys three properties that a bank of linear outputs does not have:

* predictions are non-negative, so no fixture can be assigned negative flow;
* they sum to the observed total by construction, which is the one hard
  physical constraint available in disaggregation;
* the learning problem is easier -- dividing a known quantity rather than
  regressing its magnitude from scratch.

The extra channel absorbs flow belonging to no modelled fixture. Real traces
contain draws no diary recorded, and anomalies such as a leaking toilet are
by definition unlike anything in the training data. Without somewhere to put
that flow, the model must attribute it to a real fixture, manufacturing
events in exactly the analysis where they do most damage.

**Activation head.** A parallel binary output trained with focal or weighted
cross-entropy. Squared error on flow is least sensitive at the low-magnitude
edges of an event, which is precisely where a threshold decision is made, so
thresholding a dedicated on/off output gives noticeably tighter event
boundaries than thresholding the regression. The regression is retained for
magnitudes and volumes.

Bidirectionality
----------------
The encoder is bidirectional by default. Disaggregation here is an offline
analysis, not real-time control, so there is no reason to withhold future
context -- and a filtered meter makes an event far easier to identify once
its termination has been seen. Set ``bidirectional: false`` for a causal
model if streaming inference is required.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = [
    "ModelConfig",
    "DisaggLSTM",
    "DisaggLoss",
    "train",
    "TrainHistory",
    "save_checkpoint",
    "load_checkpoint",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelConfig:
    """Model and optimisation hyperparameters.

    Defaults follow the values reported in the paper (two LSTM layers of 128
    units, Adam at 1e-3, batch size 64) with the additions documented in the
    module docstring.
    """

    n_fixtures: int
    hidden_size: int = 128
    n_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.15

    # Input scaling. The aggregate is divided by this before entering the
    # encoder; the allocation head still multiplies by the unscaled flow, so
    # predictions remain in gpm regardless.
    input_scale_gpm: float = 5.0

    # Loss weighting.
    flow_weight: float = 1.0
    activation_weight: float = 0.5
    # Relative weight on the unattributed channel. Kept below 1 so the model
    # is mildly discouraged from using it as a dumping ground, but not so low
    # that genuinely unexplained flow is forced onto a real fixture.
    unattributed_weight: float = 0.3
    # Per-fixture loss weights, indexed as the fixture list. Sparse fixtures
    # can be up-weighted here; None means uniform.
    fixture_weights: Sequence[float] | None = None

    # Optimisation.
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 100
    grad_clip: float = 1.0

    # Early stopping on validation loss. The paper reports a fixed 100
    # epochs; stopping early on a held-out split is strictly safer and does
    # not change the reported maximum.
    patience: int = 12
    min_delta: float = 1e-5

    seed: int = 20240606

    @property
    def n_channels(self) -> int:
        """Allocation channels: one per fixture, plus unattributed."""
        return self.n_fixtures + 1


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class DisaggLSTM(nn.Module):
    """Sequence-to-sequence disaggregation network.

    Parameters
    ----------
    config:
        Model hyperparameters.

    Shapes
    ------
    Input ``x``: ``(batch, time, 1)`` aggregate flow in gpm.
    Output ``flow``: ``(batch, time, n_fixtures)`` predicted flow in gpm.
    Output ``logits``: ``(batch, time, n_fixtures)`` activation logits.
    Output ``alloc``: ``(batch, time, n_fixtures + 1)`` allocation weights.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_size,
            num_layers=config.n_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        encoded = config.hidden_size * (2 if config.bidirectional else 1)

        self.norm = nn.LayerNorm(encoded)
        self.allocation_head = nn.Linear(encoded, config.n_channels)
        self.activation_head = nn.Linear(encoded, config.n_fixtures)

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal recurrent weights; forget-gate bias set to one.

        A positive forget-gate bias is the standard remedy for slow early
        learning of long dependencies, which matters here because showers run
        for hundreds of timesteps.
        """
        for name, param in self.encoder.named_parameters():
            if "weight_hh" in name:
                for chunk in param.chunk(4, 0):
                    nn.init.orthogonal_(chunk)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden = self.config.hidden_size
                param.data[hidden: 2 * hidden].fill_(1.0)

        for head in (self.allocation_head, self.activation_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the network.

        Parameters
        ----------
        x:
            ``(batch, time, 1)`` aggregate flow in gpm, unscaled.
        """
        if x.dim() != 3 or x.size(-1) != 1:
            raise ValueError(f"expected (batch, time, 1), got {tuple(x.shape)}")

        scaled = x / self.config.input_scale_gpm
        encoded, _ = self.encoder(scaled)
        encoded = self.norm(encoded)

        alloc = F.softmax(self.allocation_head(encoded), dim=-1)
        # Fixture channels are the first n_fixtures; the last is unattributed.
        flow = alloc[..., : self.config.n_fixtures] * x
        logits = self.activation_head(encoded)

        return {"flow": flow, "logits": logits, "alloc": alloc}

    @torch.no_grad()
    def predict_window(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning predicted flow only."""
        self.eval()
        return self.forward(x)["flow"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class DisaggLoss(nn.Module):
    """Combined flow-regression and activation-classification objective.

    The flow term is a masked mean absolute error rather than MSE. Squared
    error is dominated by the few high-flow fixtures and by the interior of
    long events; L1 spreads attention more evenly and is markedly less
    sensitive to the occasional mis-aligned label, which a hand-written diary
    guarantees. Set ``huber_delta`` to use a smooth L1 instead.

    Masking serves two purposes. Samples flagged invalid by the reader are
    excluded, because zero-filled dropouts are not observations of zero flow.
    Samples where the aggregate is below the noise floor are also excluded:
    the allocation of nothing is arbitrary, and including it lets the model
    reduce the loss by learning the idle distribution instead of the events.
    """

    def __init__(
        self,
        config: ModelConfig,
        noise_floor_gpm: float = 0.05,
        huber_delta: float | None = 0.25,
        activation_pos_weight: float = 4.0,
    ):
        super().__init__()
        self.config = config
        self.noise_floor = noise_floor_gpm
        self.huber_delta = huber_delta
        self.activation_pos_weight = activation_pos_weight

        if config.fixture_weights is not None:
            weights = torch.as_tensor(config.fixture_weights, dtype=torch.float32)
            if weights.numel() != config.n_fixtures:
                raise ValueError(
                    f"fixture_weights has {weights.numel()} entries, "
                    f"expected {config.n_fixtures}"
                )
        else:
            weights = torch.ones(config.n_fixtures, dtype=torch.float32)
        self.register_buffer("fixture_weights", weights)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_flow: torch.Tensor,
        valid: torch.Tensor | None = None,
        aggregate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the loss.

        Parameters
        ----------
        outputs:
            The dict returned by :meth:`DisaggLSTM.forward`.
        target_flow:
            ``(batch, time, n_fixtures)`` labelled per-fixture flow in gpm.
        valid:
            ``(batch, time)`` boolean; ``False`` marks samples to ignore.
        aggregate:
            ``(batch, time, 1)`` aggregate flow, used to build the activity
            mask. Falls back to the sum of the targets when not supplied.
        """
        predicted = outputs["flow"]
        logits = outputs["logits"]

        if predicted.shape != target_flow.shape:
            raise ValueError(
                f"prediction {tuple(predicted.shape)} does not match "
                f"target {tuple(target_flow.shape)}"
            )

        if aggregate is None:
            total = target_flow.sum(dim=-1, keepdim=True)
        else:
            total = aggregate

        mask = (total.squeeze(-1) > self.noise_floor)
        if valid is not None:
            mask = mask & valid.bool()
        mask = mask.unsqueeze(-1).float()

        # Two denominators are needed. The per-fixture terms sum over all F
        # channels, so their mean must divide by the number of *elements*
        # contributing; the unattributed term is a single channel. Using one
        # denominator for both inflates the per-fixture terms by a factor of
        # F and silently changes the balance between loss components.
        n_active = mask.sum().clamp_min(1.0)
        denom_fixture = (n_active * self.config.n_fixtures).clamp_min(1.0)
        denom_channel = n_active

        # -- flow regression -------------------------------------------
        if self.huber_delta is None:
            elementwise = (predicted - target_flow).abs()
        else:
            elementwise = F.smooth_l1_loss(
                predicted, target_flow, beta=self.huber_delta, reduction="none"
            )
        weighted = elementwise * self.fixture_weights.view(1, 1, -1)
        flow_loss = (weighted * mask).sum() / denom_fixture

        # -- activation classification ---------------------------------
        target_active = (target_flow > self.noise_floor).float()
        activation = F.binary_cross_entropy_with_logits(
            logits,
            target_active,
            reduction="none",
            pos_weight=torch.tensor(
                self.activation_pos_weight, device=logits.device
            ),
        )
        activation_loss = (activation * mask).sum() / denom_fixture

        # -- unattributed channel --------------------------------------
        # Penalise allocating flow to the escape channel, so it is used only
        # where the modelled fixtures genuinely cannot account for the total.
        alloc = outputs["alloc"]
        unattributed = alloc[..., -1:] * total
        unattributed_loss = (unattributed * mask).sum() / denom_channel

        total_loss = (
            self.config.flow_weight * flow_loss
            + self.config.activation_weight * activation_loss
            + self.config.unattributed_weight * unattributed_loss
        )

        return {
            "loss": total_loss,
            "flow": flow_loss.detach(),
            "activation": activation_loss.detach(),
            "unattributed": unattributed_loss.detach(),
            "n_active_samples": n_active.detach(),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrainHistory:
    """Per-epoch losses and the epoch selected by early stopping."""

    train_loss: list[float] = dataclasses.field(default_factory=list)
    val_loss: list[float] = dataclasses.field(default_factory=list)
    components: list[dict] = dataclasses.field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    stopped_early: bool = False

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "epoch": range(1, len(self.train_loss) + 1),
                "train_loss": self.train_loss,
                "val_loss": self.val_loss or [float("nan")] * len(self.train_loss),
            }
        )


def _device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: DisaggLSTM,
    criterion: DisaggLoss,
    loader,
    device: torch.device,
    optimizer=None,
) -> tuple[float, dict]:
    """One pass over a loader. Trains when an optimizer is supplied."""
    training = optimizer is not None
    model.train(training)

    totals: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        aggregate = batch["x"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        valid = batch.get("valid")
        if valid is not None:
            valid = valid.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            outputs = model(aggregate)
            losses = criterion(outputs, target, valid=valid, aggregate=aggregate)

        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if model.config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), model.config.grad_clip)
            optimizer.step()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1

    if n_batches == 0:
        raise ValueError("loader yielded no batches")
    means = {k: v / n_batches for k, v in totals.items()}
    return means["loss"], means


def train(
    model: DisaggLSTM,
    train_loader,
    val_loader=None,
    *,
    noise_floor_gpm: float = 0.05,
    device: str | None = None,
    checkpoint_path: Path | str | None = None,
    verbose: bool = True,
) -> TrainHistory:
    """Fit the model, keeping the best epoch by validation loss.

    Returns the training history. When ``checkpoint_path`` is given, the
    best-scoring weights are written there and reloaded into ``model`` before
    returning, so the object left in hand is the selected model rather than
    the final epoch's.
    """
    config = model.config
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dev = _device(device)
    model.to(dev)
    criterion = DisaggLoss(config, noise_floor_gpm=noise_floor_gpm).to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(config.patience // 3, 2)
    )

    history = TrainHistory()
    epochs_without_improvement = 0

    log.info(
        "training on %s: %s params, %d fixtures",
        dev,
        f"{sum(p.numel() for p in model.parameters()):,}",
        config.n_fixtures,
    )

    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_parts = _run_epoch(
            model, criterion, train_loader, dev, optimizer
        )
        history.train_loss.append(train_loss)
        history.components.append(train_parts)

        if val_loader is not None:
            val_loss, _ = _run_epoch(model, criterion, val_loader, dev)
            history.val_loss.append(val_loss)
            scheduler.step(val_loss)

            improved = val_loss < history.best_val_loss - config.min_delta
            if improved:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    save_checkpoint(model, checkpoint_path, epoch=epoch,
                                    val_loss=val_loss)
            else:
                epochs_without_improvement += 1

            if verbose:
                log.info(
                    "epoch %3d/%d  train %.5f  val %.5f%s",
                    epoch, config.max_epochs, train_loss, val_loss,
                    "  *" if improved else "",
                )

            if epochs_without_improvement >= config.patience:
                history.stopped_early = True
                log.info(
                    "early stop at epoch %d; best was epoch %d (val %.5f)",
                    epoch, history.best_epoch, history.best_val_loss,
                )
                break
        elif verbose:
            log.info("epoch %3d/%d  train %.5f", epoch, config.max_epochs, train_loss)

    if checkpoint_path is not None and history.best_epoch > 0:
        load_checkpoint(model, checkpoint_path, map_location=dev)
        log.info("restored best weights from epoch %d", history.best_epoch)

    return history


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: DisaggLSTM, path: Path | str, **metadata
) -> None:
    """Write weights, hyperparameters and arbitrary metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": dataclasses.asdict(model.config),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    model: DisaggLSTM | None, path: Path | str, map_location=None
) -> DisaggLSTM:
    """Load a checkpoint, constructing the model if one is not supplied."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is None:
        model = DisaggLSTM(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model
