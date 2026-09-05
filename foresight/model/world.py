"""The world model: learned transition dynamics over network state.

The statement is emphatic that the deliverable is not a classifier, so the
architecture is built around the transition itself:

    encoder   h_t      = f(S_{t-L+1} .. S_t)      history to latent state
    dynamics  S_{t+1}  = g(h_t)                   the world model proper
    risk      p_t      = r(h_t)                   infiltration probability

Only the dynamics head sees a regression target. It is trained to predict the
*next observed network state* -- the flags, port spread and timing that will be
seen one window from now -- which is what makes forward simulation possible:
feed the prediction back in, and the model rolls its own future.

The risk head is deliberately small and reads the same latent. It is not the
model; it is a readout. A rollout of K steps produces K latent states, and the
infiltration probability for the horizon is taken over those simulated futures
rather than over the present, which is the difference between forecasting and
detecting.

Attention is used rather than a plain LSTM state for one practical reason: the
statement requires explainability, and attention over the history window gives
a per-timestep attribution for free. Feature attribution comes separately from
the dynamics gradient.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Rollout:
    """K simulated future states and what they imply."""

    states: torch.Tensor          # (K, F) predicted network states
    risk: torch.Tensor            # (K,) infiltration probability per step
    attention: torch.Tensor       # (L,) weight over the observed history


class WorldModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.input = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.encoder = nn.LSTM(hidden, hidden, num_layers=layers,
                               batch_first=True, dropout=dropout if layers > 1 else 0.0)
        # Additive attention over the history, kept explicit rather than folded
        # into a transformer block so the weights can be read out and shown.
        self.attend = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1),
        )
        # The world model proper: latent -> the next observed state.
        self.dynamics = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_features),
        )
        # A readout, not the model.
        self.risk = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1),
        )
        self.heads = heads

    def encode(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L, F) history -> (B, H) latent and (B, L) attention weights."""
        hidden, _ = self.encoder(self.input(window))
        scores = self.attend(hidden).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        latent = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return latent, weights

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: predicted next state, infiltration logit, attention."""
        latent, weights = self.encode(window)
        return self.dynamics(latent), self.risk(latent).squeeze(-1), weights

    @torch.no_grad()
    def rollout(self, window: torch.Tensor, steps: int = 6) -> Rollout:
        """Simulate `steps` windows forward from an observed history.

        Each predicted state is appended to the history and the oldest dropped,
        so step k is conditioned on k-1 of the model's own predictions. Error
        compounds, which is honest: a world model that cannot hold its own
        trajectory for K steps should not be trusted K steps out, and the
        evaluation reports accuracy per horizon for that reason.
        """
        self.eval()
        history = window.clone()
        states, risks = [], []
        first_attention = None
        for _ in range(steps):
            nxt, logit, weights = self.forward(history)
            if first_attention is None:
                first_attention = weights[0].detach()
            states.append(nxt[0].detach())
            risks.append(torch.sigmoid(logit)[0].detach())
            history = torch.cat([history[:, 1:], nxt.unsqueeze(1)], dim=1)
        return Rollout(states=torch.stack(states), risk=torch.stack(risks),
                       attention=first_attention)


def simulate(model: "WorldModel", window: torch.Tensor, steps: int) -> torch.Tensor:
    """Differentiable K-step rollout, for training the trajectory itself.

    Training one step ahead and then rolling out K is the standard way to get a
    world model that oscillates: each predicted state is slightly off the data
    manifold, the next prediction is conditioned on that, and the error
    compounds. Ours alternated between a near-empty state and a scan signature
    on successive steps, which is not a trajectory anyone should act on.

    Supervising the rollout directly teaches the model to stay stable when fed
    its own output. Returns (B, steps, F).
    """
    history, out = window, []
    for _ in range(steps):
        latent, _ = model.encode(history)
        nxt = model.dynamics(latent)
        out.append(nxt)
        history = torch.cat([history[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(out, dim=1)


def dynamics_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber on the next-state prediction.

    Squared error is dominated by the volume features, which span orders of
    magnitude and swamp the flag and entropy signals that actually distinguish
    reconnaissance. Huber keeps large residuals from taking over the gradient.
    """
    return nn.functional.huber_loss(predicted, target, delta=1.0)
