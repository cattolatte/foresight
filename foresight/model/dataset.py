"""Turn day-windowed states into supervised sequences for dynamics learning.

Two decisions here carry the whole claim of the project.

The risk label is the future, never the present. y_t is 1 when an attack falls
anywhere in windows t+1 .. t+K, and the window at t is excluded even if it is
already malicious. A model trained on the present learns detection and will
score well the instant compromise happens; the statement asks for something
that speaks before that. Excluding the current window is what makes the metric
mean what it says.

The split is by capture day, not by random sampling. Monday through Wednesday
carry benign traffic, brute force and denial of service; Thursday and Friday
carry web attacks, infiltration, port scanning, botnet and DDoS. Splitting
randomly would put windows minutes apart into train and test and measure
memorisation. Splitting by day means the test families were never seen, which
is what "generalise to unseen attack patterns" requires.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

# Monday is pure benign; Tuesday and Wednesday supply brute force and DoS.
TRAIN_DAYS = ["2017-07-03", "2017-07-04", "2017-07-05"]
# Web attacks, infiltration, port scan, bot and DDoS -- none seen in training.
TEST_DAYS = ["2017-07-06", "2017-07-07"]


@dataclass
class Normaliser:
    """Standardisation fitted on training days only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, states: np.ndarray) -> "Normaliser":
        # Log1p first: byte and packet counts span several orders of magnitude,
        # and without it the flag and entropy features contribute nothing.
        shaped = np.log1p(np.abs(states)) * np.sign(states)
        return cls(mean=shaped.mean(0), std=shaped.std(0) + 1e-6)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        shaped = np.log1p(np.abs(states)) * np.sign(states)
        return ((shaped - self.mean) / self.std).astype(np.float32)

    def invert(self, states: np.ndarray) -> np.ndarray:
        """Back to raw units.

        The stage rules are written in the units a defender reads -- ports
        touched, flows seen, entropy in bits -- so a predicted state has to be
        returned to that scale before they are applied. Applying thresholds
        meant for raw counts to z-scores silently matched nothing, and every
        prediction came back "no distinguishing signature".
        """
        shaped = np.asarray(states, dtype=np.float64) * self.std + self.mean
        return np.sign(shaped) * np.expm1(np.abs(shaped))


class SequenceSet(Dataset):
    """(history, next state, future risk) triples from one or more days."""

    def __init__(self, windows: list, norm: Normaliser, length: int = 12,
                 horizon: int = 6, gap: int = 4):
        self.length, self.horizon, self.gap = length, horizon, gap
        self.history: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        # The true next `horizon` states, so forward simulation can be trained
        # against the trajectory it will be asked to produce -- not just the
        # single step it happens to be scored on.
        self.future: list[np.ndarray] = []
        self.risk: list[float] = []
        self.origin: list[tuple[str, int]] = []

        for day in windows:
            states = norm(day.states)
            labels = day.labels
            n = len(states)
            # A slice shorter than one history plus one horizon yields nothing;
            # skipping it is correct, but silently producing empty slices is
            # how an off-by-one turns into a crash three layers away.
            if n < length + gap + horizon + 1:
                continue
            for t in range(length, n - horizon - gap):
                self.history.append(states[t - length:t])
                self.target.append(states[t])
                self.future.append(states[t:t + horizon])
                # Strictly the future, and strictly unobserved.
                #
                # Excluding the present window is not enough when windows
                # overlap. A 60s window every 15s means window t+1 covers
                # traffic from 45s before t, half of which the history already
                # contains, and t+2 a quarter of it. Labelling from t+1 let an
                # attack visible in the model's own input set the forecast
                # label, so two of six horizon steps were detection wearing a
                # forecast's name. The gap is window/stride, the first offset
                # whose traffic the history has not seen.
                self.risk.append(
                    float(labels[t + gap:t + gap + horizon].max() > 0))
                self.origin.append((day.day, t))

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.history[i]),
                torch.from_numpy(self.target[i]),
                torch.from_numpy(self.future[i]),
                torch.tensor(self.risk[i], dtype=torch.float32))
