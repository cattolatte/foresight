"""MITRE ATT&CK stage mapping, fitted and measured rather than asserted.

The statement requires predicted states to be mapped to attack stages and says
that black-box output is not acceptable. The first version of this was a rule
cascade: thresholds on port entropy, fan-out and SYN counts, each branch
returning the evidence that fired. It read well and it was decorative.

Measured against the data, benign traffic already sits above most of the
thresholds and no attack family reaches the rest. Median benign
`unique_dst_ports` is 19 against a rule that fired above 8; `fanout` never
exceeds 0.41 in any family against a rule needing 1.2; `SYN Flag Count` peaks
at 0.13 against a rule needing 0.6. Only two outcomes were reachable -- the
branch that fired on everything, and the fallback -- so every window on both
test days was named "Initial Access" or "Exfiltration" regardless of content.

What replaces it is a multinomial logistic regression over the same named
features, fitted on the training days only and scored on the held-out days. It
is still interpretable -- one weight per feature per stage, and the evidence
returned is the features that actually carried the decision -- but its accuracy
is a measurement instead of a claim, and it is reported per stage below.

One honest gap. The statement names five stages; CIC-IDS2017 contains no
exfiltration ground truth at all, so that class cannot be fitted or scored and
is never predicted. Denial-of-service traffic maps to Impact, which is not one
of the five named stages, and is carried as its own class rather than being
forced into one that does not fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAGES = ["Reconnaissance", "Initial Access", "Lateral Movement",
          "Command & Control", "Exfiltration", "Impact (DoS)"]

# Families to stages. CIC-IDS2017 labels the attack tool; the stage is what
# that tool is doing in the kill chain.
FAMILY_STAGE: dict[str, str] = {
    "PortScan": "Reconnaissance",
    "FTP-Patator": "Initial Access",
    "SSH-Patator": "Initial Access",
    "Web Attack – Brute Force": "Initial Access",
    "Web Attack – XSS": "Initial Access",
    "Web Attack – Sql Injection": "Initial Access",
    "Heartbleed": "Initial Access",
    "Infiltration": "Lateral Movement",
    "Bot": "Command & Control",
    "DoS Hulk": "Impact (DoS)",
    "DoS GoldenEye": "Impact (DoS)",
    "DoS slowloris": "Impact (DoS)",
    "DoS Slowhttptest": "Impact (DoS)",
    "DDoS": "Impact (DoS)",
}


def stage_of(family: str) -> str | None:
    """The stage a family belongs to, or None for benign/unknown."""
    return FAMILY_STAGE.get(family)


@dataclass
class StageModel:
    """A linear stage classifier over named features."""

    weights: np.ndarray          # (F+1, C), row 0 is the intercept
    classes: list[str]
    columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    # Held-out precision per class, filled in by eval/stages.py. Softmax
    # confidence alone is not enough to judge a call: the classifier is
    # routinely confident about reconnaissance and right about it 13% of the
    # time, so the interface needs the measured reliability of the class it
    # just named, not only how sure the model felt.
    precision: dict[str, float] | None = None

    def _design(self, states: np.ndarray) -> np.ndarray:
        z = (np.log1p(np.abs(states)) * np.sign(states) - self.mean) / self.std
        return np.hstack([np.ones((len(z), 1)), z])

    def probabilities(self, states: np.ndarray) -> np.ndarray:
        logits = self._design(np.atleast_2d(states)) @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, state: np.ndarray) -> tuple[str, str, float]:
        """Stage, the evidence that carried it, and the confidence."""
        probs = self.probabilities(state)[0]
        k = int(np.argmax(probs))
        design = self._design(np.atleast_2d(state))[0]
        # Contribution of each feature to this class's logit, against the mean
        # of the others: what pushed it here rather than anywhere else.
        rival = np.delete(self.weights, k, axis=1).mean(axis=1)
        contrib = design[1:] * (self.weights[1:, k] - rival[1:])
        top = np.argsort(-np.abs(contrib))[:3]
        evidence = ", ".join(
            f"{self.columns[i]} {'raises' if contrib[i] > 0 else 'lowers'} it "
            f"({contrib[i]:+.2f})" for i in top)
        return self.classes[k], evidence, float(probs[k])


def fit_stage_model(states: np.ndarray, families: list[str], columns: list[str],
                    epochs: int = 400, lr: float = 0.4,
                    l2: float = 1e-3) -> StageModel:
    """Fit the stage classifier on labelled attack windows only.

    Benign windows are excluded: the question this answers is "given that
    something is happening, which stage is it", and the risk head already
    answers whether something is happening.
    """
    labels = [stage_of(f) for f in families]
    keep = np.array([l is not None for l in labels])
    if not keep.any():
        raise ValueError("no attack windows to fit stage model on")
    x_raw = states[keep]
    y_names = [l for l in labels if l is not None]

    classes = sorted(set(y_names), key=STAGES.index)
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[c] for c in y_names])

    signed = np.log1p(np.abs(x_raw)) * np.sign(x_raw)
    mean = signed.mean(axis=0)
    std = signed.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (signed - mean) / std
    design = np.hstack([np.ones((len(z), 1)), z])

    onehot = np.zeros((len(y), len(classes)))
    onehot[np.arange(len(y)), y] = 1.0
    # Class-balanced: DoS windows outnumber lateral movement by an order of
    # magnitude and an unweighted fit predicts the majority class everywhere.
    freq = onehot.sum(axis=0)
    weight = (len(y) / (len(classes) * np.maximum(freq, 1)))[y][:, None]

    w = np.zeros((design.shape[1], len(classes)))
    for _ in range(epochs):
        logits = design @ w
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
        grad = design.T @ ((probs - onehot) * weight) / len(y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return StageModel(weights=w, classes=classes, columns=columns,
                      mean=mean, std=std)


def confusion(model: StageModel, states: np.ndarray,
              families: list[str]) -> tuple[np.ndarray, list[str], float]:
    """Confusion matrix on held-out windows, true stages as rows.

    Rows cover every stage present in the truth, including stages the model was
    never fitted on. An earlier version listed only the fitted classes, which
    silently dropped those rows from the matrix while still counting them as
    errors in the accuracy -- the matrix looked balanced and the accuracy did
    not, with nothing on screen to explain the gap.
    """
    labels = [stage_of(f) for f in families]
    keep = np.array([l is not None for l in labels])
    truth = [l for l in labels if l is not None]
    if not keep.any():
        return np.zeros((0, 0), dtype=int), model.classes, float("nan")
    probs = model.probabilities(states[keep])
    pred = [model.classes[i] for i in probs.argmax(axis=1)]

    names = sorted(set(truth) | set(model.classes), key=STAGES.index)
    idx = {c: i for i, c in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    hits = 0
    for t, p in zip(truth, pred):
        matrix[idx[t], idx[p]] += 1
        hits += t == p
    return matrix, names, hits / len(truth)


def blocked_split(n: int, gap: int, block: int = 40,
                  fraction: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Split one day into alternating blocks, with a gap at every boundary.

    A single chronological cut does not work here. Each attack in this capture
    runs once, for a scheduled stretch of the afternoon, so one cut puts an
    entire stage on one side: at 70/30 the fitted model saw no Command &
    Control at all and every reconnaissance window was on the far side of the
    boundary. Interleaving blocks puts part of every attack period on both
    sides while keeping each part contiguous.

    The gap matters as much as the interleaving. Sliding windows overlap -- a
    60s window every 15s shares three quarters of its traffic with its
    neighbour -- so the windows either side of a boundary are near-copies. The
    gap is discarded so no fitted window shares a flow with a scored one.
    """
    train = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    cut = max(1, int(round(block * fraction)))
    for start in range(0, n, block):
        stop = min(start + block, n)
        edge = min(start + cut, stop)
        # The gap is needed at the block boundary too, not only at the internal
        # train/test edge. Without it each block's training region began one
        # index after the previous block's test region ended, which is exactly
        # the adjacency the gap exists to prevent.
        begin = start if start == 0 else min(start + gap, stop)
        train[begin:max(begin, edge - gap)] = True
        test[min(edge + gap, stop):stop] = True
    return train, test
