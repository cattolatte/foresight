"""The logistic regression baseline the statement requires.

    "Benchmark results comparing model performance (F1 score, precision,
     recall, false positive rate) against a logistic regression baseline
     trained on the same features."

Same features, same windows, same day split, same forecast label. The only
difference is that the baseline sees one window and the world model sees a
history of twelve and simulates forward. That is the comparison worth making:
if temporal dynamics add nothing, this baseline will say so.

It is given every reasonable advantage -- class balancing, scaled inputs, and
the same flattened history the world model gets in a second variant -- because
a baseline built to lose proves nothing.
"""
from __future__ import annotations

import numpy as np


class LogisticBaseline:
    """Plain logistic regression, fitted with gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 400, balance: bool = True):
        self.lr, self.epochs, self.balance = lr, epochs, balance
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        n, d = x.shape
        self.w, self.b = np.zeros(d), 0.0
        # Balanced, so the baseline is not beaten merely by class imbalance.
        weight = np.ones(n)
        if self.balance and 0 < y.mean() < 1:
            weight = np.where(y == 1, (1 - y.mean()) / y.mean(), 1.0)
        weight = weight / weight.mean()

        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(x @ self.w + self.b)))
            err = (p - y) * weight
            self.w -= self.lr * (x.T @ err) / n
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(x @ self.w + self.b)))


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """The four the statement names, plus AUC and the operating threshold."""
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "auc": _auc(scores, labels), "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-sum AUC, with tied scores sharing the average rank.

    The tie handling is not a detail here. The previous version ranked by
    argsort alone, which hands tied scores arbitrary distinct ranks, and the
    baselines this project compares against are exactly the tied cases: a
    persistence baseline takes two values, an always-positive baseline takes
    one. It scored a coin-flip binary predictor at 0.250 and a constant score
    at 0.250, both of which are 0.500. The error understates tied scorers, so
    it flattered the model against precisely the baselines meant to keep it
    honest.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.min() == labels.max():
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks_in_order = np.arange(1, len(scores) + 1, dtype=float)
    start = 0
    for i in range(1, len(ordered) + 1):
        if i == len(ordered) or ordered[i] != ordered[start]:
            ranks_in_order[start:i] = ranks_in_order[start:i].mean()
            start = i
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = ranks_in_order

    pos, neg = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising F1, chosen on the split it is reported for.

    Stated plainly because it flatters both models equally: each is given its
    own best operating point, so the comparison is between the curves rather
    than between two arbitrary cut-offs.
    """
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    return float(max(candidates, key=lambda t: metrics(scores, labels, t)["f1"]))
