"""Forecasting by surprise: a world model trained only on benign traffic.

The benchmark this project shipped measures a task that is easier than it
looks. Attack episodes in this capture run for minutes -- median 60s on
Thursday, 240s on Friday, up to 70 minutes -- against a 90s horizon, so "is
there an attack in the next 90 seconds" is very nearly "is there an attack
now". A persistence baseline that uses the ground-truth label of the last
observed window and no features at all scores 0.898 AUC. Nothing that scores
below that is forecasting; it is detecting, and being graded on autocorrelation.

The honest question is whether an attack *begins* when nothing is yet visible.
Restricted to windows whose entire observed history is benign, the ranking
inverts: the shipped world model scores 0.757 against 0.772 for a logistic
regression on the current window. It was never trained for that task, and it
cannot easily be -- the three training days contain 83 onset-positive windows,
about fourteen attack episodes, which is not enough to fit a model of this size.

So this takes the other route, and it is the one a world model is actually for.
Train the dynamics on benign traffic only, with no attack labels anywhere in the
objective, and score a window by how badly the model predicted it. An attack is
then not a class to be recognised but a state the benign dynamics did not
anticipate. Unseen attack families need no special handling, because no attack
family was ever fitted.

The control that matters is the naive dynamics: predict that the next state
equals the current one, and score by the size of the change. Traffic is bursty,
and a model that cannot beat "something moved" has learned nothing about how
networks evolve.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from foresight.baseline import LogisticBaseline, metrics
from foresight.data.flows import build_windows, load_flows
from foresight.model.dataset import (SequenceSet, Normaliser, TEST_DAYS,
                                     TRAIN_DAYS)
from foresight.model.world import WorldModel


def quiet_mask(sequences: SequenceSet, windows: list, length: int) -> np.ndarray:
    """Sequences whose entire observed history is benign."""
    label = {(w.day, i): (w.labels[i] > 0)
             for w in windows for i in range(len(w.labels))}
    return np.array([not any(label.get((d, t - k), False)
                             for k in range(1, length + 1))
                     for d, t in sequences.origin])


def train_benign(states: np.ndarray, targets: np.ndarray, n_features: int,
                 epochs: int = 80, seed: int = 0) -> WorldModel:
    """Fit one-step dynamics on benign sequences. No labels are used."""
    torch.manual_seed(seed)
    model = WorldModel(n_features=n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(states),
                                      torch.from_numpy(targets)),
                        batch_size=64, shuffle=True)
    loss_fn = torch.nn.HuberLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for history, target in loader:
            opt.zero_grad()
            predicted, _, _ = model(history)
            loss = loss_fn(predicted, target)
            loss.backward()
            opt.step()
            total += float(loss) * len(history)
        if epoch % 20 == 19:
            print(f"    epoch {epoch + 1:3d}  dynamics loss {total / len(states):.4f}",
                  flush=True)
    return model.eval()


def main() -> None:
    cfg = json.loads(Path("checkpoints/world/config.json").read_text())
    length, horizon = cfg["length"], cfg["horizon"]
    gap = cfg.get("gap", 4)
    flows = load_flows()
    train_windows = [build_windows(flows, d, cfg["window"], cfg["stride"])
                     for d in TRAIN_DAYS]
    test_windows = [build_windows(flows, d, cfg["window"], cfg["stride"])
                    for d in TEST_DAYS]

    saved = np.load("checkpoints/world/norm.npz")
    norm = Normaliser(mean=saved["mean"], std=saved["std"])
    train_set = SequenceSet(train_windows, norm, length, horizon, gap)
    test_set = SequenceSet(test_windows, norm, length, horizon, gap)

    q_train = quiet_mask(train_set, train_windows, length)
    q_test = quiet_mask(test_set, test_windows, length)
    y_train = np.array(train_set.risk)
    y_test = np.array(test_set.risk)

    history_tr = np.stack(train_set.history)
    target_tr = np.stack(train_set.target)
    history_te = np.stack(test_set.history)
    target_te = np.stack(test_set.target)

    print(__doc__)
    print("=" * 74)
    print(f"benign training sequences: {q_train.sum():,} of {len(y_train):,} "
          f"(no attack label enters the objective)")
    model = train_benign(history_tr[q_train], target_tr[q_train],
                         cfg["n_features"])

    with torch.no_grad():
        predicted, _, _ = model(torch.from_numpy(history_te))
    surprise = ((predicted - torch.from_numpy(target_te)) ** 2).mean(1).numpy()
    # The control: predict no change at all, score the size of the movement.
    naive = ((history_te[:, -1] - target_te) ** 2).mean(1)

    # A supervised reference, refit on the same restricted task.
    flat_tr = history_tr.reshape(len(history_tr), -1)
    flat_te = history_te.reshape(len(history_te), -1)
    lr = LogisticBaseline().fit(flat_tr[q_train], y_train[q_train])
    supervised = lr.predict_proba(flat_te)

    print("\n" + "=" * 74)
    for name, mask in [("all test windows", np.ones(len(y_test), bool)),
                       ("onset only — observed history entirely benign", q_test)]:
        print(f"\n{name}   n={mask.sum():,}  base rate {y_test[mask].mean():.3f}")
        print(f"  {'signal':<46}{'AUC':>7}{'F1':>7}")
        for label, score in [
                ("world model surprise (benign-trained, unsupervised)", surprise),
                ("naive dynamics: size of the change", naive),
                ("logistic regression on history (supervised)", supervised)]:
            m = metrics(score[mask], y_test[mask])
            print(f"  {label:<46}{m['auc']:>7.3f}{m['f1']:>7.3f}")

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/surprise.json").write_text(json.dumps({
        "onset_auc": {
            "surprise": metrics(surprise[q_test], y_test[q_test])["auc"],
            "naive": metrics(naive[q_test], y_test[q_test])["auc"],
            "supervised": metrics(supervised[q_test], y_test[q_test])["auc"]},
        "onset_positives_train": int(y_train[q_train].sum()),
    }, indent=2))
    torch.save(model.state_dict(), "checkpoints/world/benign_dynamics.pt")
    print("\nwrote eval/results/surprise.json and checkpoints/world/benign_dynamics.pt")


if __name__ == "__main__":
    main()
