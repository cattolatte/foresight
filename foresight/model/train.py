"""Train the world model: dynamics first, risk as a readout.

The loss is dominated by the dynamics term on purpose. If the risk head carries
the gradient the encoder becomes a classifier with extra steps, which is the
thing the statement rules out. Weighting dynamics above risk keeps the latent
answering "what happens next" rather than "is this bad", and the rollout is
only meaningful if that holds.

Selection is on a validation split carved from the *training* days, never on
the test days. Choosing the checkpoint by test AUC is test-set selection: with
a peak around epoch 11 and a decline after, picking that peak by looking at the
test set reports a number the model would not reproduce on unseen data. The
last fifth of each training day is held out chronologically instead, which
keeps the attack families in training while giving an honest stopping signal.

Selection is on forecast AUC rather than loss. Dynamics loss keeps falling long
after the forecast stops improving -- the model gets better at predicting
benign traffic, which is most of the data and none of the point.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from foresight.data.flows import build_windows, load_flows
from foresight.model.dataset import (
    TEST_DAYS, TRAIN_DAYS, Normaliser, SequenceSet,
)
from foresight.model.world import WorldModel, dynamics_loss, simulate

OUT = Path("checkpoints/world")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, no sklearn dependency in the training path."""
    if labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    scores, labels, errors = [], [], []
    for history, target, _future, risk in loader:
        history, target = history.to(device), target.to(device)
        nxt, logit, _ = model(history)
        errors.append(dynamics_loss(nxt, target).item())
        scores.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(risk.numpy())
    return (auc(np.concatenate(scores), np.concatenate(labels)),
            float(np.mean(errors)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="60s")
    ap.add_argument("--stride", default="15s")
    ap.add_argument("--graph", action="store_true",
                    help="add host-graph features; measured worse, see docs/RESULTS.md")
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--risk-weight", type=float, default=0.3,
                    help="below 1 on purpose: dynamics must lead")
    ap.add_argument("--rollout-weight", type=float, default=0.5,
                    help="supervises the K-step trajectory, not just one step")
    ap.add_argument("--rollout-steps", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    frame = load_flows()
    train_windows = [build_windows(frame, d, a.window, a.stride, a.graph) for d in TRAIN_DAYS]
    test_windows = [build_windows(frame, d, a.window, a.stride, a.graph) for d in TEST_DAYS]

    # Chronological validation split from the training days. Held out by time
    # rather than at random: adjacent windows are near-duplicates, and a random
    # split would put the same minute on both sides.
    fitted, validation = [], []
    for day in train_windows:
        cut = int(len(day.states) * 0.8)
        head = type(day)(states=day.states[:cut], labels=day.labels[:cut],
                         families=day.families[:cut], times=day.times[:cut],
                         columns=day.columns, day=day.day)
        tail = type(day)(states=day.states[cut:], labels=day.labels[cut:],
                         families=day.families[cut:], times=day.times[cut:],
                         columns=day.columns, day=day.day)
        fitted.append(head); validation.append(tail)

    norm = Normaliser.fit(np.concatenate([w.states for w in fitted]))
    train_set = SequenceSet(fitted, norm, a.length, a.horizon)
    val_set = SequenceSet(validation, norm, a.length, a.horizon)
    test_set = SequenceSet(test_windows, norm, a.length, a.horizon)
    print(f"device {device} | train {len(train_set):,} test {len(test_set):,} "
          f"| features {train_set.history[0].shape[1]}")
    print(f"train days {TRAIN_DAYS}\ntest days  {TEST_DAYS} (unseen attack families)")

    train_loader = DataLoader(train_set, batch_size=a.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    model = WorldModel(n_features=train_set.history[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # The positive class is the minority; without this the risk head learns to
    # answer "no attack" and the AUC comes from the dynamics alone.
    pos = float(np.mean(train_set.risk))
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), device=device)
    risk_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best, best_state, t0 = -1.0, None, time.time()
    for epoch in range(a.epochs):
        model.train()
        totals = np.zeros(2)
        for history, target, future, risk in train_loader:
            history, target = history.to(device), target.to(device)
            future, risk = future.to(device), risk.to(device)
            nxt, logit, _ = model(history)
            d_loss = dynamics_loss(nxt, target)
            r_loss = risk_loss(logit, risk)
            sim = simulate(model, history, a.rollout_steps)
            s_loss = dynamics_loss(sim, future[:, :a.rollout_steps])
            loss = d_loss + a.rollout_weight * s_loss + a.risk_weight * r_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals += [d_loss.item(), r_loss.item()]
        sched.step()
        totals /= len(train_loader)
        val_auc, _ = evaluate(model, val_loader, device)
        test_auc, test_err = evaluate(model, test_loader, device)
        flag = ""
        if val_auc > best:
            best, flag = val_auc, "  <- best"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or flag:
            print(f"  epoch {epoch:>3}  dyn {totals[0]:.4f}  risk {totals[1]:.4f}  "
                  f"| val AUC {val_auc:.3f}  test AUC {test_auc:.3f}"
                  f"  err {test_err:.4f}  ({time.time()-t0:.0f}s){flag}")

    if best_state:
        model.load_state_dict(best_state)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    np.savez(out / "norm.npz", mean=norm.mean, std=norm.std)
    (out / "config.json").write_text(json.dumps(
        {"window": a.window, "stride": a.stride, "graph": a.graph, "length": a.length, "horizon": a.horizon,
         "n_features": int(train_set.history[0].shape[1]),
         "columns": train_windows[0].columns,
         "train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
         "selection": "validation split from training days, 20% chronological",
         "rollout_steps": a.rollout_steps, "rollout_weight": a.rollout_weight,
         "best_val_auc": best}, indent=1))
    final_auc, _ = evaluate(model, test_loader, device)
    print(f"\nselected on validation AUC {best:.3f}")
    print(f"test AUC on unseen attack families: {final_auc:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
