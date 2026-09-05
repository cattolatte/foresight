"""Shortcut baselines: what this benchmark can be beaten by without a model.

A number is only evidence if something dumber cannot produce it. Every score in
this file comes from one feature, or from no features at all, and each is a way
of scoring well on the forecasting benchmark without learning anything about how
a network evolves. They are the bar, and they are reported alongside the model
because two of them clear what the world model achieves.

Persistence is the important one. Attack episodes in this capture run for
minutes -- median 60s on Thursday, 240s on Friday, up to seventy minutes --
against a horizon of 90 seconds. "Is there an attack in the next 90 seconds" is
therefore very nearly "is there an attack right now", and simply repeating the
ground-truth label of the last observed window scores higher than the model.
Persistence is not a deployable system; it needs the answer in order to give it.
It is a measurement of how much of the task is autocorrelation, and the answer
is most of it.

Protocol composition is the second. Every attack family in CIC-IDS2017 is
TCP-based, and 44% of benign flows are not TCP, so the fraction of TCP flows in
a window separates the classes almost as well as the model does. A system
leaning on that has learned this capture's composition, not attack behaviour,
and would be blind to a UDP-based attack -- DNS amplification, NTP reflection --
which is precisely the generalisation the statement asks for.

The conclusion drawn from this file is in RESULTS.md: the headline benchmark is
too easy to be the headline, and the onset task in eval/surprise.py replaces it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foresight.baseline import LogisticBaseline, metrics
from foresight.data.flows import build_windows, load_flows
from foresight.model.dataset import (SequenceSet, Normaliser, TEST_DAYS,
                                     TRAIN_DAYS)

STRIDE = "15s"


def _window_series(frame, day: str, column: str, span: int = 4) -> np.ndarray:
    sub = frame[frame["day"] == day].set_index("ts")
    return (sub[column].resample(STRIDE).mean()
            .rolling(span, min_periods=1).mean().fillna(0.0).to_numpy())


def main() -> None:
    cfg = json.loads(Path("checkpoints/world/config.json").read_text())
    length, horizon = cfg["length"], cfg["horizon"]
    gap = cfg.get("gap", 4)
    frame = load_flows()
    frame["_tcp"] = (frame["protocol"] == "tcp").astype(float)
    frame["_sentinel"] = (frame["Init_Win_bytes_forward"] < 0).astype(float)

    train_windows = [build_windows(frame, d, cfg["window"], cfg["stride"])
                     for d in TRAIN_DAYS]
    test_windows = [build_windows(frame, d, cfg["window"], cfg["stride"])
                    for d in TEST_DAYS]
    saved = np.load("checkpoints/world/norm.npz")
    norm = Normaliser(mean=saved["mean"], std=saved["std"])
    train_set = SequenceSet(train_windows, norm, length, horizon, gap)
    test_set = SequenceSet(test_windows, norm, length, horizon, gap)
    y_train = np.array(train_set.risk)
    y_test = np.array(test_set.risk)

    print(__doc__)
    print("=" * 74)
    print(f"test windows {len(y_test):,}   base rate {y_test.mean():.3f}   "
          f"forecast gap {gap} windows\n")
    rows = []

    # 1. Persistence: the ground-truth label of the last observed window.
    truth = {(w.day, i): (1.0 if w.labels[i] > 0 else 0.0)
             for w in test_windows for i in range(len(w.labels))}
    persistence = np.array([truth[(d, t - 1)] for d, t in test_set.origin])
    rows.append(("persistence — last observed label, no features", persistence))

    # 2. Always positive: what the F1 column looks like with no information.
    rows.append(("always predict attack", np.ones(len(y_test))))

    # 3-4. One raw feature each, fitted as a logistic regression.
    for column, label in [("_tcp", "fraction of flows that are TCP"),
                          ("_sentinel", "fraction with the -1 window sentinel")]:
        series = {d: _window_series(frame, d, column)
                  for d in TRAIN_DAYS + TEST_DAYS}
        x_tr = np.array([[series[d][t - 1]] for d, t in train_set.origin])
        x_te = np.array([[series[d][t - 1]] for d, t in test_set.origin])
        model = LogisticBaseline().fit(x_tr, y_train)
        rows.append((f"one feature — {label}", model.predict_proba(x_te)))

    # 5. Flow volume alone, from the state vector the model itself sees.
    columns = train_windows[0].columns
    k = columns.index("flow_count")
    x_tr = np.array([[h[-1][k]] for h in train_set.history])
    x_te = np.array([[h[-1][k]] for h in test_set.history])
    model = LogisticBaseline().fit(x_tr, y_train)
    rows.append(("one feature — flow count", model.predict_proba(x_te)))

    print(f"  {'baseline':<52}{'AUC':>7}{'F1':>7}")
    results = {}
    for label, score in rows:
        m = metrics(score, y_test)
        results[label] = {"auc": m["auc"], "f1": m["f1"]}
        print(f"  {label:<52}{m['auc']:>7.3f}{m['f1']:>7.3f}")
    print(f"\n  {'world model, for comparison':<52}{0.783:>7.3f}{0.557:>7.3f}")

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/shortcuts.json").write_text(json.dumps(results, indent=2))
    print("\nwrote eval/results/shortcuts.json")


if __name__ == "__main__":
    main()
