"""Benchmark the world model against the required logistic regression baseline.

The statement asks for F1, precision, recall and false positive rate against a
logistic regression trained on the same features, and those are reported. They
are not the headline.

The headline is lead time. The statement's claim is prediction "before
compromise is completed", and F1 does not measure that -- a detector that fires
at the exact window an attack lands scores perfectly and warns nobody. Lead
time asks the question directly: at the moment the model first crosses its
threshold, how many windows remain before the attack actually begins?

Both models are given their own F1-optimal threshold for the classification
metrics, so that comparison is between curves rather than arbitrary cut-offs.
The baseline is also given the same flattened history the world model sees, in
a second variant, so any advantage cannot be dismissed as the world model
simply having more input.

Lead time is measured at a fixed false-alarm budget, not at the F1 threshold.
Measured the other way it is meaningless: a model that fires on almost every
window trivially fires before every attack, and the first version of this
script duly reported a useless baseline with a false positive rate of 0.99 as
having identical lead time to the world model. Warning early is only worth
anything if the warnings are mostly right, so each model is thresholded to the
same false positive rate and asked how early it can speak within that budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from foresight.baseline import LogisticBaseline, best_threshold, metrics
from foresight.data.flows import build_windows, load_flows
from foresight.model.dataset import TEST_DAYS, TRAIN_DAYS, Normaliser, SequenceSet
from foresight.model.world import WorldModel


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray,
                     target_fpr: float) -> float:
    """The lowest threshold whose false positive rate stays within budget.

    Lowest rather than any: among thresholds meeting the budget, the lowest
    fires earliest, which is what lead time is asking about.
    """
    benign = scores[labels == 0]
    if len(benign) == 0:
        return 0.5
    return float(np.quantile(benign, 1.0 - target_fpr))


def lead_times(scores: np.ndarray, origins: list[tuple[str, int]],
               windows_by_day: dict, threshold: float,
               window_seconds: int, _fpr_hint: float = 0.05) -> dict:
    """How far ahead of each attack the model first raises the alarm.

    For every contiguous attack episode in the test days, find the earliest
    window where the score crosses threshold within the preceding ten windows.
    Reported in seconds because a defender's question is "how long have I got",
    not "how many array indices".
    """
    by_day: dict[str, dict[int, float]] = {}
    for score, (day, t) in zip(scores, origins):
        by_day.setdefault(day, {})[t] = float(score)

    leads, missed = [], 0
    for day, day_windows in windows_by_day.items():
        labels = day_windows.labels
        malicious = labels > 0
        # Episode starts: a malicious window whose predecessor was clean.
        starts = [i for i in range(1, len(malicious))
                  if malicious[i] and not malicious[i - 1]]
        for start in starts:
            fired = None
            for back in range(1, 11):
                t = start - back
                score = by_day.get(day, {}).get(t)
                if score is not None and score >= threshold:
                    fired = back          # keep searching for the earliest
            if fired is None:
                missed += 1
            else:
                leads.append(fired * window_seconds)

    # A model that fires at random still warns sometimes, because the search
    # gives it ten independent chances per episode. Without this baseline the
    # warn rate reads as skill when much of it is arithmetic.
    lookback = 10
    return {"episodes": len(leads) + missed, "warned": len(leads),
            "missed": missed,
            "chance_warn_rate": 1.0 - (1.0 - _fpr_hint) ** lookback,
            "median_lead_seconds": float(np.median(leads)) if leads else 0.0,
            "mean_lead_seconds": float(np.mean(leads)) if leads else 0.0,
            "max_lead_seconds": float(np.max(leads)) if leads else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/world")
    ap.add_argument("--out", default="eval/results/benchmark.json")
    a = ap.parse_args()

    ckpt = Path(a.checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())
    frame = load_flows()
    train_windows = [build_windows(frame, d, cfg["window"], cfg.get("stride"), cfg.get("graph", False)) for d in TRAIN_DAYS]
    test_windows = [build_windows(frame, d, cfg["window"], cfg.get("stride"), cfg.get("graph", False)) for d in TEST_DAYS]

    saved = np.load(ckpt / "norm.npz")
    norm = Normaliser(mean=saved["mean"], std=saved["std"])
    gap = cfg.get("gap", 1)
    train_set = SequenceSet(train_windows, norm, cfg["length"], cfg["horizon"], gap)
    test_set = SequenceSet(test_windows, norm, cfg["length"], cfg["horizon"], gap)

    y_train = np.array(train_set.risk)
    y_test = np.array(test_set.risk)
    stride_s = int((cfg.get("stride") or cfg["window"]).rstrip("s"))
    print(f"train {len(y_train):,}  test {len(y_test):,}  "
          f"(test days {TEST_DAYS}, attack families unseen in training)")
    print(f"forecast gap {gap} windows: the label starts {gap * stride_s}s past "
          f"the last observed traffic, base rate {y_test.mean():.3f}")

    # --- world model -------------------------------------------------------
    model = WorldModel(n_features=cfg["n_features"])
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        hist = torch.from_numpy(np.stack(test_set.history))
        world_scores = torch.sigmoid(model(hist)[1]).numpy()

    # --- baselines ---------------------------------------------------------
    last_train = np.stack([h[-1] for h in train_set.history])
    last_test = np.stack([h[-1] for h in test_set.history])
    lr_single = LogisticBaseline().fit(last_train, y_train)
    single_scores = lr_single.predict_proba(last_test)

    flat_train = np.stack([h.reshape(-1) for h in train_set.history])
    flat_test = np.stack([h.reshape(-1) for h in test_set.history])
    lr_hist = LogisticBaseline().fit(flat_train, y_train)
    hist_scores = lr_hist.predict_proba(flat_test)

    runs = {"world model": world_scores,
            "logistic regression (current window)": single_scores,
            "logistic regression (full history)": hist_scores}

    window_seconds = int((cfg.get("stride") or cfg["window"]).rstrip("s"))
    by_day = {w.day: w for w in test_windows}
    report, rows = {}, []
    for name, scores in runs.items():
        thr = best_threshold(scores, y_test)
        m = metrics(scores, y_test, thr)
        # Lead time at a common false-alarm budget, so early warning cannot be
        # bought by simply alarming more often.
        budget = 0.05
        alarm_thr = threshold_at_fpr(scores, y_test, budget)
        realised = metrics(scores, y_test, alarm_thr)["false_positive_rate"]
        lead = lead_times(scores, test_set.origin, by_day, alarm_thr,
                          window_seconds, realised)
        lead["fpr_budget"] = budget
        lead["threshold"] = alarm_thr
        lead["realised_fpr"] = metrics(scores, y_test, alarm_thr)["false_positive_rate"]
        report[name] = {"metrics": m, "lead_time": lead}
        rows.append((name, m, lead))

    print(f"\n{'model':<38}{'F1':>7}{'prec':>7}{'rec':>7}{'FPR':>7}{'AUC':>7}")
    for name, m, _ in rows:
        print(f"{name:<38}{m['f1']:>7.3f}{m['precision']:>7.3f}"
              f"{m['recall']:>7.3f}{m['false_positive_rate']:>7.3f}{m['auc']:>7.3f}")

    print(f"\nlead time at a 5% false-alarm budget "
          f"(search limited to 10 windows = {10*window_seconds}s)")
    print(f"{'model':<38}{'warn rate':>11}{'chance':>9}{'median':>9}")
    for name, _, l in rows:
        rate = l["warned"] / max(l["episodes"], 1)
        print(f"{name:<38}{rate:>10.0%}{l['chance_warn_rate']:>9.0%}"
              f"{l['median_lead_seconds']:>8.0f}s")
    print("\nA warn rate at or below chance means the early warning is an "
          "artefact of\nthe lookback window, not of the model.")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "results": report}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
