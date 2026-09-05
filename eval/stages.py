"""Evaluate the MITRE stage mapping.

Two evaluations, because they answer different questions and only one of them
is about forecasting.

The day split is the one the forecasting model is judged on: train on the first
three capture days, test on the last two, so the test attacks are families the
model never saw. For stage mapping that split is degenerate rather than hard --
the training days contain only Initial Access and denial of service, so
Reconnaissance, Lateral Movement and Command & Control exist *only* in the test
set. A classifier cannot name a class it was never shown, and the number below
measures that fact, not the method.

Stage identification is not forecasting. It describes a state the model has
already predicted, so it is fairly evaluated on a split that contains every
stage on both sides. The interleaved split cuts each day into blocks and takes 70% of every block
for fitting and the rest for scoring, discarding a gap at each boundary so that
no fitted window shares traffic with a scored one through the sliding overlap.
A single chronological cut will not do: every attack here runs once, for a
scheduled stretch, so one cut puts whole stages on one side of it.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from foresight.data.flows import build_windows, load_flows
from foresight.model.dataset import Normaliser, TEST_DAYS, TRAIN_DAYS
from foresight.model.world import WorldModel
from foresight.predict.stages import (blocked_split, confusion, fit_stage_model,
                                      stage_of)


def rolled_states(model, norm, windows, length: int, horizon: int):
    """The states the classifier will actually be shown, with their families.

    Fitting on observed windows and predicting on rolled-forward ones is a
    distribution shift, and a large one: the rollout regresses toward a mean
    the classifier reads as Command & Control, so stage accuracy fell from
    0.755 on observed states to 0.543 on predicted ones and every stage
    collapsed to the same answer. The fix is to fit on what deployment feeds
    it -- the state the model predicts -- labelled with the family of the
    window that state is predicting.
    """
    states = norm(windows.states)
    out, families = [], []
    with torch.no_grad():
        for t in range(length, len(states) - horizon):
            history = torch.from_numpy(states[t - length:t]).unsqueeze(0)
            roll = model.rollout(history, steps=horizon)
            out.append(norm.invert(roll.states[-1].cpu().numpy()))
            families.append(windows.families[t + horizon])
    return np.array(out), families

DAYS = TRAIN_DAYS + TEST_DAYS


def _table(matrix: np.ndarray, names: list[str], accuracy: float) -> str:
    width = max(len(n) for n in names)
    head = " " * (width + 2) + "".join(f"{n[:11]:>13}" for n in names)
    rows = [f"{n:<{width}}  " + "".join(f"{v:>13}" for v in matrix[i])
            for i, n in enumerate(names)]
    per = []
    for i, n in enumerate(names):
        support = matrix[i].sum()
        recall = matrix[i, i] / support if support else float("nan")
        column = matrix[:, i].sum()
        precision = matrix[i, i] / column if column else float("nan")
        per.append(f"  {n:<{width}}  precision {precision:5.3f}   "
                   f"recall {recall:5.3f}   support {support}")
    return "\n".join([head + "   <- predicted", *rows, "",
                      f"accuracy {accuracy:.3f}", *per])


def main() -> None:
    cfg = json.loads(Path("checkpoints/world/config.json").read_text())
    flows = load_flows()
    gap = max(1, int(pd.Timedelta(cfg["window"]) / pd.Timedelta(cfg["stride"])))

    saved = np.load("checkpoints/world/norm.npz")
    norm = Normaliser(mean=saved["mean"], std=saved["std"])
    model = WorldModel(n_features=cfg["n_features"])
    model.load_state_dict(torch.load("checkpoints/world/model.pt",
                                     map_location="cpu"))
    model.eval()

    per_day = {}
    for day in DAYS:
        w = build_windows(flows, day, cfg["window"], cfg["stride"])
        states, families = rolled_states(model, norm, w, cfg["length"],
                                         cfg["horizon"])
        per_day[day] = (states, families)
        print(f"  rolled {day}: {len(states)} predicted states", flush=True)
    columns = build_windows(flows, DAYS[0], cfg["window"], cfg["stride"]).columns

    print(__doc__)
    print("=" * 72)
    print(f"day split — fit on {TRAIN_DAYS}, score on {TEST_DAYS}")
    xtr = np.vstack([per_day[d][0] for d in TRAIN_DAYS])
    ftr = [f for d in TRAIN_DAYS for f in per_day[d][1]]
    xte = np.vstack([per_day[d][0] for d in TEST_DAYS])
    fte = [f for d in TEST_DAYS for f in per_day[d][1]]
    model = fit_stage_model(xtr, ftr, columns)
    print(f"stages present in training: {model.classes}")
    unseen = sorted({stage_of(f) for f in fte if stage_of(f)} - set(model.classes))
    print(f"stages appearing only at test time: {unseen}")
    matrix, names, acc = confusion(model, xte, fte)
    print(_table(matrix, names, acc))

    print("\n" + "=" * 72)
    print(f"interleaved split — alternating blocks, 70/30 within each, "
          f"{gap}-window gap discarded at every boundary")
    tr_s, tr_f, te_s, te_f = [], [], [], []
    for day in DAYS:
        states, families = per_day[day]
        train, test = blocked_split(len(states), gap)
        tr_s.append(states[train]); tr_f += list(np.array(families)[train])
        te_s.append(states[test]); te_f += list(np.array(families)[test])
    model = fit_stage_model(np.vstack(tr_s), tr_f, columns)
    matrix, names, acc = confusion(model, np.vstack(te_s), te_f)
    print(f"stages present in training: {model.classes}")
    print(_table(matrix, names, acc))

    # Carry the measured per-class precision into the model so the interface
    # can qualify a stage by how often that class is actually right.
    model.precision = {n: (matrix[i, i] / matrix[:, i].sum()
                           if matrix[:, i].sum() else 0.0)
                       for i, n in enumerate(names)}
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/stages.json").write_text(json.dumps({
        "classes": names,
        "matrix": matrix.tolist(),
        "accuracy": acc,
        "chance": 1 / len(names),
        "precision": model.precision,
    }, indent=2))
    Path("checkpoints/world/stages.pkl").write_bytes(pickle.dumps(model))
    print("\nfitted stage model written to checkpoints/world/stages.pkl")


if __name__ == "__main__":
    main()
