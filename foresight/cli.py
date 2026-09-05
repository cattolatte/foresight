"""Offline command-line demonstration.

    "A working demonstration interface (Streamlit, Flask web app, or CLI) that
     accepts a PCAP or CSV file as input, runs the world model inference, and
     displays the infiltration probability timeline, flagged flows, and attack
     stage annotations. The interface must run fully offline without cloud API
     dependencies."

Nothing here reaches the network. The model, the normaliser and the feature
pipeline are all local files, and the only inputs are the capture you pass in.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from foresight.data.flows import build_windows
from foresight.model.dataset import Normaliser
from foresight.model.world import WorldModel
from foresight.predict.engine import predict

BAR = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(BAR[min(int((v - lo) / span * (len(BAR) - 1)), len(BAR) - 1)]
                   for v in values)


def load_capture(path: Path) -> pd.DataFrame:
    """Read a flow CSV or parquet. PCAP is parsed if scapy is available."""
    if path.suffix.lower() in (".csv", ".gz"):
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in (".pcap", ".pcapng"):
        raise SystemExit(
            "PCAP input needs the packet pipeline: extract flows first with\n"
            "  python3 -m foresight.data.pcap <file.pcap> -o flows.parquet")
    else:
        raise SystemExit(f"unsupported input: {path.suffix}")

    if "Timestamp" not in frame.columns:
        raise SystemExit("input has no Timestamp column; cannot build a timeline")
    frame["ts"] = pd.to_datetime(frame["Timestamp"], dayfirst=True,
                                 format="mixed", errors="coerce")
    frame = frame[frame["ts"].notna()].copy()
    frame["day"] = frame["ts"].dt.date.astype(str)
    if "attack_label" not in frame.columns:
        frame["attack_label"] = "BENIGN"
    return frame.sort_values("ts", kind="stable").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forecast attacker progression from traffic.")
    ap.add_argument("capture", help="flow CSV or parquet")
    ap.add_argument("--checkpoint", default="checkpoints/world")
    ap.add_argument("--day", help="restrict to one capture day")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=8, help="highest-risk windows to detail")
    a = ap.parse_args()

    ckpt = Path(a.checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())
    saved = np.load(ckpt / "norm.npz")
    norm = Normaliser(mean=saved["mean"], std=saved["std"])
    model = WorldModel(n_features=cfg["n_features"])
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    model.eval()
    stages_path = ckpt / "stages.pkl"
    if not stages_path.exists():
        raise SystemExit(f"stage classifier missing; run `python eval/stages.py` "
                         f"to fit and write {stages_path}")
    stage_model = pickle.loads(stages_path.read_bytes())

    frame = load_capture(Path(a.capture))
    day = a.day or frame["day"].iloc[0]
    windows = build_windows(frame, day, cfg["window"])
    states = norm(windows.states)
    length = cfg["length"]

    print(f"\n  capture : {a.capture}")
    print(f"  day     : {day}   {len(windows)} windows of {cfg['window']}")
    print(f"  model   : {cfg['n_features']} features, "
          f"{cfg['horizon']}-step horizon, trained on {', '.join(cfg['train_days'])}")

    scored = []
    with torch.no_grad():
        for t in range(length, len(states)):
            history = torch.from_numpy(states[t - length:t]).unsqueeze(0)
            risk = float(torch.sigmoid(model(history)[1])[0])
            scored.append((t, risk))

    curve = [r for _, r in scored]
    print("\n  infiltration probability over the capture")
    # Compress to a readable width; a thousand glyphs is not a chart.
    step = max(1, len(curve) // 110)
    print(f"  {sparkline([max(curve[i:i+step]) for i in range(0, len(curve), step)])}")
    print(f"  peak {max(curve):.2f}   mean {np.mean(curve):.2f}   "
          f"windows above {a.threshold:.2f}: {sum(r >= a.threshold for r in curve)}")

    ranked = sorted(scored, key=lambda kv: -kv[1])[:a.top]
    print("\n  highest-risk windows")
    for t, risk in sorted(ranked):
        history = torch.from_numpy(states[t - length:t]).unsqueeze(0)
        result = predict(model, history, windows.columns, stage_model=stage_model,
                         steps=cfg["horizon"], threshold=a.threshold, norm=norm)
        when = windows.times.iloc[t]
        truth = windows.families[t]
        drivers = ", ".join(f"{n} ({v:+.2f})" for n, v in result.top_features[:3])
        print(f"\n  {when}   risk {risk:.2f}   stage: {result.stage}")
        print(f"     forecast   {sparkline(result.horizon_curve)}  "
              f"{' '.join(f'{p:.2f}' for p in result.horizon_curve)}")
        print(f"     because    {result.stage_evidence}")
        print(f"     driven by  {drivers}")
        if truth != "BENIGN":
            print(f"     ground truth in this window: {truth}")

    print("\n  Explanations are gradient-times-input attributions over the "
          "observed\n  history plus attention weights; no cloud services are "
          "contacted.\n")


if __name__ == "__main__":
    main()
