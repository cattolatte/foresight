"""Does modelling the host instead of the network recover the quiet attacks?

The network-wide model catches 92% of DDoS windows and 14% of botnet windows at
the same false-alarm rate. The explanation is not difficulty, it is averaging:
during its episode, botnet traffic is 1.17% of the flows on the network and
100% of the flows on the host it runs on. This runs the same experiment with
the host as the unit of modelling and compares per family.

Everything else is held fixed -- same features, same window and stride, same
architecture, same benign-only training with no attack label in the objective,
same matched false-alarm rate -- so the difference is attributable to the unit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from foresight.baseline import LogisticBaseline, metrics
from foresight.data.flows import load_flows
from foresight.data.hosts import build_host_windows, host_sequences
from foresight.model.dataset import Normaliser, TEST_DAYS, TRAIN_DAYS
from foresight.model.world import WorldModel

WINDOW, STRIDE, LENGTH, HORIZON, GAP = "60s", "15s", 12, 6, 4


def train_benign(history, target, n_features, epochs=40, seed=0):
    torch.manual_seed(seed)
    model = WorldModel(n_features=n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(history),
                                      torch.from_numpy(target)),
                        batch_size=256, shuffle=True)
    loss_fn = torch.nn.HuberLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for h, t in loader:
            opt.zero_grad()
            predicted, _, _ = model(h)
            loss = loss_fn(predicted, t)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(h)
        if epoch % 10 == 9:
            print(f"    epoch {epoch + 1:3d}  dynamics loss "
                  f"{total / len(history):.4f}", flush=True)
    return model.eval()


def signal_to_noise(flows) -> dict:
    """An attack's share of the flows, network-wide against on its own host.

    This is the measurement the whole per-host argument rests on, so it is
    computed here rather than quoted: for each family, take the window of time
    its episode spans and compare its share of all flows in that window with
    its share of the flows touching the host it appears on most.
    """
    import pandas as pd

    out = {}
    for family in flows.loc[flows["attack_label"].ne("BENIGN"),
                            "attack_label"].unique():
        attack = flows[flows["attack_label"] == family]
        day = attack["ts"].dt.date.astype(str).mode()[0]
        same_day = flows[flows["day"] == day]
        during = same_day[(same_day["ts"] >= attack["ts"].min())
                          & (same_day["ts"] <= attack["ts"].max())]
        if not len(during):
            continue
        network = float((during["attack_label"] == family).mean())
        peers = pd.concat([attack["source_ip"], attack["destination_ip"]])
        busiest = peers.value_counts().index[0]
        on_host = during[(during["source_ip"] == busiest)
                         | (during["destination_ip"] == busiest)]
        host = float((on_host["attack_label"] == family).mean()) if len(on_host) else 0.0
        out[family] = {"network": network, "host": host,
                       "gain": host / network if network else 0.0}
    return out


def main() -> None:
    flows = load_flows()
    print(__doc__)
    print("=" * 74)
    train_days = [build_host_windows(flows, d, WINDOW, STRIDE)
                  for d in TRAIN_DAYS]
    test_days = [build_host_windows(flows, d, WINDOW, STRIDE)
                 for d in TEST_DAYS]
    norm = Normaliser.fit(np.vstack([w.states for w in train_days]))

    tr = [host_sequences(w, norm, LENGTH, HORIZON, GAP) for w in train_days]
    te = [host_sequences(w, norm, LENGTH, HORIZON, GAP) for w in test_days]
    h_tr = np.concatenate([x[0] for x in tr]); t_tr = np.concatenate([x[1] for x in tr])
    y_tr = np.concatenate([x[2] for x in tr])
    h_te = np.concatenate([x[0] for x in te]); t_te = np.concatenate([x[1] for x in te])
    y_te = np.concatenate([x[2] for x in te]); f_te = np.concatenate([x[3] for x in te])

    # Benign-only: no window whose history or target carries an attack label.
    benign = y_tr == 0
    print(f"per-host sequences: train {len(y_tr):,} ({benign.sum():,} benign), "
          f"test {len(y_te):,}   base rate {y_te.mean():.4f}")
    model = train_benign(h_tr[benign], t_tr[benign], h_tr.shape[-1])

    with torch.no_grad():
        predicted, _, _ = model(torch.from_numpy(h_te))
    surprise = ((predicted - torch.from_numpy(t_te)) ** 2).mean(1).numpy()

    k = train_days[0].columns.index("h_flow_count")
    lr = LogisticBaseline().fit(np.array([[h[-1][k]] for h in h_tr]), y_tr)
    volume = lr.predict_proba(np.array([[h[-1][k]] for h in h_te]))

    print(f"\noverall   surprise AUC {metrics(surprise, y_te)['auc']:.3f}   "
          f"host flow volume AUC {metrics(volume, y_te)['auc']:.3f}")

    def threshold(score, rate=0.10):
        return np.quantile(score[y_te == 0], 1 - rate)

    print("\nDetection rate per family at a matched 10% false-alarm rate")
    print("(network-wide figures from eval/surprise.py in brackets)")
    reference = {"Bot": (0.14, 0.19), "DDoS": (0.92, 0.89),
                 "PortScan": (0.33, 0.36), "Infiltration": (0.62, 0.69),
                 "Web Attack – Brute Force": (0.11, 0.30),
                 "Web Attack – XSS": (0.09, 0.04)}
    print(f"  {'family':<28}{'surprise':>20}{'host volume':>20}{'windows':>9}")
    out = {}
    for name in sorted(set(f_te.tolist()) - {"BENIGN"}):
        sel = f_te == name
        if sel.sum() < 8:
            continue
        rs = float((surprise[sel] >= threshold(surprise)).mean())
        rv = float((volume[sel] >= threshold(volume)).mean())
        was = reference.get(name)
        a = f"{rs:.2f}" + (f"  [{was[0]:.2f}]" if was else "")
        b = f"{rv:.2f}" + (f"  [{was[1]:.2f}]" if was else "")
        print(f"  {name[:26]:<28}{a:>20}{b:>20}{sel.sum():>9}")
        out[name] = {"surprise": rs, "volume": rv, "windows": int(sel.sum())}

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/perhost.json").write_text(json.dumps(out, indent=2))
    Path("eval/results/snr.json").write_text(
        json.dumps(signal_to_noise(flows), indent=2))
    torch.save(model.state_dict(), "checkpoints/world/host_dynamics.pt")
    np.savez("checkpoints/world/host_norm.npz", mean=norm.mean, std=norm.std)
    print("\nwrote eval/results/perhost.json")


if __name__ == "__main__":
    main()
