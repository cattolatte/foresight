"""Per-host state sequences, instead of one sequence for the whole network.

The network-wide windows this project started with aggregate every flow in a
minute into one vector. That is defensible for a flood and destructive for
everything else, and the cost is measurable: during its own episode, botnet
traffic is 1.17% of the flows on the network and 100% of the flows on the host
it runs on. Aggregation costs 85x of signal-to-noise on that family, 19x on web
brute force, 11x on cross-site scripting -- and 1.1x on DDoS.

That ratio explains the results exactly. At a matched 10% false-alarm rate the
network-wide model catches 92% of DDoS windows and 14% of botnet windows. It is
not that the quiet families are intrinsically harder; they are being averaged
against two and a half thousand flows a minute of unrelated traffic.

So the unit of modelling here is the host. Each internal address gets its own
sequence of states, built from the flows it sent or received, and the model
learns how *a host* behaves rather than how the site behaves. This also fixes
the other bottleneck: network-wide, three training days yield 83 onset-positive
windows, which is not enough to fit anything; per host it is fifteen times the
sequences.

Host identity is deliberately not a feature. Attacks in this capture concentrate
on one victim -- 192.168.10.50 carries 83% of all attack flows -- so a model
given the address would learn which machine is the target of this particular
capture and nothing transferable. The features describe behaviour only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from foresight.data.flows import (FLAG_COLUMNS, TIMING_COLUMNS, VOLUME_COLUMNS,
                                  _port_entropy)

INTERNAL = "192.168.10."

HOST_FEATURES = ["h_flow_count", "h_out_frac", "h_peers", "h_ports",
                 "h_port_entropy", "h_peer_entropy", "h_new_peer_rate"]


@dataclass
class HostWindows:
    """One day of traffic as (host, time) states."""

    states: np.ndarray            # (H*T, F)
    labels: np.ndarray            # (H*T,)
    families: list[str]
    hosts: np.ndarray             # (H*T,) address, for grouping only
    times: np.ndarray
    columns: list[str]
    day: str


def _entropy(counts: np.ndarray) -> float:
    if counts.sum() <= 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def build_host_windows(frame: pd.DataFrame, day: str, window: str = "60s",
                       stride: str | None = None) -> HostWindows:
    """Aggregate one day into a state per (internal host, time window)."""
    sub = frame[frame["day"] == day]
    if sub.empty:
        raise ValueError(f"no flows for {day}")

    # Each flow is attributed to both endpoints that are internal, so a host's
    # state covers what it sent and what it received. `out` records direction
    # so the two are distinguishable inside the window.
    parts = []
    for side, peer, out in [("source_ip", "destination_ip", 1.0),
                            ("destination_ip", "source_ip", 0.0)]:
        m = sub[side].str.startswith(INTERNAL)
        piece = sub.loc[m].copy()
        piece["host"] = piece[side]
        piece["peer"] = piece[peer]
        piece["out"] = out
        parts.append(piece)
    long = pd.concat(parts, ignore_index=True)

    numeric = [c for c in FLAG_COLUMNS + TIMING_COLUMNS + VOLUME_COLUMNS
               if c in long.columns]
    step = stride or window
    long = long.set_index("ts")
    grouper = [pd.Grouper(freq=step), "host"]

    stats = long.groupby(grouper)[numeric].mean()
    g = long.groupby(grouper)
    stats["h_flow_count"] = g.size()
    stats["h_out_frac"] = g["out"].mean()
    stats["h_peers"] = g["peer"].nunique()
    stats["h_ports"] = g["destination_port"].nunique()
    stats["h_port_entropy"] = g["destination_port"].apply(_port_entropy)
    stats["h_peer_entropy"] = g["peer"].apply(
        lambda s: _entropy(s.value_counts().to_numpy()))

    malicious = g["attack_label"].apply(lambda s: float(s.ne("BENIGN").mean()))

    def dominant(series: pd.Series) -> str:
        attacks = series[series != "BENIGN"]
        return attacks.value_counts().idxmax() if len(attacks) else "BENIGN"

    families = g["attack_label"].apply(dominant)

    # Reindex onto the full (time x host) grid so every host has an evenly
    # spaced sequence; a host that sent nothing in a window is a zero state,
    # not a missing row, or the dynamics would learn a ragged clock.
    times = pd.date_range(stats.index.get_level_values(0).min(),
                          stats.index.get_level_values(0).max(), freq=step)
    hosts = sorted(stats.index.get_level_values(1).unique())
    grid = pd.MultiIndex.from_product([times, hosts], names=["ts", "host"])
    stats = stats.reindex(grid).fillna(0.0)
    malicious = malicious.reindex(grid).fillna(0.0)
    families = families.reindex(grid).fillna("BENIGN")

    # New peers this window relative to the host's previous window: the
    # clearest behavioural mark of a host beginning to reach out.
    seen = g["peer"].apply(lambda s: frozenset(s)).reindex(grid)
    seen = seen.where(seen.notna(), frozenset())
    rates = []
    previous: dict[str, frozenset] = {}
    for (_, host), peers in seen.items():
        before = previous.get(host, frozenset())
        rates.append(len(peers - before) / max(len(peers), 1))
        previous[host] = peers
    stats["h_new_peer_rate"] = rates

    stats = stats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return HostWindows(
        states=stats.to_numpy(dtype=np.float32),
        labels=malicious.to_numpy(dtype=np.float32),
        families=list(families),
        hosts=np.array(stats.index.get_level_values(1)),
        times=np.array(stats.index.get_level_values(0)),
        columns=list(stats.columns), day=day)


def host_sequences(windows: HostWindows, norm, length: int, horizon: int,
                   gap: int):
    """(history, target, label, family) per host, in time order."""
    states = norm(windows.states)
    history, target, risk, family, origin = [], [], [], [], []
    for host in np.unique(windows.hosts):
        sel = np.flatnonzero(windows.hosts == host)
        order = sel[np.argsort(windows.times[sel])]
        s = states[order]
        lab = windows.labels[order]
        fam = [windows.families[i] for i in order]
        for t in range(length, len(s) - horizon - gap):
            history.append(s[t - length:t])
            target.append(s[t])
            future = lab[t + gap:t + gap + horizon]
            risk.append(float(future.max() > 0))
            hit = [fam[t + gap + j] for j in range(horizon)
                   if fam[t + gap + j] != "BENIGN"]
            family.append(hit[0] if hit else "BENIGN")
            origin.append((windows.day, host, t))
    return (np.stack(history), np.stack(target), np.array(risk),
            np.array(family), origin)
