"""Load CIC-IDS2017 flow records and turn them into a sequence of network states.

The problem statement asks for a model of how network state evolves, so the
unit of modelling is not a flow. It is the network as observed over a time
window: what the flags looked like, which ports were active, how traffic was
distributed across hosts. A sequence of those windows is what a world model
learns transitions over.

Two things about this data cost time and are worth recording.

The timestamps are in two different formats in the same column -- benign rows
read "03/07/2017 08:55:58" and attack rows "4/7/2017 10:30", day-first and with
no seconds. Parsing with pandas defaults silently returns NaT for one of them,
which makes every attack look as though it had no timestamp and every hour look
0% malicious. The dataset is fine; the default parse is not.

The five capture days are the natural split. Monday is entirely benign, and
each later day introduces different attack families, which is what makes
"generalise to unseen attack patterns" testable rather than a claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.data.graph import add_graph_features
from foresight.data.packets import PACKET_FEATURES

DEFAULT_FLOWS = Path("data/flows.parquet")

# The flag counters the statement names explicitly, plus the timing and volume
# statistics that distinguish a slow scan from a flood.
FLAG_COLUMNS = [
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
]
TIMING_COLUMNS = [
    "Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Bwd IAT Mean", "Active Mean", "Idle Mean",
]
VOLUME_COLUMNS = [
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "Down/Up Ratio",
    "Average Packet Size", "Packet Length Std", "Packet Length Variance",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]


@dataclass
class Windows:
    """A day's traffic as an ordered sequence of network states.

    `states` is (T, F): one feature vector per time window.
    `labels` is (T,): the fraction of flows in that window that were malicious.
    `families` is (T,) of the dominant attack family, for stage mapping.
    """

    states: np.ndarray
    labels: np.ndarray
    families: list[str]
    times: pd.Series
    columns: list[str]
    day: str

    def __len__(self) -> int:
        return len(self.states)


def load_flows(path: str | Path = DEFAULT_FLOWS) -> pd.DataFrame:
    """Read the flow table with timestamps parsed correctly.

    `dayfirst` and `format="mixed"` are both required. Without them roughly
    four in five rows silently become NaT, and every attack row is among them.
    """
    frame = pd.read_parquet(path)
    frame["ts"] = pd.to_datetime(frame["Timestamp"], dayfirst=True,
                                 format="mixed", errors="coerce")
    if frame["ts"].isna().any():
        raise ValueError(f"{frame['ts'].isna().sum()} timestamps failed to parse")
    frame["day"] = frame["ts"].dt.date.astype(str)
    return frame.sort_values("ts", kind="stable").reset_index(drop=True)


def _port_entropy(ports: pd.Series) -> float:
    """How spread out destination ports are within a window.

    A scan touches many ports once each and scores high; ordinary traffic
    concentrates on a few services and scores low. This is the feature that
    distinguishes reconnaissance from volume, and volume alone will not.
    """
    if ports.empty:
        return 0.0
    counts = ports.value_counts(normalize=True).to_numpy()
    return float(-(counts * np.log2(counts + 1e-12)).sum())


def build_windows(frame: pd.DataFrame, day: str, window: str = "60s",
                  stride: str | None = None, graph: bool = False,
                  packets: pd.DataFrame | None = None) -> Windows:
    """Aggregate one day's flows into fixed time windows.

    Aggregation, not sampling: a window is a summary of everything observed in
    that interval, which is the "current observed network state" the statement
    describes. Windows with no traffic are kept as zero states so the sequence
    stays evenly spaced in time -- a model of dynamics needs a constant step.

    `stride` slides the aggregation window rather than tiling it. The window
    length is fixed by the clock, but the step between windows need not be: a
    60s window every 15s gives four times the training sequences over the same
    traffic, which matters because three capture days at one window a minute is
    only about two thousand sequences for a model with three hundred thousand
    parameters. The windows overlap, so the *split* must stay chronological --
    overlapping windows either side of a random split would leak.

    Sixty seconds is a floor imposed by the data, not a tuning choice. The
    attack rows carry minute-resolution timestamps -- "4/7/2017 10:30", no
    seconds -- so every one of them lands exactly on a minute boundary. At 30s,
    two thirds of the windows are empty by construction and the learned
    dynamics faithfully reproduce an alternation that is an artefact of the
    clock rather than of the network. Anything finer than a minute is measuring
    the timestamp format.
    """
    sub = frame[frame["day"] == day]
    if sub.empty:
        raise ValueError(f"no flows for {day}")

    numeric = [c for c in FLAG_COLUMNS + TIMING_COLUMNS + VOLUME_COLUMNS
               if c in sub.columns]
    indexed = sub.set_index("ts")
    step = stride or window
    if stride is None:
        grouped = indexed.resample(window)
    else:
        # Overlapping windows: resample at the stride, then aggregate each
        # point over the full window length via a rolling pass.
        grouped = indexed.resample(stride)

    stats = grouped[numeric].mean()
    stats["flow_count"] = grouped.size()
    stats["unique_dst_ports"] = grouped["destination_port"].nunique()
    stats["unique_dst_ips"] = grouped["destination_ip"].nunique()
    stats["unique_src_ips"] = grouped["source_ip"].nunique()
    stats["dst_port_entropy"] = grouped["destination_port"].apply(_port_entropy)
    # Fan-out: one host touching many is reconnaissance; many touching one is a flood.
    stats["fanout"] = stats["unique_dst_ips"] / stats["unique_src_ips"].clip(lower=1)

    # Topology, off by default. The features separate a sweep from a flood
    # cleanly in isolation, and they were made scale-invariant so they would
    # describe shape rather than this network's size. They still cost accuracy
    # on held-out days: test AUC 0.761 without them, 0.674 with raw degrees,
    # 0.606 with normalised ones, against a validation score above 0.9 in every
    # case. The gap is the finding -- they let the model fit the topology of
    # the days it trained on. Kept because the ablation is worth showing and
    # because a corpus with more distinct networks would likely reverse it.
    if graph:
        stats = stats.join(add_graph_features(indexed, step), how="left")

    # Packet-level features, averaged over the flows in the window that have
    # packet data. The coverage rate itself is deliberately *not* a feature:
    # which flows have packet data depends on which files the dataset publishes
    # and correlates hard with attack family -- 99.6% of PortScan flows are
    # covered against 0% of the web attacks -- so a model given the coverage
    # rate would be reading the release process rather than the network.
    if packets is not None:
        joined = indexed.join(packets, on="flow_id")
        present = [c for c in PACKET_FEATURES if c in joined.columns]
        stats = stats.join(joined.resample(step)[present].mean(), how="left")

    # Labels must be binned on the same grid as the states. Resampling them at
    # the window while the states are resampled at the stride produced arrays
    # of different lengths, four to one, and the mismatch surfaced as an
    # unrelated crash deep inside the dataset builder.
    malicious = indexed["attack_label"].ne("BENIGN")
    labels = malicious.resample(step).mean().fillna(0.0)

    # The dominant family must be taken over the same span as the label.
    #
    # Taking it per stride bucket while the label was rolled over the window
    # left three quarters of the windows labelled malicious reporting a family
    # of "BENIGN": the attack was inside the 60s window but not inside the
    # trailing 15s bucket the family was read from. Nothing crashed -- the
    # ground-truth row in the interface and every per-family breakdown simply
    # attributed most attack windows to benign traffic.
    #
    # Counting families per bucket and summing over the span keeps the two in
    # step by construction, whatever the window and stride are set to.
    attacks = indexed.loc[indexed["attack_label"].ne("BENIGN"), "attack_label"]
    if len(attacks):
        counts = (attacks.groupby([pd.Grouper(freq=step), attacks])
                  .size().unstack(fill_value=0)
                  .reindex(stats.index, fill_value=0))
    else:
        counts = pd.DataFrame(index=stats.index)

    if stride is not None:
        span = max(1, int(pd.Timedelta(window) / pd.Timedelta(stride)))
        stats = stats.rolling(span, min_periods=1).mean()
        labels = labels.rolling(span, min_periods=1).max()
        if not counts.empty:
            counts = counts.rolling(span, min_periods=1).sum()

    if counts.empty:
        families = ["BENIGN"] * len(stats)
    else:
        families = np.where(counts.to_numpy().sum(axis=1) > 0,
                            counts.idxmax(axis=1), "BENIGN").tolist()

    stats = stats.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return Windows(states=stats.to_numpy(dtype=np.float32),
                   labels=labels.to_numpy(dtype=np.float32),
                   families=families, times=stats.index.to_series(),
                   columns=list(stats.columns), day=day)
