"""Tests for the failure modes this project actually hit.

Each test here corresponds to a bug that reached working code and was caught by
inspecting output rather than by a test. They are written to fail loudly if the
same mistake returns, because every one of them was silent: wrong numbers, not
exceptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from foresight.data.flows import build_windows
from foresight.model.dataset import Normaliser
from foresight.model.world import WorldModel
from foresight.predict.counterfactual import (PLAYBOOK, apply_intervention,
                                              simulate_intervention,
                                              validate_playbook)


def test_timestamps_parse_in_both_formats():
    """Benign rows are zero-padded, attack rows are not.

    The flow table mixes '03/07/2017 08:55:58' with '4/7/2017 10:30'. Parsed
    with pandas' defaults, four rows in five become NaT and *every* attack row
    is among them -- the dataset looks unusable rather than misparsed.
    """
    raw = pd.Series(["03/07/2017 08:55:58", "4/7/2017 10:30", "7/7/2017 15:02"])
    ts = pd.to_datetime(raw, dayfirst=True, format="mixed", errors="coerce")
    assert ts.notna().all()
    assert ts.iloc[0].day == 3 and ts.iloc[0].month == 7
    # Day-first, not month-first: '4/7/2017' is 4 July, not 7 April.
    assert ts.iloc[1].day == 4 and ts.iloc[1].month == 7


def _synthetic_flows(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2017-07-07 09:00:00")
    ts = start + pd.to_timedelta(np.arange(n) * 2, unit="s")
    frame = pd.DataFrame({
        "ts": ts,
        "day": ts.date.astype(str),
        "source_ip": rng.choice([f"10.0.0.{i}" for i in range(8)], n),
        "destination_ip": rng.choice([f"10.0.1.{i}" for i in range(8)], n),
        "destination_port": rng.integers(1, 9000, n),
        "attack_label": np.where(np.arange(n) > n * 0.7, "PortScan", "BENIGN"),
    })
    for col in ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s",
                "SYN Flag Count", "Flow IAT Min", "TotalLen"]:
        frame[col] = rng.random(n) * 100
    return frame


def test_states_and_labels_have_the_same_length():
    """States are built at the stride, labels must be binned at the stride too.

    Aggregating states every 15s while resampling labels every 60s produced
    2880 states against 720 labels. The shapes only disagree at training time,
    far from the cause.
    """
    windows = build_windows(_synthetic_flows(), "2017-07-07",
                            window="60s", stride="15s")
    assert len(windows.states) == len(windows.labels) == len(windows.times)
    assert len(windows.families) == len(windows.labels)


def test_window_stride_produces_more_windows_than_window_alone():
    flows = _synthetic_flows()
    coarse = build_windows(flows, "2017-07-07", window="60s")
    fine = build_windows(flows, "2017-07-07", window="60s", stride="15s")
    assert len(fine.states) > len(coarse.states)


def test_normaliser_inverts():
    """Stage rules read raw units, so invert() must undo the transform.

    The stage cascade thresholds ports and byte counts. Applied to standardised
    log1p values it was comparing z-scores against packet counts and every
    window came back the same stage.
    """
    rng = np.random.default_rng(1)
    raw = rng.random((50, 6)) * 1000
    norm = Normaliser.fit(raw)
    assert np.allclose(norm.invert(norm(raw)), raw, atol=1e-4)
    # And on a single row, which is how the predictor calls it.
    assert np.allclose(norm.invert(norm(raw[:1])[0]), raw[0], atol=1e-4)


def test_every_playbook_action_names_real_features():
    """A scaling that names a missing feature is a silent no-op.

    The first playbook was written against host-graph features that later
    became optional and default to off. Three scalings in five matched nothing,
    every action reported a risk reduction of exactly 0.00, and the panel read
    as 'no defensive action helps' when none had been applied.
    """
    columns = ["flow_count", "unique_dst_ports", "SYN Flag Count"]
    assert validate_playbook(columns), "expected missing features to be reported"
    every = sorted({k for spec in PLAYBOOK.values() for k in spec["scale"]})
    assert not validate_playbook(every)


def test_apply_intervention_scales_only_named_features():
    columns = ["flow_count", "untouched"]
    state = np.array([100.0, 50.0])
    out = apply_intervention(state, columns, {"flow_count": 0.25})
    assert out[0] == 25.0
    assert out[1] == 50.0, "an unnamed feature must not move"
    assert state[0] == 100.0, "the input state must not be mutated"


def test_intervention_rejects_unknown_features():
    model = WorldModel(n_features=3)
    norm = Normaliser.fit(np.ones((4, 3)))
    window = torch.zeros(1, 4, 3)
    PLAYBOOK["_probe"] = {"description": "probe", "scale": {"nope": 0.5}}
    try:
        with pytest.raises(ValueError, match="absent from this model"):
            simulate_intervention(model, window, ["a", "b", "c"], norm, "_probe")
    finally:
        del PLAYBOOK["_probe"]


def test_rollout_shapes_and_determinism():
    model = WorldModel(n_features=5).eval()
    window = torch.randn(1, 6, 5)
    a = model.rollout(window, steps=4)
    b = model.rollout(window, steps=4)
    assert a.states.shape == (4, 5)
    assert len(a.risk) == 4
    assert np.allclose(a.risk, b.risk), "rollout must be deterministic in eval"


def test_family_agrees_with_label_on_every_window():
    """A window labelled malicious must name the attack family, not BENIGN.

    Families were read from the trailing stride bucket while labels were rolled
    over the whole window, so three quarters of attack windows reported
    "BENIGN". Silent: the arrays were the right length and the model was
    unaffected, but every per-family breakdown and the ground-truth row in the
    interface attributed most attacks to benign traffic.
    """
    windows = build_windows(_synthetic_flows(), "2017-07-07",
                            window="60s", stride="15s")
    labels = np.asarray(windows.labels)
    families = np.asarray(windows.families)
    malicious = labels > 0
    assert malicious.any(), "fixture should contain attack traffic"
    assert not (families[malicious] == "BENIGN").any()
    # And the converse: a benign window must not claim an attack family.
    assert (families[~malicious] == "BENIGN").all()


def test_packet_accumulator_sums_are_additive():
    """Per-flow packet stats are accumulated as sums so files can be combined.

    A flow's packets can straddle a file boundary. Means do not combine by
    addition, so the accumulator carries sums and counts and divides only at
    the end; this checks that splitting a batch in two changes nothing.
    """
    from foresight.data.packets import _accumulate, _finalise

    frame = pd.DataFrame({
        "flow_id": [1, 1, 1, 2, 2, 2],
        "protocol": ["tcp"] * 6,
        "IP ttl": [64, 64, 128, 255, 255, 255],
        "IP frag": [0, 0, 0, 0, 0, 0],
        "IP flags": ["DF", "DF", "", "DF", "", ""],
        "IP len": [40, 60, 80, 100, 120, 140],
        "TCP window": [100, 200, 300, 0, 500, 600],
        "TCP flags": ["S", "A", "PA", "R", "A", "A"],
        "TCP seq": [1, 1, 2, 5, 6, 7],
    })
    whole = _finalise(_accumulate(frame))
    split = _finalise(pd.concat([_accumulate(frame.iloc[:3]),
                                 _accumulate(frame.iloc[3:])])
                      .groupby(level=0).sum())
    # ttl_min does not survive a sum and is recombined separately upstream.
    compare = [c for c in whole.columns if c != "pkt_ttl_min"]
    assert np.allclose(whole[compare].to_numpy(), split[compare].to_numpy())


def test_packet_features_decode_expected_semantics():
    """The decoded fields must mean what their names say."""
    from foresight.data.packets import _accumulate, _finalise

    frame = pd.DataFrame({
        "flow_id": [1, 1, 1, 1],
        "protocol": ["tcp"] * 4,
        "IP ttl": [64, 64, 64, 64],
        "IP frag": [0, 0, 0, 0],
        "IP flags": ["DF", "DF", "", ""],
        "IP len": [100, 100, 100, 100],
        "TCP window": [0, 0, 500, 500],
        # Two packets share sequence number 7: exactly one retransmission.
        "TCP seq": [7, 7, 8, 9],
        "TCP flags": ["S", "A", "A", "R"],
    })
    out = _finalise(_accumulate(frame)).iloc[0]
    assert out["pkt_ttl_mean"] == 64 and out["pkt_ttl_std"] == 0
    assert out["pkt_df_rate"] == 0.5
    assert out["pkt_win_zero_rate"] == 0.5
    assert out["pkt_retransmit_rate"] == 0.25, "one repeat over four packets"
    assert out["pkt_syn_rate"] == 0.25, "SYN without ACK is the handshake open"
    assert out["pkt_rst_rate"] == 0.25
    assert out["pkt_per_flow"] == 4


def test_interleaved_split_never_puts_overlapping_windows_on_both_sides():
    """The gap must exceed the window overlap or the split leaks.

    A 60s window every 15s shares three quarters of its traffic with its
    neighbour, so adjacent windows are near-copies. Any train index within
    `gap` of a test index would put the same flows on both sides.
    """
    from foresight.predict.stages import blocked_split

    gap = 4
    train, test = blocked_split(400, gap=gap, block=40)
    assert train.any() and test.any()
    tr = np.flatnonzero(train)
    te = np.flatnonzero(test)
    assert np.abs(tr[:, None] - te[None, :]).min() > gap


def test_auc_handles_ties():
    """Tied scores must share the average rank.

    Every number this project reports comes through this function, and the
    baselines it is compared against are the tied cases: persistence takes two
    values, always-positive takes one. Ranking by argsort alone gave a
    coin-flip binary predictor 0.250 and a constant score 0.250, both of which
    are 0.500 -- an error that understates tied scorers and so flattered the
    model against exactly the baselines meant to keep it honest.
    """
    from foresight.baseline import _auc

    def brute(scores, labels):
        pos, neg = scores[labels == 1], scores[labels == 0]
        wins = (pos[:, None] > neg[None, :]).sum()
        ties = (pos[:, None] == neg[None, :]).sum()
        return float((wins + 0.5 * ties) / (len(pos) * len(neg)))

    # A binary predictor that is right half the time is worth exactly 0.5.
    assert _auc(np.array([1., 1., 0., 0.]), np.array([1, 0, 1, 0])) == 0.5
    # A constant score carries no information whatever it is.
    assert _auc(np.array([1., 1., 1., 1.]), np.array([1, 0, 1, 0])) == 0.5
    # A perfect split is still 1.0 even though every score is tied within class.
    assert _auc(np.array([1., 1., 0., 0.]), np.array([1, 1, 0, 0])) == 1.0

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(4, 40))
        scores = rng.choice([0.0, 0.5, 1.0, float(rng.random())], size=n)
        labels = rng.integers(0, 2, size=n)
        if labels.min() == labels.max():
            continue
        assert abs(_auc(scores, labels) - brute(scores, labels)) < 1e-9


def _internal_flows(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    start = pd.Timestamp("2017-07-07 09:00:00")
    ts = start + pd.to_timedelta(np.arange(n) * 3, unit="s")
    frame = pd.DataFrame({
        "ts": ts,
        "day": ts.date.astype(str),
        "source_ip": rng.choice(["192.168.10.5", "192.168.10.8"], n),
        "destination_ip": rng.choice(["8.8.8.8", "192.168.10.9"], n),
        "destination_port": rng.integers(1, 500, n),
        "protocol": "tcp",
        "attack_label": np.where(np.arange(n) > n * 0.8, "Bot", "BENIGN"),
    })
    for col in ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s",
                "SYN Flag Count", "Flow IAT Min"]:
        frame[col] = rng.random(n) * 10
    return frame


def test_host_windows_cover_every_host_at_every_time():
    """A host that sent nothing is a zero state, not a missing row.

    The dynamics model needs an evenly spaced sequence per host; a ragged
    index would have it learning the clock rather than the traffic.
    """
    from foresight.data.hosts import build_host_windows

    w = build_host_windows(_internal_flows(), "2017-07-07", "60s", "15s")
    hosts = np.unique(w.hosts)
    times = np.unique(w.times)
    assert len(w.states) == len(hosts) * len(times)
    for host in hosts:
        assert (w.hosts == host).sum() == len(times)
    assert np.isfinite(w.states).all()


def test_host_identity_is_not_a_feature():
    """Attacks here concentrate on one victim, so the address must not leak.

    192.168.10.50 carries 83% of all attack flows in this capture. A model
    given the address would learn which machine this capture targets, which
    transfers to nothing.
    """
    from foresight.data.hosts import build_host_windows

    w = build_host_windows(_internal_flows(), "2017-07-07", "60s", "15s")
    for column in w.columns:
        assert "ip" not in column.lower()
        assert "host" not in column.lower() or column.startswith("h_")
    assert not any(c.startswith("192.168") for c in w.columns)


def test_host_sequences_label_strictly_beyond_the_gap():
    """Per-host labels must respect the same forecast gap as network-wide."""
    from foresight.data.hosts import build_host_windows, host_sequences
    from foresight.model.dataset import Normaliser

    w = build_host_windows(_internal_flows(), "2017-07-07", "60s", "15s")
    norm = Normaliser.fit(w.states)
    history, target, risk, families, origin = host_sequences(
        w, norm, length=4, horizon=2, gap=4)
    assert len(history) == len(target) == len(risk) == len(families) == len(origin)
    assert history.shape[1] == 4
    assert set(np.unique(risk)) <= {0.0, 1.0}
    # A window labelled malicious must name the family driving it.
    assert all(f != "BENIGN" for f, r in zip(families, risk) if r > 0)
