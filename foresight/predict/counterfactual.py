"""Counterfactual intervention: what happens if a defender acts.

This is the capability that separates a world model from a classifier, and
nothing in the statement asks for it. A classifier scores the traffic it is
given. A model of transition dynamics can be asked a different question: if the
state at time t were changed -- this host isolated, this port blocked, this
source rate-limited -- what does the model expect to happen next?

That question is answerable here because the model predicts states rather than
labels. Edit the current state, roll forward, and compare the resulting risk
trajectory against the untouched one. The difference is the model's estimate of
what the intervention buys.

Two honest limits, stated because a defender acting on this needs them.

The model learned dynamics from observed traffic, so it has never seen a
network under intervention. An edited state is off the training distribution by
construction, and the further the edit, the less the rollout means. Isolating
one host is a small edit; zeroing every feature is not, and the confidence
returned reflects how far the edit moved the state.

Second, this estimates what the *model* expects, not what the network will do.
It is decision support, which is what the statement asks the system to provide,
not a guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Intervention:
    name: str
    description: str
    risk_without: list[float]
    risk_with: list[float]
    terminal_reduction: float
    steps_to_contain: int | None
    seconds_to_contain: float | None
    distance: float
    confidence: str


# Each intervention is a scaling applied to the features it plausibly affects.
# Scalings rather than zeroing: blocking a port does not stop the network, it
# reduces the traffic on that port, and a state edited to all zeros is further
# from anything observed than any real defensive action would be.
PLAYBOOK: dict[str, dict] = {
    "isolate_top_talker": {
        "description": "quarantine the host originating the most flows",
        # Removing a host removes everything it was doing: its flows, its
        # packets, the destinations only it was reaching, and the handshakes
        # it was opening. Scaling flow volume alone was the original mistake.
        "scale": {"flow_count": 0.25, "unique_src_ips": 0.7,
                  "unique_dst_ips": 0.3, "unique_dst_ports": 0.4,
                  "Total Fwd Packets": 0.3, "Total Length of Fwd Packets": 0.3,
                  "Total Backward Packets": 0.35, "Flow Packets/s": 0.3,
                  "Flow Bytes/s": 0.3, "SYN Flag Count": 0.3,
                  "ACK Flag Count": 0.4, "PSH Flag Count": 0.4,
                  "fanout": 0.3},
    },
    "block_scanned_ports": {
        "description": "drop traffic to the ports being swept",
        # A sweep is many short probes. Dropping them collapses port
        # diversity and the SYN/RST pairs, and the flows that remain are
        # ordinary traffic, so the minimum inter-arrival gap grows.
        "scale": {"unique_dst_ports": 0.1, "dst_port_entropy": 0.2,
                  "SYN Flag Count": 0.15, "RST Flag Count": 0.15,
                  "flow_count": 0.5, "Flow Packets/s": 0.5,
                  "Flow IAT Min": 3.0, "Flow IAT Mean": 1.8},
    },
    "rate_limit_source": {
        "description": "throttle the highest-volume source",
        # Throttling caps rate and spreads flows out in time rather than
        # removing them, so counts fall less than rates and IAT rises.
        "scale": {"Flow Bytes/s": 0.15, "Flow Packets/s": 0.15,
                  "Total Length of Fwd Packets": 0.2,
                  "Total Backward Packets": 0.4, "flow_count": 0.4,
                  "Average Packet Size": 0.7,
                  "Flow IAT Mean": 2.5, "Flow IAT Min": 2.0},
    },
    "segment_network": {
        "description": "restrict east-west movement between subnets",
        # Segmentation blocks reach, not volume: peers and ports collapse and
        # cross-segment handshakes fail, but permitted traffic is untouched.
        "scale": {"fanout": 0.15, "unique_dst_ips": 0.2,
                  "unique_dst_ports": 0.4, "flow_count": 0.6,
                  "Total Backward Packets": 0.4, "SYN Flag Count": 0.4,
                  "Down/Up Ratio": 1.4},
    },
}


def validate_playbook(columns: list[str]) -> dict[str, list[str]]:
    """Which scalings name a feature that does not exist.

    Written because the first playbook was authored against a feature set that
    included host-graph descriptors, which later became optional and default to
    off. Three of every five scalings silently matched nothing, every
    intervention reported a peak risk reduction of exactly 0.00, and the panel
    read as "no defensive action helps" when in fact no defensive action had
    been applied. A no-op that looks like a result is worse than an error.
    """
    return {name: [k for k in spec["scale"] if k not in columns]
            for name, spec in PLAYBOOK.items()
            if any(k not in columns for k in spec["scale"])}


def apply_intervention(state: np.ndarray, columns: list[str],
                       scale: dict[str, float]) -> np.ndarray:
    """Edit a raw state according to an intervention's scalings."""
    edited = state.copy()
    for name, factor in scale.items():
        if name in columns:
            edited[columns.index(name)] *= factor
    return edited


def simulate_intervention(model, window: torch.Tensor, columns: list[str],
                          norm, name: str, steps: int = 12,
                          stride_seconds: float = 15.0) -> Intervention:
    """Roll the dynamics forward with and without a defensive action."""
    spec = PLAYBOOK[name]
    missing = [k for k in spec["scale"] if k not in columns]
    if missing:
        raise ValueError(
            f"intervention {name!r} scales features absent from this model: "
            f"{missing}. A scaling that names nothing is a silent no-op.")

    baseline = model.rollout(window, steps=steps)
    risk_without = [float(r) for r in baseline.risk]

    # The intervention persists. Editing only the last observed window changes
    # one row of twelve and the risk barely moves -- the first version of this
    # reported every action as reducing peak risk by 0.00, which is what a
    # defender would have read as "nothing helps".
    #
    # A block stays in force, so the scaling is applied to every state the
    # model predicts before it is fed back, and the trajectory is simulated
    # under the constraint rather than merely started from an edited point.
    history = window.clone()
    raw_last = norm.invert(window[0, -1].cpu().numpy())
    edited_raw = apply_intervention(raw_last, columns, spec["scale"])
    edited = torch.from_numpy(norm(edited_raw[None, :])[0]).to(window.dtype)
    history[0, -1] = edited

    risk_with, drift = [], []
    with torch.no_grad():
        for _ in range(steps):
            nxt, logit, _ = model(history)
            raw = norm.invert(nxt[0].cpu().numpy())
            constrained_raw = apply_intervention(raw, columns, spec["scale"])
            constrained = torch.from_numpy(
                norm(constrained_raw[None, :])[0]).to(window.dtype)
            drift.append(float(np.linalg.norm(
                (constrained - nxt[0]).cpu().numpy()) / np.sqrt(len(columns))))
            risk_with.append(float(torch.sigmoid(logit)[0]))
            history = torch.cat(
                [history[:, 1:], constrained.unsqueeze(0).unsqueeze(0)], dim=1)

    # How far the constraint holds the trajectory from where it would have
    # gone, averaged over the horizon. The further, the less the model's
    # dynamics have anything to say about it.
    distance = float(np.mean(drift)) if drift else 0.0

    # Containment is measured at the end of the horizon, not across it.
    #
    # The first version scored every action by the reduction in *peak* risk and
    # reported 0.00 for all four. That number could not have been anything
    # else: at the first rollout step only one of the history rows carries the
    # intervention and the rest are observed attack traffic, so early risk is
    # un-intervened by construction and the maximum is the untouched peak.
    #
    # It is also the wrong question. Acting now cannot rewrite the traffic that
    # already happened -- risk stays high while that history is still in the
    # buffer, and the honest measure is how quickly the constrained trajectory
    # comes down once it flushes, which is what a defender is choosing between.
    contained = next((i for i, r in enumerate(risk_with) if r < 0.5), None)
    confidence = ("high" if distance < 0.5 else
                  "moderate" if distance < 1.5 else
                  "low — the edited state is far from anything observed")

    return Intervention(
        name=name, description=spec["description"],
        risk_without=risk_without, risk_with=risk_with,
        terminal_reduction=risk_without[-1] - risk_with[-1],
        steps_to_contain=contained,
        seconds_to_contain=None if contained is None
        else contained * stride_seconds,
        distance=distance, confidence=confidence)


def rank_interventions(model, window: torch.Tensor, columns: list[str],
                       norm, steps: int = 12, stride_seconds: float = 15.0
                       ) -> list[Intervention]:
    """Every playbook action, ordered by how fast it contains the incident.

    Ordered by effect rather than by cost, because the model has no view of
    operational cost -- isolating a database server and blocking a scanned port
    are not equally cheap, and only the defender knows which.
    """
    results = [simulate_intervention(model, window, columns, norm, name, steps,
                                     stride_seconds)
               for name in PLAYBOOK]
    # Unbounded sorts first: an action that never contains the incident ranks
    # below every action that does, however small its terminal reduction.
    return sorted(results, key=lambda i: (i.steps_to_contain is None,
                                          i.steps_to_contain or 0,
                                          -i.terminal_reduction))
