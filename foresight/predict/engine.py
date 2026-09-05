"""Forward simulation, attack-stage mapping and explanation.

Three outputs are required of every prediction:

    "infiltration probability score, predicted MITRE ATT&CK stage, and top
     contributing traffic features"

and one prohibition: "black-box outputs without interpretability are not
acceptable."

Stage mapping is rules over the *predicted* state rather than a learned head,
and that is a deliberate choice. The training days contain only brute force and
denial of service, so a learned stage classifier could not name reconnaissance,
command and control, or exfiltration at all -- it would have no examples of
them. Rules grounded in the features the statement itself names generalise to
stages never seen in training, and they can be read by a defender, which a
softmax cannot.

Attribution combines two sources. Attention gives *when* -- which of the
observed windows the model leaned on. Gradients of the risk score with respect
to the input give *what* -- which flags, ports or timing statistics moved it.
Both are needed: knowing the spike mattered is useless without knowing it was
port entropy rather than volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# The five phases the statement names.
STAGES = ["Reconnaissance", "Initial Access", "Lateral Movement",
          "Command & Control", "Exfiltration"]


@dataclass
class Prediction:
    infiltration_probability: float
    horizon_curve: list[float]
    stage: str
    stage_evidence: str
    top_features: list[tuple[str, float]]
    attention: list[float]
    lead_windows: int | None = None
    notes: list[str] = field(default_factory=list)


def _z(state: dict[str, float], name: str) -> float:
    return float(state.get(name, 0.0))


def infer_stage(state: dict[str, float]) -> tuple[str, str]:
    """Name the attack phase a predicted state most resembles, with its reason.

    Ordered by specificity, not likelihood: a port sweep and a SYN flood both
    raise flow counts, and only the port spread separates them. Each branch
    returns the evidence that fired so the answer is auditable.
    """
    entropy = _z(state, "dst_port_entropy")
    ports = _z(state, "unique_dst_ports")
    fanout = _z(state, "fanout")
    syn, ack = _z(state, "SYN Flag Count"), _z(state, "ACK Flag Count")
    flows = _z(state, "flow_count")
    fwd, bwd = _z(state, "Total Length of Fwd Packets"), _z(state, "Total Length of Bwd Packets")
    idle = _z(state, "Idle Mean")

    if entropy > 1.5 and ports > 8 and fanout > 1.2:
        return STAGES[0], (f"port spread across {ports:.0f} destinations "
                           f"(entropy {entropy:.2f}) with fan-out {fanout:.1f} — "
                           "a sweep, not a service")
    if syn > 0.6 and ack < 0.4 and flows > 5:
        return STAGES[0], (f"half-open connections: SYN {syn:.2f} against "
                           f"ACK {ack:.2f} over {flows:.0f} flows")
    if flows > 5 and ports <= 3 and entropy < 1.0:
        return STAGES[1], (f"repeated attempts against {ports:.0f} service(s) — "
                           "concentrated retry, characteristic of credential guessing")
    if fanout > 2.0 and entropy < 1.5:
        return STAGES[2], (f"one source reaching {fanout:.1f}x more hosts than "
                           "it receives from, on few ports — east-west spread")
    if idle > 0.5 and flows > 0 and abs(fwd - bwd) < max(fwd, bwd, 1.0) * 0.3:
        return STAGES[3], (f"periodic low-volume exchange (idle {idle:.2f}) with "
                           "balanced directions — beaconing")
    if bwd > fwd * 2.0 and bwd > 0:
        return STAGES[4], (f"outbound volume {bwd:.0f} against inbound {fwd:.0f} — "
                           "asymmetric egress")
    return STAGES[1], "elevated activity without a distinguishing signature"


def explain(model, window: torch.Tensor, columns: list[str],
            top_k: int = 6) -> tuple[list[tuple[str, float]], list[float]]:
    """Which features drove the risk, and which windows the model attended to.

    Gradient times input, summed over the history: the gradient alone says how
    sensitive the score is, which is not the same as how much a feature
    actually contributed here.
    """
    model.eval()
    probe = window.clone().requires_grad_(True)
    _, logit, attention = model(probe)
    logit.sum().backward()
    contribution = (probe.grad * probe).detach()[0].sum(dim=0).cpu().numpy()

    order = np.argsort(-np.abs(contribution))[:top_k]
    top = [(columns[i], float(contribution[i])) for i in order]
    return top, attention[0].detach().cpu().tolist()


def predict(model, window: torch.Tensor, columns: list[str],
            steps: int = 6, threshold: float = 0.5, norm=None) -> Prediction:
    """Roll the dynamics forward and report what the trajectory implies.

    `norm` is required for the stage rules to mean anything: they are written
    in raw units and the model works in standardised ones.
    """
    roll = model.rollout(window, steps=steps)
    curve = [float(p) for p in roll.risk]
    peak = float(max(curve))

    # The stage is read from the first simulated state that crosses threshold;
    # if none does, from the end of the horizon, which is the most developed
    # trajectory the model expects.
    crossing = next((i for i, p in enumerate(curve) if p >= threshold), None)
    index = crossing if crossing is not None else len(curve) - 1
    raw = roll.states[index].cpu().numpy()
    if norm is not None:
        raw = norm.invert(raw)
    predicted_state = dict(zip(columns, raw))
    stage, evidence = infer_stage(predicted_state)

    top, attention = explain(model, window, columns)
    notes = []
    if crossing is None:
        notes.append("no simulated window crosses the threshold; stage is "
                     "read from the end of the horizon and is indicative only")
    if peak < threshold:
        notes.append(f"peak risk {peak:.2f} below threshold {threshold:.2f}")

    return Prediction(infiltration_probability=peak, horizon_curve=curve,
                      stage=stage, stage_evidence=evidence, top_features=top,
                      attention=attention,
                      lead_windows=crossing, notes=notes)
