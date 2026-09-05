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


def infer_stage(state: dict[str, float], stage_model=None
                ) -> tuple[str, str, float]:
    """Name the attack phase a predicted state most resembles, with its reason.

    This was a rule cascade -- thresholds on port entropy, fan-out and SYN
    counts, each branch returning the evidence that fired. Measured against the
    data the thresholds were fiction: median benign `unique_dst_ports` is 19
    against a branch that fired above 8, `fanout` never exceeds 0.41 in any
    family against a branch needing 1.2, and `SYN Flag Count` peaks at 0.13
    against a branch needing 0.6. Two outcomes were reachable out of five and
    every window on both test days got the same answer.

    The fitted classifier in `foresight.predict.stages` replaces it and is
    scored in `eval/stages.py`: 0.527 accuracy over five stages against a 0.200
    chance floor, measured on rolled-forward states because that is what this
    function is given, per-stage precision and recall reported there. It stays
    interpretable -- one weight per named feature per stage -- so the evidence
    returned is still the features that carried the decision, but now they are
    the features the fit actually used.
    """
    if stage_model is None:
        raise ValueError(
            "a fitted stage model is required; run eval/stages.py to produce "
            "checkpoints/world/stages.pkl. The hand-written cascade it "
            "replaced could only return two of five stages.")
    vector = np.array([[state.get(c, 0.0) for c in stage_model.columns]])
    return stage_model.predict(vector)


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
            steps: int = 6, threshold: float = 0.5, norm=None,
            stage_model=None) -> Prediction:
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
    stage, evidence, stage_confidence = infer_stage(predicted_state, stage_model)

    top, attention = explain(model, window, columns)
    notes = []
    if crossing is None:
        notes.append("no simulated window crosses the threshold; stage is "
                     "read from the end of the horizon and is indicative only")
    if peak < threshold:
        notes.append(f"peak risk {peak:.2f} below threshold {threshold:.2f}")
    reliability = (stage_model.precision or {}).get(stage)
    if reliability is not None and reliability < 0.5:
        notes.append(f"stage \"{stage}\" is right {reliability:.0%} of the time "
                     "on held-out data; treat it as a hint, not a finding")
    if stage_confidence < 0.5:
        notes.append(f"stage is a weak call ({stage_confidence:.2f} confidence); "
                     "the classifier separates five predicted stages at 0.53 "
                     "accuracy against a 0.20 chance floor, and reconnaissance "
                     "is its weakest class at 0.08 precision")

    return Prediction(infiltration_probability=peak, horizon_curve=curve,
                      stage=stage, stage_evidence=evidence, top_features=top,
                      attention=attention,
                      lead_windows=crossing, notes=notes)
