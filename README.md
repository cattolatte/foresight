# Foresight

**A world model for network attack forecasting.** Learns the transition dynamics
`P(S_t+1 | S_t)` of a network from traffic telemetry, rolls the learned dynamics
forward K steps, and reports the probability that the current trajectory
converges on an infiltration — before the kill chain completes.

[![status](https://img.shields.io/badge/status-in%20development-blue)](https://github.com/cattolatte/foresight)
[![problem statement](https://img.shields.io/badge/SIH-26153-0b5394)](https://www.sih.gov.in/sih2026PS)
[![organisation](https://img.shields.io/badge/org-NTRO-1f6feb)](https://www.sih.gov.in/sih2026PS)
[![offline](https://img.shields.io/badge/inference-fully%20offline-3fb950)](#)
[![licence](https://img.shields.io/badge/licence-open%20source-8b98a5)](#)

---

## What this is, and what it deliberately is not

The problem statement makes the distinction three times, so it is the whole
design:

> "The core deliverable is a learned model of network state transition dynamics
> — **not a static classifier**."

A conventional intrusion detector maps each flow to benign or malicious in
isolation. That throws away the structure that actually identifies an
infiltration: the order in which ports are probed, SYN flags preceding ACK
floods, the inter-arrival timing of reconnaissance before lateral movement
begins. An infiltration is a process unfolding over time, not one anomalous
packet.

Foresight instead learns how network state *evolves*. Given the observed state
at time `t` — active flows, flag distributions, port activity, packet timing —
it models the distribution over the next state. That makes forward simulation
possible: roll out K steps and ask whether this trajectory is heading somewhere
bad, while there is still time to act.

The measurable consequence is that the system must be able to say something
before the attack completes. A classifier that reaches 0.99 F1 the instant
compromise happens has not solved this problem.

## Mandatory scope

| requirement | status |
|---|---|
| Flow-level features (NetFlow/IPFIX) | planned |
| Packet-level features (PCAP-derived) | planned |
| World model learning `P(S_t+1 \| S_t)` | planned |
| K-step forward simulation | planned |
| MITRE ATT&CK stage mapping | planned |
| Explainability (attention / SHAP) | planned |
| Offline demo interface | planned |
| Benchmark vs logistic regression baseline | planned |

Both feature levels are required, and the statement says why: flow-level
features capture aggregate behaviour such as a SYN flood, while packet-level
features expose the timing and sequencing of a slow reconnaissance scan built
to slip under flow-based thresholds.

Explainability is not optional either — *"black-box outputs without
interpretability are not acceptable."*

## Status

Early. Nothing is claimed as working until it is measured against the logistic
regression baseline the statement asks for, on held-out data, with the numbers
recorded here.
