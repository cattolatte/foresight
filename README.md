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

| requirement | status | measured |
|---|---|---|
| Flow-level features (NetFlow/IPFIX) | done | 33 features, 60s windows at a 15s stride |
| Packet-level features (PCAP-derived) | extracted, not used | TTL, TCP window, fragment flags, retransmissions — [left out for a measured reason](docs/RESULTS.md#packet-level-features-extracted-measured-and-left-out) |
| World model learning `P(S_t+1 \| S_t)` | done | LSTM + attention dynamics, 0.790 AUC |
| K-step forward simulation | done | 6-step rollout, supervised on its own trajectory |
| MITRE ATT&CK stage mapping | done | 0.493 over five stages, 0.200 chance |
| Explainability (attention / SHAP) | done | gradient×input + attention, per-stage weights |
| Offline demo interface | done | CLI and a local web UI, no network calls |
| Benchmark vs logistic regression baseline | done | beats both baselines on F1 and AUC |

Both feature levels are required, and the statement says why: flow-level
features capture aggregate behaviour such as a SYN flood, while packet-level
features expose the timing and sequencing of a slow reconnaissance scan built
to slip under flow-based thresholds.

Explainability is not optional either — *"black-box outputs without
interpretability are not acceptable."*

## Results

Held-out days, attack families never seen in training. Full write-up with the
negative results in [docs/RESULTS.md](docs/RESULTS.md).

| model | F1 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| **world model** | **0.556** | 0.386 | 0.990 | 0.546 | **0.790** |
| logistic regression, current window | 0.554 | 0.385 | 0.986 | 0.547 | 0.766 |
| logistic regression, full history | 0.540 | 0.418 | 0.762 | 0.368 | 0.721 |

Beyond the statement, the learned dynamics are used to answer a question a
classifier cannot: **what happens if a defender acts.** Editing the state and
rolling forward under the constraint estimates how fast each response contains
the incident — quarantining the top talker in 75s, rate-limiting the source in
210s — with the caveat that the model has never observed a network under
intervention.

Three things measured and reported as negative:

- **Packet features do not help.** They appear to add 0.016 AUC, but coverage
  in the published dataset correlates with attack family, and coverage alone
  scores 0.718. Held at constant coverage the gain vanishes.
- **Host-graph features hurt.** Test AUC 0.761 without, 0.606 with; they let
  the model fit the topology of the days it trained on. Behind a flag, off.
- **Lead time is at chance.** The warn rate at a 5% false-alarm budget is 42%
  against a 40% floor from the lookback window alone.

## Running it

```bash
python -m uvicorn foresight.server:app --port 8090   # web interface
python foresight/cli.py capture.parquet --top 8      # terminal
python eval/benchmark.py                             # the table above
python -m pytest tests/                              # 12 tests
```
