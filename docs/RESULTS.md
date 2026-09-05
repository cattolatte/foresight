# Results

All numbers are on capture days the model never trained on, carrying attack
families it never saw. Selection was on a chronological split of the training
days; the test days were touched once, to produce this table.

## The benchmark the statement requires

> "Benchmark results comparing model performance (F1 score, precision, recall,
> false positive rate) against a logistic regression baseline trained on the
> same features."

Train: 3–5 July (benign, FTP/SSH brute force, DoS).
Test: 6–7 July (web attacks, infiltration, port scan, botnet, DDoS).

| model | F1 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| world model | 0.538 | 0.426 | 0.728 | **0.340** | 0.761 |
| logistic regression, current window | **0.554** | 0.385 | 0.986 | 0.547 | **0.766** |
| logistic regression, full history | 0.540 | 0.418 | 0.762 | 0.368 | 0.721 |

**The world model does not beat the baseline on F1 or AUC.** It is level with
it, and the statement asks for a demonstrated improvement, so this is reported
as a failure to demonstrate rather than dressed up.

Where it does differ is the false positive rate: 0.340 against 0.547 at
comparable AUC, meaning roughly one third fewer benign windows raised as alarms
for the same discrimination. In an operational setting that is the difference
between a queue an analyst can work and one they learn to ignore, but it is a
narrower claim than the one asked for.

## Lead time, and why the obvious version of it is worthless

The statement's real claim is prediction "before compromise is completed", and
F1 does not measure that. A detector that fires in the same window the attack
lands scores perfectly and warns nobody.

Lead time asks the question directly: at the moment a model first crosses its
threshold, how long remains before the attack actually begins? The first
version of this measurement gave every model its F1-optimal threshold, and duly
reported that a baseline with a false positive rate of 0.99 had lead time
identical to the world model. Of course it did — it was alarming almost
constantly.

Measured properly, at a common 5% false-alarm budget:

| model | warn rate | chance | median lead |
|---|---|---|---|
| world model | 47% | 40% | 150s |
| logistic regression, current window | 44% | 41% | 150s |
| logistic regression, full history | 42% | 40% | 150s |

The chance column matters. Searching ten windows back for a crossing gives any
model ten independent opportunities, so at a 5% false-alarm rate roughly 40% of
episodes are "warned" by arithmetic alone. Every model here sits within a few
points of that floor.

**Early warning is not yet demonstrated.** The world model is marginally above
chance and marginally above the baselines, and neither margin is large enough
to claim.

## What was learned about the data, which cost the most time

**The timestamps are two different formats in one column.** Benign rows read
`03/07/2017 08:55:58`, attack rows `4/7/2017 10:30` — day-first, no seconds.
Parsed with pandas defaults, four rows in five become `NaT`, and every attack
row is among them: the dataset appears to span one day, and every hour appears
0% malicious. With `dayfirst=True, format="mixed"` it parses completely and the
canonical five-day structure appears.

**The window length is set by the clock, not by tuning.** Because attack rows
carry minute-resolution timestamps, every attack flow lands exactly on a minute
boundary. At 30-second windows, two thirds are empty by construction, and the
learned dynamics faithfully reproduce an alternation that is an artefact of the
timestamp format. That model scored AUC 0.811 — higher than the honest 60s
result — which is worth stating plainly: the better-looking number came from
modelling the clock.

**Overlapping windows are how the sequence count is recovered.** Three capture
days at one window a minute is about two thousand sequences for a model with
three hundred thousand parameters, and the risk loss collapsed to 0.0007.
A 60s window stepped every 15s gives four times the sequences over the same
traffic. The split stays chronological, because overlapping windows either side
of a random split would leak.

## Rollout stability

Trained one step ahead and rolled out K, the model oscillated: alternating
between a near-empty state and a scan signature on successive steps. That
turned out to be faithful — the data alternates, for the clock reason above —
but it is not a trajectory a defender could act on. The dynamics are now
supervised over a four-step rollout as well as one step, so the model is
trained on the distribution it is asked to produce rather than only on the one
it sees.

## Open

- **Packet-level features are not implemented.** The statement requires both
  levels and names TTL variance, TCP window size, fragment flags and
  retransmission counts. Only flow-level features are built. The dataset ships
  a packet-level table; it is 4.3 GB and has not been ingested.
- **The improvement over logistic regression is not demonstrated.** That is the
  central claim of the statement and the honest status is: not yet.
- The infiltration class specifically has 36 flows in the entire corpus, which
  is too few to evaluate as its own family.
