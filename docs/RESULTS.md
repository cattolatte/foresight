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
| world model | **0.557** | 0.387 | 0.992 | 0.545 | **0.783** |
| logistic regression, current window | 0.553 | 0.384 | 0.984 | 0.548 | 0.761 |
| logistic regression, full history | 0.540 | 0.412 | 0.784 | 0.388 | 0.716 |

**Read [ANALYSIS.md](ANALYSIS.md) before this table.** A single feature — flow
count — scores 0.783 AUC on the same task, and a persistence baseline using the
ground-truth label of the last observed window and no features at all scores
0.873. This benchmark does not separate a world model from a threshold on
traffic volume, and the numbers below are reported for completeness rather than
as evidence.

The world model leads on both headline measures, but the margin over the
current-window baseline is 0.004 F1 and 0.022 AUC, which is thin. Read the
operating point before the ranking: at threshold 0.5 every model here recalls
almost everything and is wrong about three times in five, against a test base
rate of 0.256. AUC is the more meaningful comparison, and 0.783 against 0.761
is a real but modest gain for roughly three hundred thousand parameters over
a linear model on the same features.

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

## Packet-level features: extracted, measured, and left out

The statement requires both feature levels and names the packet-level ones:
TTL variance, TCP window size, fragment flags, payload distribution,
retransmission counts. None survive into a flow record, so they had to come
from the packet tables.

Those tables are 272 GB across eighteen files. Downloading a prefix does not
help — the files are ordered by capture time, so the first 7.7 GB covered only
the benign Monday and every attack day sat past the end of it. Parquet is
columnar, so the fix was to stop downloading files: ten of roughly two hundred
and fifty columns carry everything named above, and a column chunk can be
fetched on its own. **23 MB per file instead of 15 GB, 0.4 GB instead of 272.**

The features decode correctly — TTL concentrates on the 64/128/255 OS defaults,
TCP flag bytes take canonical values, the DF bit is set on 72% of packets.
Coverage after extraction is 1,455,598 of 2,827,677 flows, 35–65% on every day.

They are not in the model, for a measured reason. Coverage is not random with
respect to attack family: 99.6% of PortScan flows have packet data against 0%
of the web attacks. That is a property of which files the dataset publishes,
not of the network. A logistic regression given **the coverage rate as its only
feature** reaches test AUC 0.718 — against 0.783 for the full world model.

So the apparent gain from packet features is checked against that:

| evaluation | flow features | flow + packet | delta |
|---|---|---|---|
| all windows, packet features mean-imputed | 0.778 | 0.794 | +0.016 |
| windows with coverage ≥ 0.3 | 0.638 | 0.643 | +0.005 |
| windows with coverage ≥ 0.5 | 0.676 | 0.662 | −0.014 |

Holding coverage roughly constant removes the gain and then reverses it. The
+0.016 is the release process, not the packets. The extractor ships and runs
behind `build_windows(..., packets=...)`; the headline model does not use it.

## MITRE stage mapping

The first version was a rule cascade over port entropy, fan-out and SYN counts.
Measured against the data its thresholds were fiction: median benign
`unique_dst_ports` is 19 against a branch firing above 8, `fanout` never exceeds
0.41 in any family against a branch needing 1.2, `SYN Flag Count` peaks at 0.13
against a branch needing 0.6. Two of five stages were reachable, and every
window on both test days got the same answer. It read well and measured nothing.

It is now a multinomial logistic regression over the same named features — one
weight per feature per stage, so the evidence shown is what carried the
decision. Two evaluations, because they answer different questions:

Both numbers below are computed on states the model *predicted*, not on
observed windows. That distinction cost 23 points and was nearly missed: fitted
on observed windows and applied to rolled-forward ones — which is what the
interface actually shows a stage for — accuracy fell from 0.755 to 0.543 and
every stage collapsed onto Command & Control, because the rollout regresses
toward a mean the classifier read as that one class. Naming the stage of a
future state is a harder task than labelling the current one, and the honest
number is the harder one.

| split | accuracy | chance | what it measures |
|---|---|---|---|
| day split (train 3–5 July) | 0.073 | 0.200 | a structural limit, not the method |
| interleaved blocks, gap enforced | **0.527** | 0.200 | the operating characteristic |

The day split is degenerate for this task: the training days contain only
Initial Access and denial of service, so Reconnaissance, Lateral Movement and
Command & Control exist *only* in the test set. A classifier cannot name a
class it was never shown. Stage identification is not forecasting — it
describes a state the model has already predicted — so it is fairly scored on a
split containing every stage on both sides. Each attack here runs once, for a
scheduled stretch, so blocks are interleaved rather than cut chronologically,
and a four-window gap is discarded at every boundary because a 60s window every
15s shares three quarters of its traffic with its neighbour.

| stage | precision | recall | support |
|---|---|---|---|
| Reconnaissance | 0.075 | 0.179 | 28 |
| Initial Access | 0.734 | 0.583 | 156 |
| Lateral Movement | 0.483 | 0.636 | 22 |
| Command & Control | 0.738 | 0.617 | 128 |
| Impact (DoS) | 0.316 | 0.347 | 72 |

Reconnaissance is the weak class: 0.075 precision means most windows called
reconnaissance are not, which is worth knowing before trusting that label.
Command & Control and Initial Access are the two that carry their weight.

The dataset carries no exfiltration ground truth, so that stage is never fitted
and never predicted. Denial of service maps to Impact, which is not one of the
five stages the statement names, and is carried as its own class rather than
forced into one that does not fit. Where the classifier is unsure it says so:
the interface reports low-confidence calls as weak and names the confusion the
matrix above shows rather than presenting a stage as settled.

## Counterfactual intervention

Not asked for. A classifier scores the traffic it is given; a model of
transition dynamics can be asked what happens if the state were different.

The first version reported every action as reducing peak risk by exactly 0.00.
Two causes, both silent. The playbook scaled features that no longer existed —
it was written against host-graph descriptors that later became optional and
default to off — so three scalings in five matched nothing and the panel read
as "no defensive action helps" when none had been applied. And the metric was
the reduction in *peak* risk over the horizon, which could not have been
anything but zero: at the first rollout step only one row of the history
carries the intervention and the rest is observed attack traffic.

Acting now cannot rewrite traffic that already happened. The honest measure is
how fast the constrained trajectory comes down once that history flushes:

| action | risk at horizon | contains in |
|---|---|---|
| quarantine the top talker | 0.99 → 0.01 | 75s |
| restrict east-west movement | 0.99 → 0.02 | 90s |
| block the scanned ports | 0.99 → 0.02 | 105s |
| rate-limit the source | 0.99 → 0.29 | 210s |

The ordering is what the actions mean: quarantine removes the host, throttling
only slows it. Each is a hand-specified physical approximation of a defensive
action, not a learned one, and the model has never seen a network under
intervention — the reported drift measures how far each edit moves the state
from anything observed, and all four stay inside the range where the dynamics
still have something to say.
