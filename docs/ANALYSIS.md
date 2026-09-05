# What the benchmark was actually measuring

This is a review of the project's own results. It was prompted by a number that
should have been suspicious from the start: the world model scored 0.790 AUC
against 0.766 for a logistic regression on the same features. Three hundred
thousand parameters, an LSTM, an attention layer and a learned dynamics model,
for 0.024 AUC over a linear classifier. Either the model was doing very little,
or the task was not asking for much.

It was the task.

## The bar nobody had drawn

A score is only evidence if something simpler cannot produce it. So here is
what the forecasting benchmark can be beaten by without a model
(`eval/shortcuts.py`, test days, forecast gap of four windows):

| baseline | AUC | F1 |
|---|---|---|
| **persistence** — the ground-truth label of the last observed window, no features | **0.871** | **0.837** |
| one feature — flow count | 0.783 | 0.540 |
| one feature — fraction of flows that are TCP | 0.742 | 0.508 |
| one feature — fraction with the `-1` window sentinel | 0.701 | 0.498 |
| always predict attack | 0.500 | 0.410 |
| *world model, for comparison* | *0.785* | *0.353* |

**A single feature — flow count — matches the world model.** Persistence beats
it by nearly nine points. The headline result is not evidence that anything was
learned about attack dynamics.

## Why persistence wins: the horizon is shorter than the attacks

Attack episodes in this capture are long. Median duration is 60s on Thursday
and 240s on Friday, with the longest running seventy minutes. The forecast
horizon is 90 seconds. "Is there an attack in the next 90 seconds" is therefore
very nearly "is there an attack right now", and the benchmark is graded largely
on the autocorrelation of the label.

There was a related defect underneath it, now fixed. Windows overlap — 60s of
traffic every 15s — so the label window at `t+1` covered traffic the model had
already observed, half of it, and `t+2` a quarter. Two of six horizon steps
were detection wearing a forecast's name. `SequenceSet` now takes a `gap`,
defaulting to `window/stride`, so the first labelled window is the first one the
history has not seen. Correcting it moved AUC 0.790 → 0.785, which is itself
the finding: the leak was not what was holding the number up. Autocorrelation
was.

## Why TCP-ness wins: every attack here is TCP

`Init_Win_bytes_forward` is `-1` for 35% of flows. It is not a value; it is a
"not applicable" sentinel, set for every UDP and non-TCP flow and never for
TCP. And **44% of benign flows carry it against 0.0% of every attack family**,
because every attack in CIC-IDS2017 is TCP-based.

A model leaning on that has learned this capture's protocol composition. It
would be blind to a UDP-based attack — DNS amplification, NTP reflection — which
is exactly the generalisation the problem statement asks for. The normaliser
also mapped `-1` through `log1p(|x|)·sign(x)`, treating a missing-value flag as
a small negative quantity, so window means silently mixed "how much TCP" into
"how large the TCP window was".

## The question that is actually hard

Restrict to windows whose entire observed history is benign, and ask whether an
attack *begins*. Persistence is undefined here by construction, and the ranking
changes:

| model | onset AUC |
|---|---|
| one feature — `unique_dst_ports` (supervised) | **0.787** |
| one feature — flow count (supervised) | 0.782 |
| logistic regression, current window (supervised) | 0.770 |
| shipped world model risk head (supervised) | 0.762 |
| world model surprise (unsupervised, benign-trained) | 0.710 |

Single features still win. And the reason the world model cannot simply be
retrained for this task is a hard data limit: **the three training days contain
83 onset-positive windows — roughly fourteen attack episodes.** No model of
this size fits fourteen examples.

## What does work: surprise, and what it proves

Train the dynamics on benign traffic only — no attack label anywhere in the
objective — and score a window by how badly the model predicted it
(`eval/surprise.py`). The control that matters is naive dynamics: predict no
change, score the size of the movement.

| signal | AUC (all) | AUC (onset) |
|---|---|---|
| world model surprise, benign-trained, unsupervised | **0.752** | 0.710 |
| naive dynamics — size of the change | 0.551 | 0.577 |
| logistic regression on history, supervised | 0.726 | **0.751** |

Beating the naive control 0.752 to 0.551 is the one clean result in this
document: **the model has learned something real about how traffic evolves**, not
merely that traffic moved. And it does so without ever seeing an attack label,
which is a stronger generalisation claim than any supervised number here — it
cannot have overfitted to attack families it was never shown.

It does not beat the supervised baselines. On an earlier run it edged the
supervised regression on the onset task, 0.714 to 0.705; after the feature
correction and a retrain that reversed to 0.710 against 0.751, and a gap of
that size across a reseed was never worth a claim in the first place. The
defensible statement is the one against the naive control, where the margin is
0.20 AUC rather than 0.01.

## The mechanism behind every failure: aggregation

Detection rate per family at a matched 10% false-alarm rate, network-wide:
DDoS 0.92, Infiltration 0.62, PortScan 0.33, Bot 0.14, web brute force 0.11,
XSS 0.09. The pattern is not difficulty. It is loudness.

An attack's share of the flows, during its own episode:

| family | network-wide | on the host it touches | signal lost to aggregation |
|---|---|---|---|
| Bot | 1.17% | 100.00% | **85×** |
| Web Attack – Brute Force | 2.96% | 55.12% | 19× |
| Web Attack – XSS | 4.30% | 48.01% | 11× |
| Infiltration | 0.02% | 0.05% | 2× |
| PortScan | 57.27% | 99.50% | 1.7× |
| DDoS | 71.08% | 80.33% | 1.1× |

The families the model misses are exactly the families that averaging destroys.
Botnet traffic is one flow in eighty-five on the network and every flow on its
own host. Nothing about the architecture fixes that; the unit of modelling does.

## Modelling the host instead of the network

Same features, same window and stride, same architecture, same benign-only
training, same matched false-alarm rate — only the unit changes
(`eval/perhost.py`). Fifteen internal hosts give 128,520 training sequences
against 8,568 network-wide.

| family | network-wide | per host |
|---|---|---|
| Bot | 0.14 | **0.21** |
| Web Attack – XSS | 0.09 | **0.24** |
| Web Attack – Brute Force | 0.11 | **0.17** |
| Web Attack – Sql Injection | 0.06 | **0.14** |
| PortScan | **0.33** | 0.16 |
| Infiltration | **0.62** | 0.30 |
| DDoS | **0.92** | 0.24 |

The split is exactly along the loudness axis. Per-host modelling recovers the
families aggregation was destroying and loses the ones aggregation was helping —
a distributed flood is a network-wide phenomenon, and splitting it across
fifteen hosts is the wrong description of it.

Folding the two back into one score does not work: taking the per-window maximum
over hosts scores 0.730 AUC against 0.752 for the network view alone, because
the maximum over fifteen hosts tracks the noisiest host rather than the
compromised one. That is the correct lesson rather than a failed experiment.
**Per-host alerts should stay per-host.** "Host .17 is behaving unlike itself" is
both the finer signal and the more useful sentence than any single network
number, and collapsing it away is what discarded the advantage.

Host identity is not a feature anywhere in this. Attacks in this capture
concentrate on one victim — 192.168.10.50 carries 83% of all attack flows — so a
model given the address would learn which machine this capture targets and
nothing that transfers.

## What to do about it

**1. Change the headline metric.** Lead with onset AUC and the naive-dynamics
control, and print the shortcut table beside every result. A benchmark a single
feature can match is not a benchmark. This costs nothing and is the difference
between a number and a claim.

**2. Keep the surprise model as the primary system.** It is unsupervised, uses
no attack labels, beats its control decisively (0.752 against 0.551), and
generalises to unseen families by construction rather than by assertion. The
supervised risk head should be reported as a baseline, not as the product.

**3. Move alerting to per-host, and keep it there.** It recovers the quiet
families — botnet, web attacks — that matter operationally, and it answers
"which machine", which the network-wide model structurally cannot. Retain the
network view for volumetric attacks and run both, reported separately.

**4. The data is the ceiling, and this is the highest-leverage change.** Three
training days yield 83 onset-positive windows — about fourteen attack episodes.
Every supervised result here is fitted on fourteen events, which is why single
features win: there is nothing for a larger model to find. CSE-CIC-IDS2018 has
ten capture days, multiple victim hosts and many more distinct episodes, and is
the same feature schema. Nothing else on this list will move the numbers as
much.

**5. Fixed, and worth keeping fixed:** the `-1` sentinel in `Init_Win_bytes_*`
is now split into an explicit absence rate and a mean over the flows where the
field applies, instead of being averaged as though it were a window size.

## Honest summary

The system works, the interface works, and the engineering is sound. What the
original benchmark did not support is the claim that it works *because* it
learned network dynamics. One result does support a weaker and more defensible
version of that claim — benign-trained surprise beating naive dynamics 0.752 to
0.551 — and one measurement explains every remaining failure, which is that
network-wide averaging costs 85× of signal on the attacks that matter.

## A correction to this document's own arithmetic

The AUC these numbers are computed with was wrong for tied scores. It ranked by
`argsort` alone, which hands equal scores arbitrary distinct ranks instead of
the average rank they share, and the baselines in the table above are precisely
the tied cases: persistence takes two values, always-positive takes one. It
scored a coin-flip binary predictor at 0.250 and a constant score at 0.250, both
of which are 0.500.

The error understates tied scorers, so it was flattering the model against
exactly the baselines meant to keep it honest. Corrected, persistence moves
0.873 → 0.871 and always-positive 0.454 → 0.500; the continuous scorers are
unaffected because they have no ties. The conclusion is unchanged and slightly
firmer. `_auc` is now checked against a brute-force pair count over randomised
tie-heavy inputs in the test suite.
