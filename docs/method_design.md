# FedCoda-P Method Design

## One-Sentence Method

FedCoda-P separates PFL conflicts by parameter role, keeps decision-boundary
parameters local, and aggregates only eligible shared representations through
spatial-temporal update compatibility. FedCoda-P+ additionally uses global
class prototypes to regularize personalized heads without averaging head
weights.

## Motivation

FedACE weights sparse parameters by FIM-derived importance. This is useful but
does not answer a more basic PFL question: should this parameter group be shared
at all? In label-skew PFL, classifier heads encode client-specific decision
boundaries. They can be high-confidence, high-importance, and still bad global
knowledge.

FedCoda-P makes shareability explicit at three levels: which clients should
participate, which backbone tensors should be shared, and which semantic
features should guide local representation learning.

## Parameter Domains

| Domain | Default route | Reason |
|---|---|---|
| Classifier/head | local-only | Label skew creates incompatible decision boundaries. |
| Class prototypes | server-averaged | Share semantic class information without sharing samples or head weights. |
| BN/norm statistics | local-only | Statistics encode client/domain distribution. |
| Backbone convolution/feature layers | conflict-aware shared | Shared representation is useful but conflict-prone. |
| High-conflict backbone block | damped or frozen shared | Do not force harmful consensus. |

## Core Signals

For each eligible shared tensor:

- client shareability: selected clients mix low-conflict/high-coverage clients with random exploration;
- client update: `delta_i = w_i - w_global`;
- spatial conflict: sampled average pairwise direction disagreement among selected client updates;
- temporal conflict: disagreement between current aggregate update and EMA aggregate direction;
- conflict gate: high-conflict high-level tensors are not shared in this round;
- layer freeze: tensors with consecutive conflict can be skipped before minimum eligibility is applied;
- conflict-free projection: shared updates are projected away from directions opposite to client consensus;
- eligibility score: `E = 1 - clipped(spatial + temporal pressure)`;
- adaptive gate: the conflict threshold tightens after a personalized accuracy drop and relaxes after a gain;
- class prototype: `P_c` is the weighted mean of client feature means for class `c`;
- prototype head loss: local heads classify global prototypes without direct head averaging;
- prototype warmup: early rounds aggregate prototypes but do not train on them until `coda_proto_warmup`;
- prototype ramp: the prototype head loss is batch-count normalized and linearly increased after warmup;
- anchor alignment: backbone features classify EMA-smoothed class anchors with normalized logits;
- server update: sample-weighted update scaled by eligibility.

The current implementation uses one eligibility score per named shared-base
parameter tensor. This is not scalar element-wise gating and not semantic
module-level grouping.
ponytail: tensor-level eligibility matches PFLlib parameter names with minimal
code; upgrade to semantic blocks only if ablations show tensor granularity is
too noisy.

## Algorithm Sketch

1. Server samples clients randomly during warmup, then mixes shareability-ranked clients with random exploration.
2. Server sends shared backbone. Clients keep local head and norm state.
3. Clients train personalized model locally.
4. Clients upload shared-backbone updates only.
5. Server groups updates by parameter domain.
6. Classifier and norm updates are ignored for global aggregation.
7. Backbone updates pass through conflict gate, projection, and spatial-temporal conflict damping.
8. Server aggregates class-wise feature prototypes from participating clients.
9. Clients use EMA-smoothed global prototypes as feature anchors in the next round.
10. Server records conflict diagnostics and stability metrics.

## What Is New Versus FedACE And Recent PFL

| Existing line | FedCoda-P |
|---|---|
| FedACE weights parameters by FIM importance. | Decides aggregation eligibility by conflict type. |
| FedALA/FedAS adapt or align local/global models. | Preserves local head continuity and only damps eligible shared updates. |
| FedCAC cautiously shares sensitive parameters. | Adds temporal conflict and domain-specific eligibility to decide shareability. |
| FedLAG/FLAYER decide layer personalization. | Separates classifier, norm, and backbone conflicts, not only layers. |
| ConFREE removes conflicting update components. | Adds a lightweight projection before eligibility damping. |
| FedLAG/FLAYER adapt sharing by layer or performance. | Uses high-level tensor gates and accuracy-trend gate adaptation without client changes. |
| FedPAC/FedProto/FedGH use prototypes or classifier collaboration. | Adds prototype-guided head collaboration without directly averaging local heads. |
| FedDBE/pFedFDA focus representation distribution. | Adds lightweight class prototype calibration before considering heavier distribution matching. |

## Minimal Implementation In PFLlib

Files to add:

- `system/flcore/clients/clientcoda.py`
- `system/flcore/servers/servercoda.py`

Files to patch:

- `system/main.py`: import and branch for `FedCoda-P`.

Avoid modifying PFLlib dataset loaders or baseline implementations.

## Ablations

| Variant | Purpose |
|---|---|
| FedCoda-P full | Main method. |
| Local head only | Tests whether conflict-aware backbone matters. |
| Conflict backbone + averaged head | Tests whether head personalization matters. |
| Spatial only | Tests pairwise update conflict. |
| Temporal only | Tests historical direction conflict. |
| Projection off | Tests whether conflict-free projection helps beyond damping. |
| Adaptive gate off | Tests whether performance-trend gate control helps. |
| Prototypes off | Tests whether head collaboration helps beyond backbone aggregation. |
| Alignment on/off | Tests whether feature calibration helps after prototype head loss. |
| No eligibility damping | Tests whether eligibility is more than logging. |
| Client shareability selection on/off | Tests whether selecting lower-conflict, higher-coverage clients helps. |

## Failure Criteria

Stop this direction if:

- local-head-only matches full FedCoda-P across CIFAR-100 alpha=0.1/0.3;
- FedCP/FedDBE/FedAS dominate by more than 3 points without a cost/stability win;
- diagnostics do not correlate with performance or stability.

## Updated Position After 2023-2026 Review

FedCoda-P must be presented as **domain-aware, temporal-aware aggregation
eligibility**, not merely layer-wise personalization. This matters because
FLAYER and FedLAG already cover adaptive layer-wise decisions, and ConFREE
already covers conflict-free aggregation. The differentiated claim is the
combination:

```text
client shareability + parameter domain + spatial/temporal conflict + semantic anchors -> shareable personalized representation
```

This gives a cleaner engineering contribution than adding another optimizer or
adaptive initialization module.
