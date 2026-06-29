# Research Memory

This file is the persistent project memory for FedCoda-P. Update it only when a
paper or experiment changes the research decision.

## Current Thesis

FedCoda-P should be positioned as:

> conflict-decoupled PFL with aggregation eligibility.

The core claim is not "personalized head works." The claim is:

> under non-IID data, parameter importance, parameter sensitivity, and parameter
> shareability are different. FedCoda-P estimates shareability through
> domain-specific spatial-temporal conflict and only aggregates eligible shared
> representations.

## Must Remember

1. FedAS already owns "inconsistency" framing at CVPR 2024.
2. FedCAC already argues sensitive parameters may still benefit from cautious
   collaboration.
3. FLAYER and FedLAG already move toward layer-wise decisions.
4. ConFREE already uses conflict-free update aggregation.
5. FedDBE and pFedFDA show representation distribution matters.
6. FedCP/GPFL are strong feature-separation baselines.
7. FedACE-style manuscript is a direct二区 rival with high CIFAR numbers.

Therefore:

- Do not sell FedCoda-P as layer-wise personalization alone.
- Do not sell it as local-head FedRep.
- Do not sell it as generic conflict-free aggregation alone.
- Sell it as domain-aware, temporal-aware aggregation eligibility.

## Best First Paper Path

Title candidate:

**FedCoda-P: Conflict-Decoupled Personalized Federated Learning via
Eligibility-Aware Representation Aggregation**

Problem statement:

Existing PFL methods either partition parameters by architecture, adaptively mix
models, or weight updates by importance. These strategies can confuse important
but unshareable client-specific knowledge with genuinely collaborative
knowledge. This is especially harmful under label skew and feature shift.

Method:

- local classifier head;
- local normalization/statistics;
- conflict-aware backbone aggregation;
- spatial conflict for selected-client disagreement;
- temporal conflict for round-to-round instability;
- eligibility score for aggregation damping;
- diagnostics for domain conflict.

First implementation:

- build on PFLlib FedRep style;
- add `clientcoda.py`, `servercoda.py`;
- register in `main.py`;
- keep block-level eligibility.

Do not add:

- FedALA local-global mixing;
- FedDecomp low-rank residual;
- SAM;
- prototypes;
- foundation-model prompts.

Upgrade path if results are weak:

1. add low-rank personalized residual on high-conflict backbone blocks;
2. add feature-distribution alignment inspired by pFedFDA;
3. add cautious collaboration for classifier heads inspired by FedCAC;
4. add client graph only if similarity effects are obvious.

## Public Code Cloned

Stored under `third_party/`:

- PFLlib: benchmark base.
- FedALA: adaptive local aggregation.
- FedPAC: feature alignment and classifier collaboration.
- FedCP: conditional feature separation.
- GPFL: generic/personalized feature routes.
- DBE: domain bias eliminator.
- FedAS: parameter alignment and synchronization.
- FedTGP: prototype transfer.
- pFedFDA: feature distribution adaptation.
- FLAYER: adaptive layer-wise learning.

These are references, not dependencies.

## Publication Go / No-Go

Go if:

- FedCoda-P beats or matches strong baselines on CIFAR-100 alpha=0.1;
- or top-2 average rank with lower communication/stability advantage;
- or clear feature-shift gain with conflict diagnostics.

No-go if:

- local-head-only matches FedCoda-P;
- FedCP/GPFL/FedDBE/FedAS dominate by more than 3 points everywhere;
- conflict diagnostics do not explain performance.
