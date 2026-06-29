# Recent PFL Literature Review, 2023-2026

Search date: 2026-06-27.

Scope: personalized federated learning in recent top venues and strong journals,
with emphasis on publishable gaps for FedCoda-P. Sources include OpenAlex,
Semantic Scholar-style metadata, OpenReview, arXiv, JMLR, CVF, AAAI, PMLR,
GitHub, and cloned public code under `third_party/`.

## Executive Conclusion

The last three years of PFL are moving away from simple "local head + shared
backbone" and toward five themes:

1. inconsistency between shared and personalized parameters;
2. representation bias and feature-distribution shift;
3. adaptive layer/parameter collaboration;
4. conflict-aware aggregation and negative-transfer avoidance;
5. foundation-model / prompt-based personalization.

FedCoda-P should not claim novelty as another local-head method. The defensible
research angle is:

> PFL needs conflict-aware aggregation eligibility: before weighting updates,
> decide which knowledge is shareable, local-only, or conditionally shareable.

This angle absorbs FedACS conflict diagnostics while aligning with recent
top-venue concerns in FedAS, FedCAC, ConFREE, FLAYER, pFedFDA, FedDBE, FedCP,
GPFL, and FedDecomp.

## Search Strategy

Queries:

- personalized federated learning 2023 2024 2025 2026;
- PFL label skew, feature shift, representation bias;
- FedAS, FedDBE, FedCAC, FedCP, GPFL, FedALA, FedPAC, pFedFDA, FLAYER;
- conflict-free client update aggregation PFL;
- layer-wise personalized FL conflicting gradients;
- federated prompt learning personalization foundation models.

Inclusion rule:

- Prefer 2023-2026 NeurIPS, ICML, ICLR, CVPR, ICCV, KDD, AAAI, IJCAI, ACM MM,
  JMLR, TPAMI/TNNLS/TON.
- Include older methods only as unavoidable baselines.
- Include public code when available.

Raw search artifact:

- `sources/openalex_pfl_2023_2026.json`
- `sources/search_log.md`

## Paper Map

| Paper / method | Venue | Main problem | Key idea | What FedCoda-P should absorb |
|---|---:|---|---|---|
| PFLlib | JMLR 2025 | Reproducible PFL benchmarking | 37+ FL/PFL algorithms, 3 heterogeneity scenarios, 24 datasets | Use as base and baseline source. |
| FedAS | CVPR 2024 | Intra-client and inter-client inconsistency | Parameter alignment and client synchronization | Inconsistency framing; straggler/under-trained-client robustness. |
| pFedFDA | NeurIPS 2024 | Data-scarce personalization under distribution shifts | Global feature distribution as generative classifier, adapted locally | Feature-distribution adaptation without blindly sharing classifier heads. |
| FedALA | AAAI 2023 | Local initialization under heterogeneity | Adaptive local aggregation of global and local models | Compare against; avoid copying expensive per-round local-global mixing. |
| FedPAC | ICLR 2023 | Representation/head collaboration | Feature alignment and classifier collaboration | Classifier collaboration is a direct rival to local-only head. |
| FedCP | KDD 2023 | Sample-specific global/personalized feature separation | Conditional policy with global and personalized heads | Fine-grained feature routing; strong baseline. |
| GPFL | ICCV 2023 | Simultaneous generic and personalized features | Global guidance route + personalized task route | Dual-route representation learning. |
| FedDBE | NeurIPS 2023 | Domain/representation bias | Bias eliminator in representation space | Representation bias as a first-class PFL target. |
| FedCAC | ICCV 2023 | Over-conservative personalization | Cautiously aggressive collaboration based on sensitivity and similarity | Conditional sharing instead of fixed local/shared split. |
| FedTGP | AAAI 2024 | Data and model heterogeneity with low communication | Trainable global prototypes with adaptive-margin contrastive learning | Prototype-level knowledge transfer as optional future branch. |
| FedAS/FedDBE/PFLlib family | 2023-2025 | Benchmarkable PFL systems | PFLlib-compatible code | Keep implementation PFLlib-native. |
| FedDecomp | ACM MM 2024 | General/personalized knowledge entangled inside each parameter | Additive decomposition plus low-rank personalized component | Strong theoretical alternative to hard parameter partition. |
| ConFREE | AAAI 2025 | Conflicting updates and negative transfer in PFL | Conflict-free guidance vector by projection | Directly relevant to FedACS-style conflict aggregation. |
| FedFSA | AAAI 2025 | Local generalization/sharpness ignored | Flexible sharpness-aware local optimization | Optional local optimizer add-on, not first module. |
| FLAYER | IJCAI 2025 | Layer-wise personalization | Adaptive layer-wise learning | Strongly supports layer/block-level eligibility. |
| FedLAG / adaptive layer disentanglement | 2024 preprint / 2026 TON | Gradient conflict determines layer personalization | Exclude high-conflict layers from aggregation | Very close to FedCoda-P; must differentiate by temporal/domain eligibility and diagnostics. |
| pFedGraph | ICML 2023 | Who should collaborate with whom | Inferred collaboration graph | Optional client similarity branch. |
| FOL | ICML 2025 | One-shot PFL | Model alignment, ensemble refinement, distillation | Future communication-limited scenario. |
| FedPKDA | AAAI 2026 | Privacy-preserving dynamic knowledge alignment | Privacy-aware key information alignment | Future privacy branch; not first paper. |
| FedPHA / pFedMoAP / DP-FPL | ICML/ICLR 2025 | Foundation-model prompt personalization | Global/local prompts, heterogeneous prompt lengths, MoE prompts, privacy | Future high-risk/high-reward project, not the first CIFAR paper. |

## What Problems Are Still Not Fully Solved

### 1. Shareability is still under-specified

Most PFL work chooses one of three assumptions:

- some layers are local and others are global;
- all parameters contain mixed general/personal knowledge;
- samples or clients should be routed to global/personal branches.

The unresolved issue is deciding *when a parameter group is eligible for
aggregation*. FedCAC and FLAYER move toward this, but FedCoda-P can make
eligibility explicit and measurable using domain-level and temporal conflict.

### 2. Classifier collaboration is still risky under label skew

FedPAC shows classifier collaboration can improve generalization. FedACE-like
methods use classifier/global-local mixing. However, under severe label skew,
classifier heads can encode incompatible decision boundaries. A parameter can be
important and still not shareable. This is the central opening for FedCoda-P.

### 3. Representation bias is now a primary target

FedDBE and pFedFDA show that representation space, not only the classifier,
matters. A local head alone is insufficient when backbone features degenerate or
encode client/domain bias. FedCoda-P should aggregate backbone with conflict
awareness and optionally add a lightweight feature-distribution statistic.

### 4. Layer-wise personalization is becoming crowded

FLAYER and FedLAG already target adaptive layer-wise decisions. FedCoda-P cannot
claim "layer-wise personalization" alone. It must differentiate through:

- conflict type: classifier, norm, backbone;
- temporal conflict, not only pairwise gradient conflict;
- eligibility score used for both aggregation and diagnostics;
- compatibility with PFLlib baselines and FedACE-style protocol.

### 5. Communication and one-shot settings are rising

FOL and prototype/prompt methods show pressure toward fewer communication
rounds. FedCoda-P should report communication, but one-shot PFL is a separate
paper unless early results are exceptional.

### 6. Foundation-model PFL is hot but expensive

Prompt-based PFL is active in 2025, especially for VLM/LLM settings. It is a
good second project. For the current publication target, starting with CIFAR
and PFLlib is lower risk. The FedCoda-P idea can later transfer from "layer
eligibility" to "prompt/component eligibility."

## Theory To Absorb

### Bias-variance view

pFedFDA frames classifier personalization through feature-distribution
adaptation. Local classifiers reduce bias to local distribution but raise
variance under data scarcity. Global classifiers reduce variance but can be
biased. FedCoda-P should discuss head locality as a bias-variance tradeoff, not
just a heuristic.

### Negative transfer / update conflict

ConFREE and FedLAG make conflict explicit. FedACS already has spatial/temporal
conflict. FedCoda-P should define aggregation eligibility as an anti-negative
transfer criterion:

```text
high compatibility -> shared aggregation
high persistent conflict -> local or damped aggregation
transient conflict -> temporal smoothing, not permanent personalization
```

### Knowledge decomposition

FedDecomp argues hard partitioning is limited because every parameter may carry
both general and personalized knowledge. FedCoda-P should acknowledge this and
position block-level eligibility as a lightweight engineering alternative. If
results are weak, low-rank personalized residuals are the next upgrade.

### Representation-space alignment

FedDBE, FedCP, GPFL, FedTGP, and pFedFDA all indicate that shared knowledge is
often better represented as features, prototypes, distributions, or prompts
rather than raw parameters. FedCoda-P can start parameter-based, but diagnostics
should include feature/domain conflict so the paper does not look outdated.

## Recommended Method Direction

Keep FedCoda-P, but sharpen it:

**FedCoda-P = conflict-type decoupling + aggregation eligibility + conflict-aware
shared representation aggregation.**

Implementation order:

1. PFLlib FedRep-style base: local classifier, shared backbone.
2. Domain split: head local, BN/norm local, backbone eligible.
3. Spatial conflict score per layer/block.
4. Temporal conflict score per layer/block.
5. Eligibility-damped backbone aggregation.
6. Diagnostics: domain conflict, eligibility curve, stability.
7. Optional: coverage-aware client selection.

Do not add FedALA-style adaptive initialization, FedDecomp low-rank residuals,
SAM, prototypes, or prompt learning in the first implementation.

## Strong Baseline List

Minimum:

- FedAvg, FedProx, SCAFFOLD, FedDyn;
- FedPer, FedRep, FedBABU, FedBN;
- Ditto, pFedMe, APFL;
- FedALA, FedPAC, FedCP, GPFL, FedDBE, FedCAC, FedAS;
- ConFREE and FLAYER if implementation time allows;
- FedACE-style manuscript numbers as external direct rival.

## Experiment Design Implications

Required settings:

- label skew: CIFAR-10/100 Dirichlet alpha 0.1, 0.3, 0.5;
- feature/domain shift: Digit5 or DomainNet if main result is competitive;
- real-world: HAR or Camelyon17 only if resources allow.

Required metrics:

- average personalized accuracy;
- best/final accuracy;
- last-k standard deviation;
- best-final gap;
- communication;
- per-domain conflict and eligibility curves.

Important ablations:

- local head only;
- local head + spatial conflict;
- local head + temporal conflict;
- local head + spatial-temporal conflict;
- head local vs classifier collaboration;
- BN local vs global;
- eligibility damping vs fixed backbone aggregation.

## Risk Assessment

High risk:

- FedLAG/FLAYER already cover layer-wise conflict/personalization.
- FedPAC/FedCP/GPFL may beat simple local-head designs.
- FedACE manuscript numbers are high on CIFAR-100.

Mitigation:

- Make temporal conflict and domain-specific eligibility central.
- Include feature shift, not only label skew, if results permit.
- Report stability and communication wins, not only peak accuracy.
- If accuracy lags, upgrade with low-rank personalized residuals inspired by
  FedDecomp, not with many unrelated modules.

## Source Links

- PFLlib JMLR 2025: https://www.jmlr.org/papers/v26/23-1634.html
- FedAS CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Yang_FedAS_Bridging_Inconsistency_in_Personalized_Federated_Learning_CVPR_2024_paper.html
- pFedFDA NeurIPS 2024: https://openreview.net/forum?id=Wl2optQcng
- FedALA AAAI 2023: https://ojs.aaai.org/index.php/AAAI/article/view/26330
- FedPAC ICLR 2023: https://openreview.net/forum?id=SXZr8aDKia
- FedCP KDD 2023: https://arxiv.org/abs/2307.01217
- GPFL ICCV 2023: https://arxiv.org/abs/2308.10279
- FedDBE NeurIPS 2023: https://openreview.net/forum?id=nO5i1XdUS0
- FedCAC ICCV 2023: https://arxiv.org/abs/2309.11103
- FedTGP AAAI 2024: https://arxiv.org/abs/2401.03230
- FLAYER IJCAI 2025 code: https://github.com/lancasterJie/FLAYER
- FOL ICML 2025: https://proceedings.mlr.press/v267/huang25ae.html
- pFedGraph ICML 2023: https://proceedings.mlr.press/v202/ye23b.html
- ConFREE AAAI 2025: https://ojs.aaai.org/index.php/AAAI/article/view/34449
- FedFSA AAAI 2025: https://ojs.aaai.org/index.php/AAAI/article/view/35475
- FedDecomp ACM MM 2024: https://arxiv.org/abs/2406.19931
- FedPHA ICML 2025: https://openreview.net/forum?id=y7pDvbi9xz
- Foundation-model FL list: https://github.com/lishenghui/awesome-fm-fl
