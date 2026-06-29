# Literature Matrix

Purpose: select a publishable PFL problem, not collect every paper.

## Target Problem

The target is personalized FL under strong label skew and representation
conflict. The paper should argue that current PFL methods often mix three
different conflicts:

- decision-boundary conflict in classifier heads;
- representation conflict in shared backbone layers;
- normalization/statistics conflict in BN or norm layers.

FedCoda-P separates these conflicts before aggregation.

## Baseline Tiers

| Tier | Methods | Why they matter |
|---|---|---|
| Required FL | FedAvg, FedProx, SCAFFOLD, FedDyn | Traditional non-IID references. |
| Required PFL | FedPer, FedRep, Ditto, pFedMe, APFL | Classic personalization families. |
| Strong PFL | FedALA, FedPAC, FedCP, GPFL, FedGH, FedDBE, FedCAC, FedAS | Recent strong PFLlib baselines. |
| Direct rival | FedACE-style method from the provided manuscript | Same publication level, same CIFAR-10/100 PFL setting. |

## Key Papers And What To Absorb

| Method | Venue/year | Public code/base | Main idea | FedCoda-P response |
|---|---:|---|---|---|
| PFLlib | JMLR 2025 | TsingZ0/PFLlib | Unified PFL benchmark with 39 algorithms, 3 scenarios, 24 datasets. | Use as base. Do not rebuild benchmark code. |
| FedRep | ICML 2021 | PFLlib | Shared representation, local head. | Keep split, replace plain backbone averaging with conflict-aware aggregation. |
| FedBN | ICLR 2021 | PFLlib | Local BN for feature shift. | Treat normalization as a separate conflict domain. |
| Ditto | ICML 2021 | PFLlib | Global and personalized models with proximal regularization. | Use as regularization baseline, avoid two-model overhead initially. |
| FedALA | AAAI 2023 | PFLlib / official code | Adaptive local aggregation before local training. | FedCoda-P avoids repeated full local-global mixing; only eligible shared parts collaborate. |
| FedPAC | ICLR 2023 | PFLlib | Feature alignment and classifier collaboration. | Compare against classifier-collaboration methods; FedCoda-P keeps head local under label skew. |
| FedCP | KDD 2023 | PFLlib / official code | Conditional policy separates feature information. | Important strong baseline for feature separation. |
| GPFL | ICCV 2023 | PFLlib / official code | Learn generic and personalized feature information. | Strong baseline for representation personalization. |
| FedDBE | NeurIPS 2023 | PFLlib | Remove domain bias in representation space. | Required if testing feature shift/domain bias. |
| FedAS | CVPR 2024 | PFLlib | Bridging inconsistency in PFL. | Recent high-tier inconsistency baseline. |
| FedACE-style | manuscript | local doc only | Adaptive initialization, contribution-aware selection, FIM element-wise aggregation. | Rival: parameter importance vs update compatibility. |

## Research Gap

FedACE and FedAS-style methods address inconsistency or parameter importance,
but they do not make aggregation eligibility a first-class decision. Under
label skew, classifier parameters can be important and still harmful to share.
FedCoda-P turns this into a method principle:

> importance is not eligibility.

## Publication Angle

Claim a practical engineering contribution:

1. conflict-type decoupling for PFL;
2. eligibility-aware shared representation aggregation;
3. lower overhead than FIM or adaptive initialization heavy methods;
4. diagnostics that explain when personalization helps.

## Must Beat Or Match

FedACE manuscript numbers to beat under its protocol:

| Dataset | alpha=0.1 | alpha=0.3 | alpha=0.5 |
|---|---:|---:|---:|
| CIFAR-10 | 91.87 | 84.28 | 79.29 |
| CIFAR-100 | 60.55 | 51.81 | 47.61 |

If absolute accuracy cannot beat all cells, acceptable fallback is:

- better average rank across strong PFL baselines;
- better strong-skew result at alpha=0.1;
- similar accuracy with lower communication or more stable tail behavior.
