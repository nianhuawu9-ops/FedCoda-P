# Experiment Protocol

Goal: produce a credible二区工程性 PFL paper, with FedACE-style direct
comparison and PFLlib strong baselines.

## Stage 0: Sanity

Dataset:

- MNIST or CIFAR-10, 10 clients, 20 rounds.

Methods:

- FedAvg
- FedRep
- FedALA
- FedCoda-P

Pass condition:

- FedCoda-P runs end-to-end and saves PFLlib `.h5` results.

## Stage 1: FedACE Protocol

Datasets:

- CIFAR-10
- CIFAR-100

Setting:

- clients: 20
- join ratio: 0.6
- local epochs: 5
- Dirichlet alpha: 0.1, 0.3, 0.5
- model: PFLlib CNN first; ResNet only after CNN claims are stable
- metric: average personalized client test accuracy

Primary target:

- beat or match FedACE under alpha=0.1 on CIFAR-100.

## Stage 2: Strong PFLlib Baselines

Run:

- FedAvg, FedProx, SCAFFOLD, FedDyn
- FedPer, FedRep, FedBABU, FedBN
- Ditto, pFedMe, APFL, FedALA
- FedPAC, FedCP, GPFL, FedGH, FedDBE, FedCAC, FedAS

Keep only methods that run cleanly under the selected dataset/model setting.
Document failures instead of silently dropping methods.

## Stage 3: Generality

Add one feature-shift or real-world scenario if main results are competitive:

- Digit5 or DomainNet for feature shift;
- Camelyon17 or HAR for real-world clients.

Use FedDBE/FedBN/FedAS as key baselines here.

## Metrics

Required:

- best personalized accuracy;
- final personalized accuracy;
- average accuracy curve;
- communication per round;
- total communication to target accuracy;
- last-20 accuracy std;
- best-final gap;
- spatial conflict curve;
- temporal conflict curve;
- domain conflict curves for classifier, norm, backbone.

Optional:

- client fairness/std accuracy;
- coverage ratio;
- wall-clock time.

## Figure Plan

1. Main accuracy table: CIFAR-10/100 alpha=0.1/0.3/0.5.
2. Convergence curves against FedACE-tier baselines.
3. Ablation table.
4. Conflict diagnostic plot.
5. Communication/accuracy tradeoff.

## Paper Decision Rule

Proceed to manuscript if one of these holds:

- FedCoda-P is best on CIFAR-100 alpha=0.1 and top-2 average rank overall;
- FedCoda-P is within 1 point of best accuracy but has clearly lower communication;
- FedCoda-P is not best accuracy but has the best stability under strong skew and a clear diagnostic explanation.
