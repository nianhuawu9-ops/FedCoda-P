# FedCoda-P

Publication-first project for personalized federated learning.

Working title:

**FedCoda-P: Conflict-Decoupled Personalized Federated Learning with Eligibility-Aware Shared Representation Aggregation**

Core claim:

FedCoda-P does not ask only how to weight client parameters. It first decides
which parameter groups are eligible for collaboration under non-IID conflicts:
classifier heads stay local, normalization statistics stay local by default,
and shared representations are aggregated through spatial-temporal conflict
signals absorbed from the FedACS line.

## Layout

```text
docs/
  literature_matrix.md      # publication-oriented literature and baseline map
  method_design.md          # FedCoda-P method definition
  experiment_protocol.md    # experiments needed to beat FedACE-style papers
  fedacs_absorption.md      # what is absorbed from the old FedACS project
third_party/
  PFLlib/                   # benchmark base, cloned from TsingZ0/PFLlib
fedcoda/
  patches/                  # future minimal PFLlib patches
  notes/                    # implementation notes and run logs
scripts/
  setup_pfllib.ps1
  run_baselines.ps1
  run_fedcoda.ps1
```

## Baseline Policy

Use PFLlib as the benchmark base. Do not rebuild common FL/PFL plumbing unless
PFLlib blocks a required experiment.

Primary baselines:

- FedAvg, FedProx, SCAFFOLD, FedDyn
- FedPer, FedRep, FedBABU, FedBN
- Ditto, pFedMe, APFL, FedALA
- FedPAC, FedCP, GPFL, FedGH, FedDBE, FedCAC, FedAS
- FedACE-style protocol from the provided manuscript

## Next Step

Implement FedCoda-P as the smallest PFLlib extension:

- `servercoda.py`
- `clientcoda.py`
- one branch in `system/main.py`

Skipped: standalone framework rewrite. Add it only if PFLlib cannot express the
publication protocol.

## Environment

Preferred reproducible environment follows the user's MDFL torch stack:

```powershell
conda env create -f env_fedcoda_mdfl.yaml
conda activate fedcoda-mdfl
```

For quick local checks, `D:\anaconda3\envs\mytorch\python.exe` is known to
import torch 2.11 on this machine.
