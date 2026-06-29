# Implementation Notes

## FedCoda-P v0

Implemented files:

- `third_party/PFLlib/system/flcore/clients/clientcoda.py`
- `third_party/PFLlib/system/flcore/servers/servercoda.py`
- `third_party/PFLlib/system/main.py`

Behavior:

- client uses FedRep-style training: local head update, then shared base update;
- server receives client models but aggregates only `model.base`;
- classifier head remains client-local;
- optional client shareability selection mixes top-score clients with random clients after warmup, using historical conflict and class coverage signals;
- aggregation uses tensor-level spatial and temporal conflict on shared-base parameters;
- spatial conflict uses sampled client pairs to avoid full `O(N^2D)` overhead by default;
- temporal conflict compares the current aggregate with an EMA history, not a single noisy previous round;
- high-conflict tensors matching `coda_gate_keywords` are skipped when spatial conflict exceeds `coda_conflict_gate`;
- optional layer freeze skips shared tensors whose spatial or temporal conflict exceeds `coda_freeze_spatial` / `coda_freeze_temporal` for `coda_freeze_patience` consecutive rounds;
- conflict-free projection removes aggregate components opposite to the client consensus direction;
- adaptive gate tightens after an accuracy drop and relaxes after an accuracy gain;
- prototype-guided head collaboration sends class-wise feature prototypes, not classifier weights;
- prototype loss is disabled for the first `coda_proto_warmup` rounds to avoid noisy early prototypes destabilizing heads;
- prototype head loss is divided by local batch count and linearly ramped by `coda_proto_ramp`, so one fixed prototype set is not over-optimized every batch;
- optional feature-anchor alignment uses normalized prototype logits with `coda_anchor_tau` during backbone training;
- global class anchors are updated by EMA with `coda_anchor_ema` to reduce prototype drift;
- eligibility damps the shared-base update;
- diagnostics include mean spatial conflict, temporal conflict, eligibility, sampled pair count, gated/frozen/projected tensor count, and current gate.

Test:

```powershell
$env:PYTHONNOUSERSITE='1'
C:\Users\user\.conda\envs\mdfl\python.exe tests\test_fedcoda_aggregator.py
```

Known environment note:

- `mdfl` has torch `2.13.0.dev20260423+cu130`.
- `h5py`, `scikit-learn`, `cvxpy`, and `calmsize` are required by the selected PFLlib baselines.
- Use `env_fedcoda_mdfl.yaml` for a full fresh environment.

Skipped:

- element-wise eligibility;
- low-rank residuals;
- FedALA-style adaptive initialization.
