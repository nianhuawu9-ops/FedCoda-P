# FedACS Absorption Plan

This project is not a continuation of the old FedACS codebase. It absorbs only
paper-useful ideas.

## Absorb

| FedACS item | Use in FedCoda-P |
|---|---|
| Spatial conflict | Estimate selected-client update incompatibility. |
| Temporal conflict | Detect update direction instability across rounds. |
| Parameter domain split | Separate classifier, normalization, and feature layers. |
| Coverage/fairness selection | Optional client selection ablation. |
| Stability metrics | last-k std, best-final gap, conflict curves. |

## Do Not Absorb

- old run-comparison framework;
- staged ablation scripts;
- NTD, vacant distillation, feature anchor calibration;
- tail guard and checkpoint restore logic;
- large config surface;
- any module not needed for the FedCoda-P claim.

## Source Pointers

Old project:

- `fl_framework/servers/fedacs_core.py`
- `fl_framework/servers/fedacs_p.py`
- `fl_framework/metrics.py`

Useful concepts:

- `_cosine_conflict`
- domain conflict diagnostics;
- coverage-aware client selection;
- `tail_std_top1`, best-final gap.

Implementation policy:

Reimplement the small math in PFLlib style. Do not import old project modules.
