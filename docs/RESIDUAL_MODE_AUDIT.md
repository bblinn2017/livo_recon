# Residual-mode audit

This audit separates the physical residual covariance model from historical
heuristics that happened to modify the same weights. The canonical coupled
estimator should have one measurement model and one joint solve, not a stack
of independently selectable approximations.

## Retained

| Mechanism | Status | Reason |
|---|---|---|
| Independent residuals | Retained default/control | Exact baseline needed for every comparison. |
| Per-`VoxelPlane` Woodbury covariance | Retained experimental mode | Direct marginalization of the plane-fit uncertainty already present in each member's variance; preserves every point-time Jacobian. |
| Gating state uncertainty | Retained, admission only | Changes correspondence admission and does not enter solve weights, avoiding double-counting prior state uncertainty. |

`lio/residual_redundancy/mode` now accepts only `off` and `woodbury`.

## Removed in R45

| Former mode | Removal basis |
|---|---|
| `plane_averaged` | Replaced a plane group with one scalar, discarded within-plane spatial/time Jacobian diversity, and had a documented catastrophic regression. It is not equivalent to correlated information aggregation. |
| `count_weighted` | Ad-hoc group-size variance scaling starved load-bearing geometric directions and catastrophically regressed prior experiments. |
| `count_weighted_renorm` | Restored only a global weight budget; it did not restore lost directional information and shared the same failure. |
| per-residual `info_gain` | Order-dependent sequential heuristic with documented divergence/regression; not a residual covariance model. |
| `woodbury_rescale` | Modified the exact covariance result by globally rescaling information to protect a legacy 6-D position eigenvalue; no canonical extended-state analogue. |
| `woodbury_directional` | Projected the covariance correction away from a selected legacy 6-D pose direction; no canonical extended-state analogue. |
| `woodbury_divpos` | Already unreachable/retired; its runtime tombstone was removed with the other obsolete variants. |
| `sigma_scale_mode=info_gain_derived` | Documented regression, one-frame-lag aggregate, and superseded by direct same-frame per-plane covariance. |
| covariance-only redundancy `info_gain` | Changed posterior covariance without the corresponding mean update, so it was not a self-consistent Kalman measurement model. |
| covariance-only redundancy `fixed` | Served only as an artificial diagnostic contraction control and shared the same mean/covariance inconsistency. |
| `density_linear`, `density_sqrt`, `density_quadratic` | Global residual-count heuristics with no current experiment configuration or permanent behavioral validation; not covariance models. |
| adaptive `chi2` sigma scale | Cross-frame feedback could hide errors in P0, Q, or Gamma_L instead of identifying them. Reduced chi-square remains a read-only diagnostic. |

The implementation files for the collapse/per-residual axis and their
always-zero frame-stat columns were removed. Old configuration keys now fail
the existing unclaimed-key validation instead of silently selecting the
independent path.

## Separate mechanisms not removed

These are not aliases for per-plane aggregation and require a separate
decision:

| Mechanism | Current scope | Audit concern |
|---|---|---|
| prior inflate/floor/fading controls | Prior covariance controls | Not residual modes. Keep outside this cleanup unless the prior-control experiment is also retired. |

## Validation after cleanup

The coding agent should build all targets and verify that configurations using
the removed keys fail clearly. Behavioral validation remains the incremental
independent-versus-per-`VoxelPlane` real-data campaign; no removed heuristic
should be reintroduced as a comparison arm.
