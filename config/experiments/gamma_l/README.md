# Phase-3 joint-knot LiDAR information controls

These overlays compare the unchanged independent-residual solve with the
existing matched-plane covariance model ported to the full joint-knot
Jacobian.  `rho` is deliberately swept rather than selected here.  The
0.9 information-discount cap is a stability guard and must be reported when
it activates; it is not part of the unbounded textbook covariance model.

The current grouping corrects only uncertainty shared by returns matched to
the same fitted map plane.  It does not yet model temporal, ray, voxel-neighbor,
or scan-registration correlations.  Do not call this a complete LiDAR
correlation model.
