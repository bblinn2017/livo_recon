#!/usr/bin/env python3
"""CQ-32 item 3: the design-integrity preflight assert_responses.py's own
comment names as the precedent ("a preflight that prints a per-item
pass/fail table and exits nonzero, which the generator calls and refuses
to emit jobs on").

WHAT THIS CATCHES.  A dead-scope key is FORCED to its default regardless of
what a cell's label says (config_resolve.h's nested()/nestedMode() -- see
CQ-32 item 1). If a design crosses a factor that collapses under another
factor's setting, two cells with DIFFERENT declared labels can resolve to
the IDENTICAL effective configuration -- the exact TQ-20/use_bins defect:
voxel_map/plane/use_bins is refused (and therefore omitted, forced false)
whenever plane_fit_mode=debiased, so every debiased cell ran with
use_bins=false regardless of its own +1/-1 label, and a seven-factor design
quietly became six-factor on half its blocks with nothing refusing it.

This script re-derives the EFFECTIVE config for each cell in a design from
its declared overrides, using the SAME dead-scope table
lio_processing.cpp/voxelmap.cpp actually apply (NESTING_TABLE below,
extracted by reading every cfg.nested<T>()/nestedMode() call site directly
-- grep the two files for "cfg.nested" to re-verify this table against the
live source before trusting it on a new design; a call site added there
without a matching row here is exactly the gap this script exists to
close). It then digests each cell's effective config and refuses if two
DISTINCT labels share a digest.

Usage:
    assert_design.py --cells cells.json [--base base_overrides.json]
    assert_design.py --config-dir DIR   (one *.yaml per cell, filename = label)

cells.json: {"label1": {"key/path": value, ...}, "label2": {...}, ...}
Keys use the same slash-path convention as the C++ side
(voxel_map/plane/use_bins, not nested YAML) for a flat, order-independent
representation -- see `flatten_yaml()` if starting from real config.yaml
files (also what --config-dir uses internally).
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# STRUCT DEFAULTS -- every key ANY nesting rule below can force, or that a
# design might declare, needs a default here so a cell that never sets it
# still resolves to something (matching the C++ member-default / nested()'s
# own `def` argument, whichever this project's constructor uses. Re-verify
# against the .h/.cpp sources if either drifts).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "spline/mode": "spline",
    "spline/control_points/hz": 100.0,
    "spline/refine/iters": 1,
    "spline/log_en": False,
    "spline/trajectory_log/mode": "off",
    "spline/trajectory_log/hz": 200.0,
    "adaptive_q/enable": False,
    "adaptive_q/beta_acc": 0.3,
    "adaptive_q/beta_gyr": 0.3,
    "adaptive_q/z_rate_limit": 0.02,
    "adaptive_q/acf1_max": 1.00,  # CQ-35
    "adaptive_q/bounds/max_ratio": 100.0,
    "adaptive_q/bounds/min_ratio": 0.01,
    "adaptive_q/warmup_frames": 20,
    "adaptive_q/ema": 0.9,
    "adaptive_q/noise_floor/mode": "allan",
    "adaptive_q/noise_floor/scale": 1.0,
    "adaptive_q/log_en": False,
    "lio/residual_redundancy/mode": "off",
    "lio/residual_redundancy/rho": 1.0,
    "lio/residual_redundancy/max_discount": 0.9,
    "imu/ds/mode": "first",
    "imu/ds/ds_leaf_size": 0.15,
    "lio/ekf/density_sigma_mode": "off",
    "voxel_map/plane/weight_floor/mode": "sensor_range",
    "voxel_map/plane/weight_floor/constant": 1e-3,
    "voxel_map/plane/weight_floor/incidence_k": 1.0,
    "voxel_map/plane/plane_fit_mode": "pca",
    "voxel_map/plane/plane_var_mode": "eigengap",
    "voxel_map/plane/plane_var_denom_floor_en": False,
    "voxel_map/plane/use_bins": False,
    "voxel_map/plane/bin_size_fraction": 0.2,
    "voxel_map/plane/bin_weight_mode_fit": "count",
    "voxel_map/plane/bin_weight_mode_var": "count",
    "voxel_map/plane/log_consistency_mode": "off",
    "voxel_map/plane/log_consistency_corr_stride": 1,
}

# ---------------------------------------------------------------------------
# NESTING TABLE -- (key, scope_description, live_predicate, dead_default),
# IN THE SAME ORDER cfg.nested()/nestedMode() calls them in the C++, since a
# later predicate can read an earlier key's OWN resolved (possibly-forced)
# value. `live_predicate` takes the cell's resolved-so-far dict.
#
# Source: src/processing/lio_processing.cpp (spline/adaptive_q/redundancy/ds
# block, ~line 142-218) and src/map/voxelmap.cpp (weight_floor/plane_var/
# use_bins block, ~line 190-322). Re-grep both files for "cfg.nested" before
# trusting this on a design that crosses a key not listed here.
# ---------------------------------------------------------------------------
NESTING_TABLE = [
    ("spline/control_points/hz", "spline/mode",
     lambda r: r["spline/mode"] != "raw_imu", 100.0),
    ("spline/refine/iters", "spline/mode=spline+refine",
     lambda r: r["spline/mode"] == "spline+refine", 1),
    ("spline/log_en", "spline/mode",
     lambda r: r["spline/mode"] != "raw_imu", False),
    ("spline/trajectory_log/mode", "spline/mode",
     lambda r: r["spline/mode"] != "raw_imu", "off"),
    ("spline/trajectory_log/hz", "spline/trajectory_log/mode=dense",
     lambda r: r["spline/mode"] != "raw_imu" and r["spline/trajectory_log/mode"] == "dense", 200.0),
    ("adaptive_q/enable", "spline/mode",
     lambda r: r["spline/mode"] != "raw_imu", False),
    ("adaptive_q/beta_acc", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 0.3),
    ("adaptive_q/beta_gyr", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 0.3),
    ("adaptive_q/z_rate_limit", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 0.02),
    ("adaptive_q/acf1_max", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 1.00),
    ("adaptive_q/bounds/max_ratio", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 100.0),
    ("adaptive_q/bounds/min_ratio", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 0.01),
    ("adaptive_q/warmup_frames", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 20),
    ("adaptive_q/ema", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], 0.9),
    ("adaptive_q/noise_floor/mode", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], "allan"),
    ("adaptive_q/noise_floor/scale", "adaptive_q/noise_floor/mode=allan",
     lambda r: (r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"]
                and r["adaptive_q/noise_floor/mode"] == "allan"), 1.0),
    ("adaptive_q/log_en", "adaptive_q/enable",
     lambda r: r["spline/mode"] != "raw_imu" and r["adaptive_q/enable"], False),
    ("lio/residual_redundancy/rho", "lio/residual_redundancy/mode!=off",
     lambda r: r["lio/residual_redundancy/mode"] != "off", 1.0),
    ("lio/residual_redundancy/max_discount", "lio/residual_redundancy/mode!=off",
     lambda r: r["lio/residual_redundancy/mode"] != "off", 0.9),
    ("imu/ds/ds_leaf_size", "imu/ds/mode != off",
     lambda r: r["imu/ds/mode"] != "off", 0.15),
    ("voxel_map/plane/weight_floor/constant", "voxel_map/plane/weight_floor/mode=constant|legacy",
     lambda r: r["voxel_map/plane/weight_floor/mode"] in ("constant", "legacy"), 1e-3),
    ("voxel_map/plane/weight_floor/incidence_k", "voxel_map/plane/weight_floor/mode=incidence",
     lambda r: r["voxel_map/plane/weight_floor/mode"] == "incidence", 1.0),
    ("voxel_map/plane/plane_var_denom_floor_en", "voxel_map/plane/plane_var_mode=eigengap",
     lambda r: r["voxel_map/plane/plane_var_mode"] == "eigengap", False),
    ("voxel_map/plane/use_bins", "voxel_map/plane/plane_fit_mode=pca",
     lambda r: r["voxel_map/plane/plane_fit_mode"] == "pca", False),
    ("voxel_map/plane/bin_size_fraction", "voxel_map/plane/use_bins=true",
     lambda r: r["voxel_map/plane/plane_fit_mode"] == "pca" and r["voxel_map/plane/use_bins"], 0.2),
    ("voxel_map/plane/bin_weight_mode_fit", "voxel_map/plane/use_bins=true",
     lambda r: r["voxel_map/plane/plane_fit_mode"] == "pca" and r["voxel_map/plane/use_bins"], "count"),
    ("voxel_map/plane/bin_weight_mode_var", "voxel_map/plane/use_bins=true",
     lambda r: r["voxel_map/plane/plane_fit_mode"] == "pca" and r["voxel_map/plane/use_bins"], "count"),
    ("voxel_map/plane/log_consistency_corr_stride", "voxel_map/plane/log_consistency_mode=corr|corr+covariates",
     lambda r: r["voxel_map/plane/log_consistency_mode"] in ("corr", "corr+covariates"), 1),
]

NESTED_KEYS = {row[0] for row in NESTING_TABLE}


def resolve_cell(declared: dict) -> dict:
    """Effective config for one cell: DEFAULTS + declared overrides, then
    every nesting rule applied in source order, forcing a dead key to its
    default regardless of what was declared. Returns {key: (value, forced)}."""
    resolved = dict(DEFAULTS)
    for k, v in declared.items():
        resolved[k] = v
    forced = {}
    for key, scope, live_pred, dead_default in NESTING_TABLE:
        if key not in resolved:
            resolved[key] = dead_default
        if not live_pred(resolved):
            # Matches config_resolve.h's nested(): the dead branch ALWAYS
            # marks the key FORCED, regardless of whether a value was ever
            # declared or what it was -- "denied" is denied either way, and
            # two cells that both got denied must produce the SAME marker
            # (and therefore the same digest) even if one of them never
            # bothered to declare the key at all.
            forced[key] = scope
            resolved[key] = dead_default
    return resolved, forced


def digest_cell(resolved: dict, forced: dict, key_universe) -> str:
    pairs = []
    for k in sorted(key_universe):
        v = resolved.get(k, DEFAULTS.get(k))
        marker = f"[FORCED:{forced[k]}]" if k in forced else ""
        pairs.append(f"{k}={v!r}{marker}")
    blob = "\n".join(pairs).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def flatten_yaml(doc, prefix=""):
    out = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.update(flatten_yaml(v, f"{prefix}{k}/" if prefix == "" else f"{prefix}{k}/"))
        return out
    else:
        return {prefix.rstrip("/"): doc}


def load_cells_from_config_dir(d):
    import os
    if yaml is None:
        print("ERROR: pyyaml not available; cannot parse --config-dir", file=sys.stderr)
        sys.exit(2)
    cells = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".yaml"):
            continue
        label = fn[:-len(".yaml")]
        with open(os.path.join(d, fn)) as fh:
            doc = yaml.safe_load(fh)
        cells[label] = flatten_yaml(doc or {})
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="", help="JSON: {label: {key/path: value, ...}, ...}")
    ap.add_argument("--config-dir", default="", help="directory of label.yaml files, one per cell")
    ap.add_argument("--base", default="", help="JSON of overrides applied to every cell before its own")
    args = ap.parse_args()

    if not args.cells and not args.config_dir:
        print("ERROR: need --cells or --config-dir", file=sys.stderr)
        sys.exit(2)

    if args.config_dir:
        cells = load_cells_from_config_dir(args.config_dir)
    else:
        with open(args.cells) as fh:
            cells = json.load(fh)

    base_overrides = {}
    if args.base:
        with open(args.base) as fh:
            base_overrides = json.load(fh)

    key_universe = set(DEFAULTS) | set(base_overrides)
    for decl in cells.values():
        key_universe |= set(decl)

    print(f"=== assert_design.py: {len(cells)} cells, {len(key_universe)} keys in the union ===")

    by_digest = defaultdict(list)
    per_cell_forced = {}
    invalid_cells = {}  # CQ-32 item 4: label->effective round-trip
    for label, decl in cells.items():
        merged = dict(base_overrides)
        merged.update(decl)
        resolved, forced = resolve_cell(merged)
        d = digest_cell(resolved, forced, key_universe)
        by_digest[d].append(label)
        per_cell_forced[label] = forced
        if forced:
            for k, scope in forced.items():
                print(f"  {label}: {k} FORCED by dead scope '{scope}' "
                      f"(declared {merged.get(k)!r}, effective {resolved[k]!r})")
        # item 4: for every key THIS cell's own label explicitly declared,
        # a dead-scope force means the label's intent and the effective
        # config disagree -- that is a round-trip failure, not merely a
        # note, and makes the cell INVALID under the n_valid/n_designed
        # rule regardless of whether it also collides with another label's
        # digest (a cell can fail its own round-trip with no collision at
        # all, e.g. if it is the only cell ever declaring that factor).
        mismatches = {k: (decl[k], resolved[k]) for k in decl
                      if k in forced and decl[k] != resolved[k]}
        if mismatches:
            invalid_cells[label] = mismatches

    n_refused = 0
    for d, labels in by_digest.items():
        if len(labels) > 1:
            n_refused += 1
            # Name the factors these labels actually differ in.
            decls = [cells[l] for l in labels]
            diff_keys = set()
            for i in range(1, len(decls)):
                for k in set(decls[0]) | set(decls[i]):
                    if decls[0].get(k) != decls[i].get(k):
                        diff_keys.add(k)
            print(f"REFUSED: {len(labels)} distinct labels resolve to the SAME effective "
                  f"digest {d[:12]}...: {', '.join(labels)}", file=sys.stderr)
            print(f"  they claim to differ in: {sorted(diff_keys) or '(nothing declared differently)'}",
                  file=sys.stderr)
            for l in labels:
                if per_cell_forced[l]:
                    print(f"  {l} forced: {per_cell_forced[l]}", file=sys.stderr)

    if invalid_cells:
        print(f"\nINVALID (label != effective, item 4's round-trip): {len(invalid_cells)} of {len(cells)} cells")
        for label, mism in invalid_cells.items():
            for k, (declared_v, effective_v) in mism.items():
                print(f"  {label}: declared {k}={declared_v!r}, but effective is {effective_v!r} "
                      f"(dead scope) -- INVALID")

    print(f"\n{len(by_digest)} distinct effective configs over {len(cells)} cells; "
          f"{n_refused} digest collision(s) between distinct labels; "
          f"{len(invalid_cells)} cell(s) INVALID under the label-round-trip check")
    if n_refused or invalid_cells:
        sys.exit(1)
    print("OK: every distinct label resolves to a distinct effective config, "
          "and every cell's label round-trips to its effective value.")
    sys.exit(0)


if __name__ == "__main__":
    main()
