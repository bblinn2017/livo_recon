#!/usr/bin/env python3
"""Analyze real 30 s stationary runs from the new all-scan diagnostics.

This script intentionally does not fabricate measurements.  Pass each run as
LABEL=DEBUG_LOG_DIRECTORY.  It compares final-per-scan stationary errors,
per-iteration requested-versus-realized pose motion, coupled knot updates, and
paired gating identities.  Raw all-frame CSVs may stay at the execution site;
redirect this script's compact output into the returned evidence package.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def floats(items, key):
    return [float(item[key]) for item in items]


def summarize(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "max": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def last_iteration_per_scan(items):
    result = {}
    for item in items:
        key = int(item["scan_id"])
        if key not in result or int(item["iteration"]) > int(result[key]["iteration"]):
            result[key] = item
    return list(result.values())


def analyze_run(directory):
    stationary = rows(directory / "stationary_iteration.csv")
    finals = last_iteration_per_scan(stationary)
    result = {
        "scan_count": len(finals),
        "position_distance": summarize(floats(finals, "after_pos_distance")),
        "attitude_distance_rad": summarize(floats(finals, "after_attitude_distance")),
        "gravity_tilt_error_rad": summarize(floats(finals, "after_gravity_tilt_error")),
        "velocity_distance": summarize(floats(finals, "after_velocity_distance")),
        "speed": summarize(floats(finals, "after_speed")),
        "ideal_position_request": summarize(floats(stationary, "ideal_dp_norm")),
        "realized_position_step": summarize(floats(stationary, "realized_dp_norm")),
        "ideal_attitude_request_rad": summarize(floats(stationary, "ideal_dtheta_norm")),
        "realized_attitude_step_rad": summarize(floats(stationary, "realized_dtheta_norm")),
    }
    gating_path = directory / "gating_all_scans.csv"
    if gating_path.exists():
        gating = rows(gating_path)
        result["gating"] = {
            "row_count": len(gating),
            "statistical_candidates": sum(int(r["statistical_candidates"]) for r in gating),
            "statistical_rejections": sum(int(r["statistical_gate_rejections"]) for r in gating),
            "accepted": sum(int(r["accepted_count"]) for r in gating),
            "mean_gate_cov_trace": summarize(floats(gating, "mean_gating_cov_trace")),
        }
    knots_path = directory / "joint_knot_all_scans.csv"
    if knots_path.exists():
        knots = rows(knots_path)
        result["knots"] = {
            "dp": summarize(floats(knots, "dp_norm")),
            "dtheta_rad": summarize(floats(knots, "dtheta_norm")),
            "dv": summarize(floats(knots, "dv_norm")),
            "adjacent_bg": summarize(floats(knots, "adjacent_bg_difference")),
            "adjacent_ba": summarize(floats(knots, "adjacent_ba_difference")),
        }
    return result


def compare_gating(off_dir, on_dir):
    off = {(r["scan_id"], r["iteration"]): r
           for r in rows(off_dir / "gating_all_scans.csv")}
    on = {(r["scan_id"], r["iteration"]): r
          for r in rows(on_dir / "gating_all_scans.csv")}
    keys = sorted(set(off) & set(on), key=lambda k: (int(k[0]), int(k[1])))
    identity_fields = ("accepted_count", "accepted_index_hash", "solve_input_hash")
    differences = []
    for key in keys:
        changed = [field for field in identity_fields if off[key][field] != on[key][field]]
        if changed:
            differences.append({"scan_id": key[0], "iteration": key[1], "fields": changed})
    return {
        "off_rows": len(off), "on_rows": len(on), "paired_rows": len(keys),
        "different_rows": len(differences), "first_differences": differences[:20],
        "identical_admission_over_all_paired_rows": len(differences) == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", metavar="LABEL=DIR")
    parser.add_argument("--gating-pair", action="append", default=[],
                        metavar="OFF_LABEL,ON_LABEL")
    args = parser.parse_args()
    directories = {}
    for specification in args.runs:
        label, raw_path = specification.split("=", 1)
        directories[label] = Path(raw_path)
    report = {"runs": {label: analyze_run(path)
                       for label, path in directories.items()}}
    report["gating_pairs"] = {}
    for pair in args.gating_pair:
        off, on = pair.split(",", 1)
        report["gating_pairs"][pair] = compare_gating(directories[off], directories[on])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
