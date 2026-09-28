#!/usr/bin/env python3
"""Reject incomplete, stale, mixed-generation, or all-frame diagnostic returns."""

import argparse
import csv
from pathlib import Path


COUPLED_CSV = {
    "joint_knot_iterations.csv": "scan_id",
    "joint_knot_corrections.csv": "scan_id",
    "joint_knot_states.csv": "scan_id",
    "joint_knot_scan_summary.csv": "scan_id",
    "joint_knot_first_frame_state_chain.csv": "scan_id",
}

COUPLED_FILES = {
    "joint_knot_first_frame_matrices.txt",
    "joint_knot_first_frame_covariances.txt",
}

DECOUPLED_FILES = {
    "pose_control_first_frame.csv",
    "pose_control_first_frame_solve.csv",
    "pose_control_first_frame_state_chain.csv",
    "pose_control_first_frame_solve_matrices.txt",
    "pose_control_first_frame_covariance_budget.csv",
    "pose_control_first_frame_covariance_budget.txt",
}

FORBIDDEN_ALL_FRAME_FILES = {
    "odometry.txt", "frame_stats.txt", "pose_pair.csv",
    "decoupled_gn_iteration.csv", "decoupled_knot_update.csv",
}


def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty diagnostic: {path}")


def validate_coupled(path: Path) -> None:
    for name in COUPLED_FILES:
        require(path / name)
    for name, key in COUPLED_CSV.items():
        file_path = path / name
        require(file_path)
        with file_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"no data rows: {file_path}")
        scan_ids = {int(row[key]) for row in rows}
        if scan_ids != {1}:
            raise RuntimeError(
                f"{file_path} is not first-frame-only; scan IDs={sorted(scan_ids)}")
    with (path / "joint_knot_states.csv").open(newline="") as stream:
        state_rows = list(csv.DictReader(stream))
    required_state_columns = {
        "knot_index", "knot_t", "immutable_head",
        "theta_x", "theta_y", "theta_z",
        "p_x", "p_y", "p_z", "v_x", "v_y", "v_z",
        "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z",
        "g_x", "g_y", "g_z",
    }
    missing_state_columns = required_state_columns.difference(state_rows[0])
    if missing_state_columns:
        raise RuntimeError(
            f"joint-knot state columns missing: {sorted(missing_state_columns)}")
    if not any(row["knot_index"] == "0" and row["immutable_head"] == "1"
               for row in state_rows):
        raise RuntimeError("joint-knot state log does not identify the immutable head")
    with (path / "joint_knot_first_frame_state_chain.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "first_post_calibration_lio_frame", "map_seed_complete",
        "imu_propagated", "n_imu_samples", "scan_start_t", "scan_tail_t",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"state-chain metadata missing: {sorted(missing)}")
    if any(row["first_post_calibration_lio_frame"] != "1" or
           row["map_seed_complete"] != "1" or
           row["imu_propagated"] != "1" or
           int(row["n_imu_samples"]) <= 0 for row in rows):
        raise RuntimeError("state chain does not prove a propagated post-calibration frame")
    covariance_text = (path / "joint_knot_first_frame_covariances.txt").read_text()
    for marker in (
            "knot_specific_biases 1",
            "full_production_process_covariance 1",
            "P_prior_tail_marginal ",
            "P_prior_tail_vs_state_post_imu_frobenius "):
        if marker not in covariance_text:
            raise RuntimeError(
                f"knot-specific-bias covariance evidence missing: {marker}")


def validate_decoupled(path: Path) -> None:
    for name in DECOUPLED_FILES:
        require(path / name)
    with (path / "pose_control_first_frame_covariance_budget.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not any(row.get("phase") == "post_covariance" for row in rows):
        raise RuntimeError("decoupled posterior covariance row is missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--architecture", choices=("coupled", "decoupled"), required=True)
    args = parser.parse_args()
    forbidden = sorted(name for name in FORBIDDEN_ALL_FRAME_FILES
                       if (args.directory / name).exists())
    if forbidden:
        raise RuntimeError(
            f"all-frame outputs must not be packaged: {forbidden}")
    (validate_coupled if args.architecture == "coupled" else validate_decoupled)(
        args.directory)
    print(f"PASS: {args.architecture} first-post-calibration diagnostics: {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
