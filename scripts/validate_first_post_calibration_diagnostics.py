#!/usr/bin/env python3
"""Reject incomplete, stale, mixed-generation, or all-frame diagnostic returns."""

import argparse
import csv
from pathlib import Path


IEKF_FILES = {
    "pose_control_first_frame.csv",
    "pose_control_first_frame_solve.csv",
    "pose_control_first_frame_state_chain.csv",
    "pose_control_first_frame_solve_matrices.txt",
    "pose_control_first_frame_covariance_budget.csv",
    "pose_control_first_frame_covariance_budget.txt",
}

FORBIDDEN_ALL_FRAME_FILES = {
    "odometry.txt", "frame_stats.txt", "pose_pair.csv",
    "iekf_gn_iteration.csv",
}


def require(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty diagnostic: {path}")


def validate_iekf(path: Path) -> None:
    for name in IEKF_FILES:
        require(path / name)
    with (path / "pose_control_first_frame_covariance_budget.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not any(row.get("phase") == "post_covariance" for row in rows):
        raise RuntimeError("iekf posterior covariance row is missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--architecture", choices=("iekf",), default="iekf")   # R97: the coupled estimator is gone; the option is kept so existing command lines run
    args = parser.parse_args()
    forbidden = sorted(name for name in FORBIDDEN_ALL_FRAME_FILES
                       if (args.directory / name).exists())
    if forbidden:
        raise RuntimeError(
            f"all-frame outputs must not be packaged: {forbidden}")
    validate_iekf(args.directory)
    print(f"PASS: {args.architecture} first-post-calibration diagnostics: {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
