#!/usr/bin/env python3
"""selftest_r83_hilti_point_time: the Hilti config always uses the point's own time (F-504, R83).

Pure Python, no ROS. Checks the yaml keys only; the C++ key parser is unchanged.
  python3 tools/r69/selftest_r83_hilti_point_time.py --config-dir config
"""
import argparse, os, sys
import yaml


def lidar(cfg):
    return ((cfg or {}).get("cbk") or {}).get("lidar") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default="config")
    a = ap.parse_args()
    fails = []
    h = lidar(yaml.safe_load(open(os.path.join(a.config_dir, "slam_2022_handheld.yaml"))))
    if h.get("point_time_mode") != "absolute_s_f64":
        fails.append("slam_2022_handheld.yaml cbk.lidar.point_time_mode is %r, want absolute_s_f64" % h.get("point_time_mode"))
    if h.get("point_time_field") != "timestamp":
        fails.append("slam_2022_handheld.yaml cbk.lidar.point_time_field is %r, want timestamp" % h.get("point_time_field"))
    if "point_time_ref" in h:
        fails.append("point_time_ref must stay unset (default header, what R82 pt-on ran with); found %r" % h.get("point_time_ref"))
    n = lidar(yaml.safe_load(open(os.path.join(a.config_dir, "ntu_viral.yaml"))))
    for k in ("point_time_mode", "point_time_field", "point_time_ref"):
        if k in n:
            fails.append("ntu_viral.yaml must not set cbk.lidar.%s (NTU stays on offset_ns_u32, byte-identical to R82)" % k)
    for f in fails:
        print("FAIL", f)
    print("selftest_r83_hilti_point_time:", "FAILED" if fails else "ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
