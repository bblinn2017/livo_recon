#!/usr/bin/env python3
"""R83 selftest of the collector changes (synthetic arm dirs only, planner sandbox, not evidence about the node). The compact copies below are written only with --r83-extras; the default run returns none of them.
   python3 tools/r69/selftest_r83_collect.py
Checks: D1B returns point_cov_window.csv as compact/<arm>/point_cov_window.csv.gz (gzip copy, readable, same rows) and no other group does; D4P / OWN return peak_rss.txt; peak_rss_all.csv lists the arms that have the file and flags the
ABORTED_RSS marker; MQS arms return motion_q_scan_all.csv.gz and an ordinary arm does not; DST arms return residual_dump_all.csv.gz (the voxel_layer column survives); VSW arms are summaries only (pos_trace only when gt_max_mm > 500)."""
import gzip, json, os, subprocess, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r69_collect as C  # noqa: E402


def mk(runs, arm, files=None, gt_max=1.0):
    d = os.path.join(runs, arm); os.makedirs(d, exist_ok=True)
    json.dump({"arm": arm, "kind": "closed", "meta": {"exit_code": 0}, "gt_max_mm": gt_max, "n_scans": 3}, open(os.path.join(d, "arm_summary.json"), "w"))
    json.dump({"sentinel": "DONE"}, open(os.path.join(d, "arm.json"), "w"))
    open(os.path.join(d, "state_trace_blank_runid.sha256"), "w").write("ab" * 32 + "\n")
    pd.DataFrame({"scan_id": [1, 2, 3], "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    pd.DataFrame({"scan": [1, 2, 3], "t": ["1.0", "2.0", "3.0"], "ep_x_mm": 0.0, "ep_y_mm": 0.0, "ep_z_mm": 0.0}).to_csv(os.path.join(d, "arm_scan.csv"), index=False)
    for k, v in (files or {}).items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(d, k), index=False, compression="gzip" if k.endswith(".gz") else None)
        else:
            open(os.path.join(d, k), "w").write(v)


def collect(runs, tp, out, *extra):
    return subprocess.run([sys.executable, os.path.join(HERE, "r69_collect.py"), "--runs", runs, "--arms", tp, "--out", out, "--level", "-1", *extra], capture_output=True, text=True)


def main():
    fails = []; n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            fails.append(m)
    # unit level: return_file (with the extras on, as the checks below describe; the default is off)
    C.R83_EXTRAS = True
    ck(C.return_file("motion_q_scan_all.csv.gz", "r83_rt_mqs_eee_01_g3_a0p15_mq", "MQS") and not C.return_file("motion_q_scan_all.csv.gz", "r83_rt_ml_eee_01_l2_mq", "MXL"), "motion_q_scan_all.csv.gz only for _mqs_ arms")
    ck(C.return_file("motion_q_scan_all.csv.gz", "r80_b3_eee_02_mq", "B3") and not C.return_file("motion_q_scan_all.csv.gz", "r80_b3_eee_01_mq", "B3"), "the R80 single-arm rule still holds")
    ck(C.return_file("point_cov_window.csv.gz", "a", "D1B") and not C.return_file("point_cov_window.csv.gz", "a", "MXL") and not C.return_file("point_cov_window.csv.gz", "a", "DST"), "point_cov_window.csv.gz only for D1B (and OWN)")
    ck(C.return_file("peak_rss.txt", "a", "D4P") and not C.return_file("peak_rss.txt", "a", "MQS"), "peak_rss.txt only for D4P (and OWN)")
    ck(not C.return_file("peak_rss.txt", "a", "VSW") and not C.return_file("residual_dump_all.csv.gz", "a", "VSW") and C.return_file("pos_trace_all.csv.gz", "a", "VSW", 900.0) and not C.return_file("pos_trace_all.csv.gz", "a", "VSW", 10.0), "VSW is a summary group")
    ck(C.return_file("residual_dump_all.csv.gz", "a", "DST"), "DST returns the residual dump")
    C.R83_EXTRAS = False
    ck(not C.return_file("point_cov_window.csv.gz", "a", "D1B") and not C.return_file("peak_rss.txt", "a", "D4P") and not C.return_file("motion_q_scan_all.csv.gz", "r83_x_mqs_arm", "MQS") and C.return_file("residual_dump_all.csv.gz", "a", "DST"),
       "default (archive items): none of the three is returned, the DST dump still is")
    t = tempfile.mkdtemp(); runs = os.path.join(t, "runs"); out = os.path.join(t, "out")
    pcw = "scan_index,t_abs,t_rel_s,window_s,n_steps,sd_p_med_m\n" + "".join("%d,%d,%d,0.02,4,0.001\n" % (i, i, i) for i in range(1, 8))
    mk(runs, "pc_d1b", {"point_cov_window.csv": pcw})
    mk(runs, "pc_other", {"point_cov_window.csv": pcw})
    mk(runs, "mem_d4p", {"peak_rss.txt": "peak_rss_kb 5555\n"})
    mk(runs, "mem_abort", {"peak_rss.txt": "peak_rss_kb 9999\n", "ABORTED_RSS": "x"})
    mk(runs, "mq_arm", {"motion_q_scan_all.csv.gz": pd.DataFrame({"scan_index": [1, 2], "acc_scale": 0.15, "gyro_scale": 3.0})})
    mk(runs, "mqs_other", {"motion_q_scan_all.csv.gz": pd.DataFrame({"scan_index": [1, 2]})})
    mk(runs, "r83_x_mqs_arm", {"motion_q_scan_all.csv.gz": pd.DataFrame({"scan_index": [1, 2], "acc_scale": 0.15, "gyro_scale": 3.0})})
    mk(runs, "dst_arm", {"residual_dump_all.csv.gz": pd.DataFrame({"frame": [5, 5], "iter": [0, 0], "voxel_layer": [1, 2]})})
    mk(runs, "vsw_lo", {"residual_dump_all.csv.gz": pd.DataFrame({"frame": [5]})}, gt_max=10.0)
    mk(runs, "vsw_hi", {}, gt_max=900.0)
    tab = pd.DataFrame([dict(arm=a, tier="FB", kind="closed", duration_s="full", seq="eee_01", keep_raw=0, group=g_, dup_of="", notes="") for a, g_ in
                        (("pc_d1b", "D1B"), ("pc_other", "MXL"), ("mem_d4p", "D4P"), ("mem_abort", "D4P"), ("mq_arm", "MXL"), ("mqs_other", "MQS"), ("r83_x_mqs_arm", "MQS"), ("dst_arm", "DST"), ("vsw_lo", "VSW"), ("vsw_hi", "VSW"))])
    tp = os.path.join(t, "arm_table.tsv"); tab.to_csv(tp, sep="\t", index=False)
    r0 = collect(runs, tp, out + "_default")
    ck(r0.returncode == 0 and not any(os.path.exists(os.path.join(out + "_default", "compact", a, f)) for a, f in (("pc_d1b", "point_cov_window.csv.gz"), ("mem_d4p", "peak_rss.txt"), ("r83_x_mqs_arm", "motion_q_scan_all.csv.gz")))
       and os.path.exists(os.path.join(out + "_default", "peak_rss_all.csv")), "default run (no --r83-extras): the three archive items are not returned, peak_rss_all.csv still written: %s" % r0.stderr[-300:])
    r = collect(runs, tp, out, "--r83-extras")
    ck(r.returncode == 0, "collector exit %s: %s" % (r.returncode, r.stderr[-400:]))
    has = lambda a, f: os.path.exists(os.path.join(out, "compact", a, f))
    ck(has("pc_d1b", "point_cov_window.csv.gz") and not has("pc_other", "point_cov_window.csv.gz"), "point_cov_window.csv.gz for the D1B arm only")
    if has("pc_d1b", "point_cov_window.csv.gz"):
        ck(len(pd.read_csv(os.path.join(out, "compact", "pc_d1b", "point_cov_window.csv.gz"))) == 7, "gzip copy keeps every row")
    ck(has("mem_d4p", "peak_rss.txt") and has("mem_abort", "peak_rss.txt") and not has("pc_d1b", "peak_rss.txt"), "peak_rss.txt for the D4P arms")
    ck(has("r83_x_mqs_arm", "motion_q_scan_all.csv.gz") and not has("mqs_other", "motion_q_scan_all.csv.gz") and not has("mq_arm", "motion_q_scan_all.csv.gz"), "motion_q_scan_all.csv.gz only for names with _mqs_")
    ck(has("dst_arm", "residual_dump_all.csv.gz") and "voxel_layer" in pd.read_csv(os.path.join(out, "compact", "dst_arm", "residual_dump_all.csv.gz")).columns, "DST dump returned with its voxel_layer column")
    ck(not has("vsw_lo", "residual_dump_all.csv.gz") and not has("vsw_lo", "pos_trace_all.csv.gz") and has("vsw_hi", "pos_trace_all.csv.gz"), "VSW: summaries only, trace only when gt_max_mm > 500")
    pr = os.path.join(out, "peak_rss_all.csv")
    if os.path.exists(pr):
        R = {x["arm"]: x for x in pd.read_csv(pr).to_dict("records")}
        ck(set(R) == {"mem_d4p", "mem_abort"} and R["mem_abort"]["aborted"] == 1 and R["mem_d4p"]["aborted"] == 0, "peak_rss_all.csv rows and the aborted flag: %r" % R)
    else:
        ck(False, "peak_rss_all.csv not written")
    ck(all(os.path.exists(os.path.join(out, f)) for f in ("phase_summary.csv", "completeness.csv", "summaries_all.json")), "the usual collector outputs")
    print("SELFTEST r83_collect", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (n[0], len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
