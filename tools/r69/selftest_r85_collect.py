#!/usr/bin/env python3
"""R85 selftest of the collector changes (synthetic arm dirs only, planner sandbox, not evidence about the node).
   python3 -I tools/r69/selftest_r85_collect.py
Checks: the R85 groups L0, L2R, SHT are known groups; L0 and L2R are sweep groups (summaries only, pos_trace_all.csv.gz only when gt_max_mm > 500, a collapsed arm's trace is returned);
SHT and OWN arms return the F1_KEEP files including the per-scan short_scan_all.csv.gz (copied into compact/<arm>/ by the "*_all.csv.gz" pattern; it is not dropped by DROP_FROM_RETURN or
GROUP_ONLY_FILES for SHT / OWN, and is not returned for a sweep group); the new arm_table columns (fit lines edge layer cell nb l2_ref l2_source) pass through the collector; an identity arm,
a whole-bag determinism arm (det_fb) and a short determinism arm (det_sht) reach identity_controls.txt; a refuse arm (arm.json only) still gets a phase_summary row; R83 extras
(point_cov_window, peak_rss, motion-Q) are inert by default; every group of the real r85 arm_table (r85_manifest/arm_table.tsv two levels above this file, when present) is a known group."""
import json, os, subprocess, sys, tempfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r69_collect as C  # noqa: E402
REAL_TABLE = os.path.join(HERE, "..", "..", "r85_manifest", "arm_table.tsv")
SHA = "ab" * 32


def mk(runs, arm, files=None, gt_max=1.0, summary=True, sha=SHA):
    d = os.path.join(runs, arm); os.makedirs(d, exist_ok=True)
    if summary:
        json.dump({"arm": arm, "kind": "closed", "meta": {"exit_code": 0}, "gt_max_mm": gt_max, "gt_ate_mm": 3.0, "n_scans": 3}, open(os.path.join(d, "arm_summary.json"), "w"))
        pd.DataFrame({"scan": [1, 2, 3], "t": ["1.0", "2.0", "3.0"], "ep_x_mm": 0.0, "ep_y_mm": 0.0, "ep_z_mm": 0.0}).to_csv(os.path.join(d, "arm_scan.csv"), index=False)
        pd.DataFrame({"scan_id": [1, 2, 3], "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    json.dump({"sentinel": "DONE", "exit_code": 0}, open(os.path.join(d, "arm.json"), "w"))
    if sha:
        open(os.path.join(d, "state_trace_blank_runid.sha256"), "w").write(sha + "\n")
    for k, v in (files or {}).items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(d, k), index=False, compression="gzip" if k.endswith(".gz") else None)
        else:
            open(os.path.join(d, k), "w").write(v)


def collect(runs, tp, out, *extra):
    return subprocess.run([sys.executable, "-I", os.path.join(HERE, "r69_collect.py"), "--runs", runs, "--arms", tp, "--out", out, "--level", "-1", *extra], capture_output=True, text=True)


def main():
    fails = []; n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            fails.append(m)
    # unit level
    ck({"L0", "L2R", "SHT", "OWN"} <= C.F1_GROUPS, "L0, L2R, SHT, OWN are known groups")
    ck({"L0", "L2R"} <= C.SW_GROUPS and "SHT" not in C.SW_GROUPS and "OWN" not in C.SW_GROUPS, "L0 and L2R are sweep groups, SHT and OWN are not")
    for g_ in ("L0", "L2R"):
        ck(not C.return_file("residual_dump_all.csv.gz", "a", g_) and not C.return_file("lidar_msg_log_all.csv.gz", "a", g_) and not C.return_file("line_rows_per_scan_all.csv.gz", "a", g_) and not C.return_file("short_scan_all.csv.gz", "a", g_), "%s: summaries only" % g_)
        ck(C.return_file("pos_trace_all.csv.gz", "a", g_, 900.0) and not C.return_file("pos_trace_all.csv.gz", "a", g_, 500.0) and not C.return_file("pos_trace_all.csv.gz", "a", g_, 10.0) and C.return_file("pos_trace_all.csv.gz", "a", g_, None), "%s: trace only when gt_max_mm > 500 (500.0 itself no; missing yes)" % g_)
    ck("short_scan_all.csv.gz" in C.F1_KEEP and "short_scan_all.csv.gz" not in C.DROP_FROM_RETURN, "short_scan_all.csv.gz is in F1_KEEP and not in DROP_FROM_RETURN")
    for g_ in ("SHT", "OWN"):
        ck(all(C.return_file(f, "r85_rt_sht_eee_01_v0p2_A", g_) for f in ("short_scan_all.csv.gz", "pos_trace_all.csv.gz", "line_rows_per_scan_all.csv.gz", "state_reference.txt")), "%s returns the F1_KEEP files incl. short_scan_all.csv.gz" % g_)
    ck(not C.return_file("short_scan_all.csv.gz", "a", "FB") and not C.return_file("short_scan_all.csv.gz", "a", "DP"), "short_scan_all.csv.gz is bound to SHT / OWN (second guard)")
    ck(not C.return_file("point_cov_window.csv.gz", "a", "SHT") and not C.return_file("peak_rss.txt", "a", "OWN") and not C.return_file("motion_q_scan_all.csv.gz", "r85_x_mqs_arm", "SHT"), "R83 extras are inert by default")
    ck(C.return_file("pos_trace_all.csv.gz", "a", "SW", 900.0) and not C.return_file("pos_trace_all.csv.gz", "a", "SW", 10.0) and C.return_file("pos_trace_all.csv.gz", "a", "VSW", 900.0), "older sweep groups unchanged")
    # real manifest groups, when present
    if os.path.exists(REAL_TABLE):
        rt = pd.read_csv(REAL_TABLE, sep="\t")
        ck(set(rt["group"].dropna().astype(str)) <= C.F1_GROUPS and {"fit", "lines", "edge", "layer", "cell", "nb", "l2_ref", "l2_source"} <= set(rt.columns), "every group of the real arm_table is known, new columns present")
        ck(sum(1 for _, r_ in rt.iterrows() if C.return_file("short_scan_all.csv.gz", r_["arm"], str(r_["group"]))) == 144 + 26, "short_scan_all.csv.gz is returnable for the 144 SHT arms and the 26 OWN arms of the real arm_table (per-arm presence is decided by the files)")
    else:
        print("INFO real arm_table not found at", REAL_TABLE)
    # collector run on synthetic arms
    t = tempfile.mkdtemp(); runs = os.path.join(t, "runs"); out = os.path.join(t, "out")
    msg = pd.DataFrame({"msg_index": [1, 2], "seq": [1, 2]})
    rowsps = pd.DataFrame({"scan": [1, 2], "n": [3, 4]})
    short = pd.DataFrame({"scan": [1, 2, 3], "t_s": [0.0, 0.1, 0.2], "residual_count": [10, 11, 12], "fs_n_residuals": [10, 11, 12], "fs_n_voxels": [5, 5, 6], "fs_n_voxels_is_plane": [3, 3, 4],
                          "fs_iters": [2, 2, 2], "fs_dx_pos_mm": [0.1, 0.1, 0.1], "fs_nis": [1.0, 1.0, 1.0], "fs_reduced_chi2": [1.0, 1.0, 1.0], "pos_err_mm": [0.0, 0.1, 0.2]})
    A_ = "r85_rt_sht_eee_01_v0p25_A"
    arms = [  # arm, group, extra columns, files, gt_max, summary
        ("r85_dc_l0_eee_01_pca_off_v0p25_fq", "L0", dict(arm_kind="fb", fit="pca", lines="off", edge=0.25, layer=0, l2_ref="r81_x", l2_source="R81"), {"residual_dump_all.csv.gz": pd.DataFrame({"frame": [1]})}, 10.0, True),
        ("r85_rt_l0_eee_01_deb_on_v0p2_fq", "L0", dict(arm_kind="fb", fit="deb", lines="on", edge=0.2, layer=0, l2_ref="r85_rt_l2_eee_01_deb_on_v0p2_fq", l2_source="R85"), {}, 900.0, True),
        ("r85_rt_l2_eee_01_pca_off_v0p2_fq", "L2R", dict(arm_kind="fb", fit="pca", lines="off", edge=0.2, layer=2), {"lidar_msg_log_all.csv.gz": msg}, 20.0, True),
        ("r85_rt_l2_eee_01_deb_off_v0p25_fq", "L2R", dict(arm_kind="fb", fit="deb", lines="off", edge=0.25, layer=2), {}, 750.0, True),
        (A_, "SHT", dict(arm_kind="sht", fit="deb", lines="on", edge=0.25, layer=0, cell="A", nb=1),
         {"short_scan_all.csv.gz": short, "line_rows_per_scan_all.csv.gz": rowsps, "state_reference.txt": "ref\n", "residual_dump_all.csv.gz": pd.DataFrame({"frame": [1]}), "point_cov_window.csv": "scan_index,t_abs\n1,1\n", "peak_rss.txt": "peak_rss_kb 1\n",
          "motion_q_scan_all.csv.gz": pd.DataFrame({"scan_index": [1]})}, 4.0, True),
        ("r85_rt_sht_eee_01_v0p25_H", "SHT", dict(arm_kind="sht", fit="deb", lines="on", edge=0.25, layer=0, cell="H", nb=2), {"short_scan_all.csv.gz": short}, 4.0, True),
        ("r85_own_det_s_eee_01", "OWN", dict(arm_kind="det_sht", ref=A_, cell="A"), {"short_scan_all.csv.gz": short}, 4.0, True),
        ("r85_own_det_x_eee_01", "OWN", dict(arm_kind="det_fb", ref="r85_rt_l0_eee_01_deb_on_v0p2_fq"), {}, 4.0, True),
        ("r85_rt_id_eee_01_fq", "OWN", dict(arm_kind="id", ref="r84_rt_id_eee_01_fq"), {}, 4.0, True),
        ("r85_dc_refuse_sample_window", "OWN", dict(arm_kind="refuse", needles="sample_window_s"), {}, None, False),
    ]
    recs = []
    for arm, g_, ex, files, gt, summ in arms:
        mk(runs, arm, files, gt_max=gt, summary=summ, sha=(SHA if arm != "r85_dc_refuse_sample_window" else None))
        r_ = dict(arm=arm, tier="FB", kind="closed", duration_s=("5" if "refuse" in arm else ("60" if ("_sht_" in arm or "det_s_" in arm) else "full")), seq="eee_01", keep_raw=0, group=g_, dup_of="", notes="")
        r_.update(ex); recs.append(r_)
    tab = pd.DataFrame(recs)
    tp = os.path.join(t, "arm_table.tsv"); tab.to_csv(tp, sep="\t", index=False)
    refc = os.path.join(t, "ref_completeness.csv")
    pd.DataFrame([dict(arm="r84_rt_id_eee_01_fq", state_trace_sha256_run_id_blanked=SHA)]).to_csv(refc, index=False)
    r = collect(runs, tp, out, "--ref-completeness", refc)
    ck(r.returncode == 0, "collector exit %s: %s" % (r.returncode, r.stderr[-400:]))
    has = lambda a, f: os.path.exists(os.path.join(out, "compact", a, f))
    ck(not has("r85_dc_l0_eee_01_pca_off_v0p25_fq", "pos_trace_all.csv.gz") and not has("r85_dc_l0_eee_01_pca_off_v0p25_fq", "residual_dump_all.csv.gz"), "L0 gt_max 10: no per-arm file returned")
    ck(has("r85_rt_l0_eee_01_deb_on_v0p2_fq", "pos_trace_all.csv.gz"), "L0 collapsed arm (gt_max 900): trace returned")
    ck(not has("r85_rt_l2_eee_01_pca_off_v0p2_fq", "pos_trace_all.csv.gz") and not has("r85_rt_l2_eee_01_pca_off_v0p2_fq", "lidar_msg_log_all.csv.gz"), "L2R gt_max 20: summaries only")
    ck(has("r85_rt_l2_eee_01_deb_off_v0p25_fq", "pos_trace_all.csv.gz"), "L2R collapsed arm (gt_max 750): trace returned")
    ck(all(has(A_, f) for f in ("short_scan_all.csv.gz", "pos_trace_all.csv.gz", "line_rows_per_scan_all.csv.gz", "state_reference.txt")), "SHT arm: F1_KEEP files and short_scan_all.csv.gz returned into compact/<arm>/")
    ck(has("r85_rt_sht_eee_01_v0p25_H", "short_scan_all.csv.gz") and has("r85_own_det_s_eee_01", "short_scan_all.csv.gz"), "SHT cell H arm and the OWN short determinism arm: short_scan_all.csv.gz returned")
    ck(pd.read_csv(os.path.join(out, "compact", A_, "short_scan_all.csv.gz")).shape == (3, 11), "the returned short_scan_all.csv.gz is the arm's file (3 rows, 11 columns)")
    ck(not has(A_, "point_cov_window.csv.gz") and not has(A_, "peak_rss.txt") and not has(A_, "motion_q_scan_all.csv.gz"), "SHT arm: point_cov_window, peak_rss, motion-Q not returned")
    ps = pd.read_csv(os.path.join(out, "phase_summary.csv"))
    ck(set(ps["arm"]) == {a[0] for a in arms}, "phase_summary.csv has a row for every arm incl. the refuse arm (arm.json only): %s" % sorted(ps["arm"]))
    sc = os.path.join(out, "arm_scan_all.csv.gz")
    if os.path.exists(sc):
        sa = set(pd.read_csv(sc)["arm"])
        ck(not any("_l0_" in a or "_l2_" in a for a in sa) and A_ in sa, "arm_scan_all.csv.gz: no sweep (L0/L2R) arm, SHT arm present")
    else:
        ck(False, "arm_scan_all.csv.gz not written")
    ic = open(os.path.join(out, "identity_controls.txt")).read()
    ck("determinism: " + SHA in ic and "r85_own_det_x_eee_01" in ic and "r85_rt_l0_eee_01_deb_on_v0p2_fq" in ic, "identity_controls.txt lists the whole-bag determinism pair (det_fb): %s" % ic[-600:])
    ck("r85_own_det_s_eee_01" in ic and A_ in ic, "identity_controls.txt lists the short determinism pair (det_sht)")
    ck("identity: " + SHA in ic and "r85_rt_id_eee_01_fq" in ic and "r84_rt_id_eee_01_fq" in ic and ic.count("IDENTICAL") >= 3, "identity_controls.txt lists the identity arm against the R84 reference digest (--ref-completeness) as IDENTICAL: %s" % ic[-600:])
    cp = pd.read_csv(os.path.join(out, "completeness.csv"))
    ck(bool(cp.set_index("arm").loc["r85_dc_refuse_sample_window", "has_dir"]) and not bool(cp.set_index("arm").loc["r85_dc_refuse_sample_window", "has_summary"]), "refuse arm: dir present, no summary, listed in completeness.csv")
    ck(all(os.path.exists(os.path.join(out, f)) for f in ("completeness.csv", "summaries_all.json", "compress_manifests.json", "sensor_stamps_summaries.json")), "the usual collector outputs")
    print("SELFTEST r85_collect", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (n[0], len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
