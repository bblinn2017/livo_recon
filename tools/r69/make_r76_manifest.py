#!/usr/bin/env python3
"""R76: build the job files, arm_table.tsv, listener specs, required_outputs.json, switch_pairs.json and the size ledger from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r76_manifest.py --jobs r69_inputs/r66_jobs.json --out r76_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (only the counts / key checks in tools/r69/r76_gates.py --selftest and the check at the end of this file).
Reuses the R74 helpers (make_r74_manifest as M): the same R66 base jobs, the same overrides.

Bryce 2026-10-02: line configuration = a flag plus mutually exclusive choices, default hard absolute (F-445); "test all of our line modes on the entirety of the EEE01 bag";
grid = voxel {0.425, 1.0} x Q {fixed, motion} x pfn {1, 3}; offline voxel-size statistics on all nine bags (no bag is run with an estimated value).
Modes (6): off (line residual disabled, plane rows only), hardabs, hardrel, hardrelnoise, hardshape, smooth. Keys (R76): voxel_map.plane.line_residual.{enabled, form, rule, rel_sigmas, shape_ratio}.
Batches (queues):
  r76_id   7 identity arms (EF cell n1600 ds off pfn3 fq, 16 s): the new keys reproduce the R74 / R75 arm of the same mode (state is compared by gates, not assumed), plus ONE determinism repeat.
  r76_ref  7 refuse arms (5 s each, MUST FAIL at startup naming the key).
  r76_fb   48 full-bag arms (eee_01, whole bag) + ONE full-bag determinism repeat = 49.
"""
import argparse, copy, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r74_manifest as M  # noqa: E402

P = "voxel_map.plane.line_residual."
LINE76 = {"off": {},
          "hardabs": {P + "enabled": True},                                                   # form and rule at their defaults (hard, absolute): the default Bryce chose
          "hardrel": {P + "enabled": True, P + "form": "hard", P + "rule": "relative"},
          "hardrelnoise": {P + "enabled": True, P + "form": "hard", P + "rule": "relative_noise"},
          "hardshape": {P + "enabled": True, P + "form": "hard", P + "rule": "shape"},
          "smooth": {P + "enabled": True, P + "form": "smooth"}}
# what the parameter dump line must say for each mode: (enabled, form, rule)
WANT = {"off": ("false", "hard", "absolute"), "hardabs": ("true", "hard", "absolute"), "hardrel": ("true", "hard", "relative"), "hardrelnoise": ("true", "hard", "relative_noise"),
        "hardshape": ("true", "hard", "shape"), "smooth": ("true", "smooth", "absolute")}
ENUM = {"off": 0, "hardabs": 1, "hardrel": 2, "smooth": 3, "hardrelnoise": 4, "hardshape": 5}
DESC = {"off": "line residual off (plane rows only)", "hardabs": "hard absolute, e1 < 2.5e-3 (the default)", "hardrel": "hard relative (e1 <= (1+3 sqrt(2/(N-1))) e0)",
        "hardrelnoise": "hard relative with the noise floor 4e-4", "hardshape": "hard shape (e1 <= 0.1 e2)", "smooth": "smooth (three rows per matched point)"}
MODES = ("off", "hardabs", "hardrel", "hardrelnoise", "hardshape", "smooth")
VOXELS = (0.425, 1.0)
QS = ("fq", "mq")
PFNS = (1, 3)
CELL = {"calib.stationary.num_samples": 1600, "imu.ds.mode": "off", "cbk.lidar.point_filter_num": 3, "imu.ds.ds_leaf_size": None}
ID_DUR = int(math.ceil(1600 / 197.5 + 8.0))
# identity arm -> (mode, voxel, the R74 / R75 arm of the same mode at the same EF cell whose data are the reference)
IDENT = [("r76_id_hardabs_v0p425", "hardabs", 0.425, "r74_line_hardabs_v0p425_n1600_dsoff_pfn3_fq"),
         ("r76_id_hardabs_v1p0", "hardabs", 1.0, "r74_line_hardabs_v1p0_n1600_dsoff_pfn3_fq"),
         ("r76_id_smooth_v0p425", "smooth", 0.425, "r74_line_smooth_v0p425_n1600_dsoff_pfn3_fq"),
         ("r76_id_off_v1p0", "off", 1.0, "r74_vox_v1p0_ml2_n1600_dsoff_pfn3_fq"),
         ("r76_id_hardrelnoise_v1p0", "hardrelnoise", 1.0, "r75_line_hardrelnoise_v1p0_n1600_dsoff_pfn3_fq"),
         ("r76_id_hardshape_v1p0", "hardshape", 1.0, "r75_line_hardshape_v1p0_n1600_dsoff_pfn3_fq")]
ID_DET, ID_DET_ORIG = "r76_own_det_id_hardabs_v1p0", "r76_id_hardabs_v1p0"
FB_DET_ORIG = "r76_fb_hardabs_v1p0_pfn3_fq"
FB_DET = "r76_own_det_fb_hardabs_v1p0_pfn3_fq"
# refuse arms: (id suffix, overrides, needles that must all appear in the startup failure text)
REFUSE = [("old_mode", {P + "mode": "hard"}, ["line_residual/mode", "RETIRED"]),
          ("old_threshold", {P + "threshold": "absolute"}, ["line_residual/threshold", "RETIRED"]),
          ("rule_while_off", {P + "enabled": False, P + "rule": "shape"}, ["line_residual/rule", "is dead"]),
          ("rule_with_smooth", {P + "enabled": True, P + "form": "smooth", P + "rule": "absolute"}, ["line_residual/rule", "is dead"]),
          ("shape_ratio_with_absolute", {P + "enabled": True, P + "rule": "absolute", P + "shape_ratio": 0.2}, ["line_residual/shape_ratio", "is dead"]),
          ("rule_bogus", {P + "enabled": True, P + "rule": "bogus"}, ["line_residual/rule = 'bogus' is not one of"]),
          ("form_bogus", {P + "enabled": True, P + "form": "bogus"}, ["line_residual/form = 'bogus' is not one of"])]
ASKED_CFG = ("Bryce (2026-10-02): 'we just use the hard cutoff, the one that measures E1 against the exact same threshold that E0 would be measured against ... restructure the "
             "configuration to have a flag that turns line mode on or off ... no two mutually exclusive options can be on at the same time ... default should just be the basic hard absolute threshold'")
ASKED_FB = ("Bryce (2026-10-02): 'test all of our line modes on the entirety of the EEE01 bag'; grid chosen by him: voxel {0.425, 1.0} x Q {fixed, motion} x point filter {1, 3}; "
            "'Voxel x Q mode x pfn'")
EXPECTED = {"residual_dump_all.csv.gz": 450000, "r73_f417_hth_frames.csv": 2000, "pos_trace_all.csv.gz": 5000}
MAXFILE = {"residual_dump_all.csv.gz": 2800000, "r73_f417_hth_frames.csv": 1000000, "pos_trace_all.csv.gz": 200000}
FB_POS_EXPECTED = 110000   # R74 full-bag median 104 KB (make_r74_manifest EXPECTED); the bag is the same eee_01
DUMP_COLS = M.DUMP_COLS + ["row_kind"]


def arms():
    A = []
    for aid, mode, v, ref in IDENT:
        A.append(dict(id=aid, base=M.base_of("eee_01", "fq"), group="IDN", tier="EF", batch="id", dur=ID_DUR, mode=mode, voxel=v, kind="id", ref=ref,
                      delta=dict(M.VOX(v, 2), **LINE76[mode], **M.F1DUMP, **CELL), f417=1,
                      asked="OWN (planning agent's validation of the config restructure, chosen with Bryce: 'ctest plus a small identity batch'): %s, root voxel %s, EF cell n1600 ds off pfn3 fq; the state "
                            "must reproduce %s. %s" % (DESC[mode], v, ref, ASKED_CFG),
                      purpose="R76 identity: mode %s at voxel %s under the new keys vs %s" % (mode, v, ref)))
    A.append(dict(id=ID_DET, base=M.base_of("eee_01", "fq"), group="DET", tier="EF", batch="id", dur=ID_DUR, mode="hardabs", voxel=1.0, kind="det_id", ref=ID_DET_ORIG,
                  delta=dict(M.VOX(1.0, 2), **LINE76["hardabs"], **M.F1DUMP, **CELL), f417=1,
                  asked="OWN (rule 3.7, determinism control in every batch): an exact repeat of %s; its state-trace digest must equal that arm's. Needs Bryce's sight: not asked for in his words" % ID_DET_ORIG,
                  purpose="OWN determinism control: repeat of %s" % ID_DET_ORIG))
    for suf, delta, needles in REFUSE:
        A.append(dict(id="r76_refuse_%s" % suf, base="00_id_baseline_60s", group="REF", tier="REF", batch="ref", dur=5, mode=None, voxel=None, kind="refuse", ref=None, needles=needles,
                      delta=dict(delta), f417=0,
                      asked="OWN (the restructure is Bryce's; testing that it refuses a contradictory or retired setting is the planning agent's validation, 'no two mutually exclusive options can be on at the same time'): "
                            "MUST FAIL at startup, text contains %s. Needs Bryce's sight for arms old_threshold (a seventh refusal beyond the six listed in the handoff plan)" % " and ".join(repr(n) for n in needles),
                      purpose="MUST FAIL: %s" % suf))
    for mode in MODES:
        for v in VOXELS:
            for q in QS:
                for pfn in PFNS:
                    d = dict(M.VOX(v, 2), **LINE76[mode], **{"cbk.lidar.point_filter_num": pfn})
                    if mode != "off":
                        d.update(M.F1DUMP)   # frame 1 dump for engagement evidence; the off arm has no line rows to show
                    A.append(dict(id="r76_fb_%s_v%s_pfn%d_%s" % (mode, M.tag(v), pfn, q), base=M.base_of("eee_01", q), group="FB", tier="FB", batch="fb", dur=None, mode=mode, voxel=v, kind="fb", ref=None,
                                  delta=d, f417=0 if mode == "off" else 1,
                                  asked="%s: eee_01 whole bag, %s, root voxel %s max_layer 2, point filter %d, %s Q" % (ASKED_FB, DESC[mode], v, pfn, "fixed" if q == "fq" else "motion"),
                                  purpose="R76 full bag: %s, voxel %s / max_layer 2, pfn %d, %s" % (mode, v, pfn, q)))
    d = dict(M.VOX(1.0, 2), **LINE76["hardabs"], **{"cbk.lidar.point_filter_num": 3}, **M.F1DUMP)
    A.append(dict(id=FB_DET, base=M.base_of("eee_01", "fq"), group="DET", tier="FB", batch="fb", dur=None, mode="hardabs", voxel=1.0, kind="det_fb", ref=FB_DET_ORIG, delta=d, f417=1,
                  asked="OWN (rule 3.7, determinism control in every batch): an exact whole-bag repeat of %s; its state-trace digest must equal that arm's. Needs Bryce's sight: not asked for in his words" % FB_DET_ORIG,
                  purpose="OWN determinism control: whole-bag repeat of %s" % FB_DET_ORIG))
    return A


def required_outputs(table):
    sel = lambda f: [t["arm"] for t in table if f(t)]
    dumpers = sel(lambda t: t["f417"] == "1")
    ident = sel(lambda t: t["group"] in ("IDN",) or t["arm"] == ID_DET)
    fb_all = sel(lambda t: t["batch"] == "r76_fb")
    R = []

    def add(fu, title, arms_, file, cols, min_rows, where="return", source="arm", expected=None, maxb=None):
        eb = (expected if expected is not None else EXPECTED.get(file, 50000)) * max(1, len(arms_)) if where == "return" else 0
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=eb,
                      max_bytes_per_file=maxb or MAXFILE.get(file, 400000)))
    add("F-447", "per-residual dump of frame 1 with row_kind (0 plane, 1-2 hard line u0 u1, 3-5 smooth) for every arm that enables a line mode", dumpers, "residual_dump_all.csv.gz", DUMP_COLS, 100)
    add("F-447", "residual report of frame 1 (HtH / Htz metrics) for every dumping arm", dumpers, "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 1)
    add("F-447", "per-scan posterior position and yaw of every identity arm (scan_id key)", ident, "pos_trace_all.csv.gz", M.POS_COLS, 20)
    add("F-447", "per-scan posterior position and yaw of every full-bag arm (excursion and yaw movement of scans 1-10 are computed from it)", fb_all, "pos_trace_all.csv.gz", M.POS_COLS, 1000,
        expected=FB_POS_EXPECTED, maxb=200000)
    add("F-445", "gates: identity vs R74/R75 (facts), determinism digests, refusals, effective line config, row kinds, engagement", [], "r76_gates.txt", [], 1, source="generated")
    add("F-445", "identity comparison of every identity arm against its R74/R75 counterpart: per-scan max abs difference of position and yaw, dump signature equality", [], "r76_identity_vs_old.csv",
        ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "digest_equal_if_known"], 6, source="generated")
    add("F-445", "determinism controls and their originals", [], "identity_controls.txt", [], 1, source="generated")
    add("F-435", "F-426 part 1 per-frame shadow-vs-driver step difference (existing R73 shadow files, r76_step0_f426.py, no new run)", [], "r74_f426_part1_frames.csv",
        ["pair", "frame", "step_diff_mm", "at_or_after_onset"], 200, source="generated")
    add("F-435", "F-426 part 1 summary", [], "r74_f426_part1_summary.csv", ["pair", "onset", "max_before_mm", "max_from_onset_mm"], 4, source="generated")
    add("F-446", "F-426 part 2 per-frame alignment-free comparison keyed on scan_id (R74 switch arms against the R73 controls; t_rel_s is NaN for the controls)", [], "r74_f426_part2_frames.csv",
        ["pair", "frame", "dA_mm", "dB_mm", "dAB_mm"], 200, source="generated")
    add("F-446", "F-426 part 2 summary", [], "r74_f426_part2_summary.csv", ["pair", "switch_arm", "onset", "dA_before_max_mm", "dB_from_onset_max_mm"], 4, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated", [], "engagement_all.txt", [], 50, source="generated")
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 50, source="generated")
    add("F-407", "engagement_imu.txt present for EVERY dispatched arm that is not a refuse arm", sel(lambda t: t["group"] != "REF"), "engagement_imu.txt", [], 1, where="archive")
    VS = "voxel_stats/"
    add("F-448", "voxel-size estimates of all nine bags (r76_voxel_stats.py --collect)", [], VS + "voxel_estimates_all.csv",
        ["seq", "pfn", "scan", "estimator", "param", "value_edge_m", "value_max_layer", "plane_leaf_fraction", "note", "n_scans"], 9 * 2 * 3, source="generated")
    return R


def ledger(RO):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"] or e["source"] == "generated":   # generated files have their own ceilings below
            continue
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * 3 if e["file"] != "residual_dump_all.csv.gz" else e["expected_bytes"] * 2),
                    "basis": "measured R74/R75 return (median per file, expected_bytes); per-file ceiling = R74 maximum + 15 %%; %d arms" % len(e["arms"]), "note": e["followup"]})
    for pat, mx, tot in (("r73_f417_classes.csv", 20000, 4000000), ("r73_f417_consistency.csv", 5000, 1000000), ("r76_gates.txt", 400000, 400000), ("identity_controls.txt", 150000, 150000),
                         ("r76_identity_vs_old.csv", 20000, 20000), ("engagement_all.txt", 400000, 400000), ("completeness.csv", 200000, 200000), ("r74_f426_part*", 200000, 800000),
                         ("step0_f426_log.txt", 20000, 20000), ("phase_summary.csv", 3000000, 3000000), ("arm_scan_all.csv.gz", 30000000, 30000000), ("raw_sha256.txt", 20000, 20000),
                         ("R76_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000), ("voxel_stats_*.csv", 400000, 8000000), ("voxel_estimates_all.csv", 600000, 600000),
                         ("voxel_stats_log.txt", 20000, 200000), ("voxel_stats_run_log.txt", 100000, 100000), ("listener_*.log", 5000000, 20000000), ("*.md", 1000000, 3000000), ("*.json", 2000000, 10000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000),
                         ("*.tsv", 1000000, 3000000), ("*.csv", 1000000, 5000000), ("*.sha256", 100000, 100000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": tot, "basis": "generic ceiling (R74/R75 ledger values or a round figure); not a measurement", "note": "non-arm file"})
    # the same file name may be required by two groups (pos_trace_all.csv.gz: 7 identity arms and 49 full-bag arms): the first R75 ledger kept only the first entry, so the second group went OVER-TOTAL.
    # Merge: per-file ceiling = the larger, total = the sum.
    res, at = [], {}
    for x in out:
        if x["pattern"] in at:
            y = res[at[x["pattern"]]]
            y["max_bytes_per_file"] = max(y["max_bytes_per_file"], x["max_bytes_per_file"]); y["max_total_bytes"] += x["max_total_bytes"]
            y["note"] += " + " + x["note"]
        else:
            at[x["pattern"]] = len(res); res.append(dict(x))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r76_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pairs-from", default=None, help="switch_pairs.json of R75 to copy unchanged (F-426 pairs)")
    a = ap.parse_args()
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    os.makedirs(a.out, exist_ok=True)
    out = {"id": [], "ref": [], "fb": []}; table = []; seen = set()
    for r in arms():
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        assert r["id"] not in seen, r["id"]; seen.add(r["id"])
        j = copy.deepcopy(J[r["base"]]); j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            o[k] = v   # None for imu.ds.ds_leaf_size: gen_jobs.py's null-deletion strips the key the base config sets (as R72 to R75)
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r76_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch="r76_" + r["batch"], sampling=sampling, requires=" ".join(M.requires_of(o, "refuse" if r["batch"] == "ref" else "x")),
                          f417=str(r["f417"]), plane_check="0", asked=r["asked"]))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r76_%s" % b, "results_root": "%s/r76_%s" % (a.results_root, b), "expected_jobs": n}
    for nm, rn, bs in (("id", "R76-id", ["id", "ref"]), ("fb", "R76-fb", ["fb"])):
        sp = M.listener_spec(rn, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r76_manifest") for c in blk["cmd"]]
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(ledger(RO), open(os.path.join(a.out, "r76_file_size_ledger.json"), "w"), indent=1)
    if a.pairs_from:
        import shutil
        shutil.copy(a.pairs_from, os.path.join(a.out, "switch_pairs.json"))
    print("id %d arms (6 identity + 1 determinism), ref %d refuse arms, fb %d arms (48 + 1 determinism)" % (len(out["id"]), len(out["ref"]), len(out["fb"])))


if __name__ == "__main__":
    main()
