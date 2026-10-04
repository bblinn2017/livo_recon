#!/usr/bin/env python3
"""R77: build the job files, arm_table.tsv, listener specs, required_outputs.json, switch_pairs.json and the size ledger from the R66 job list (r66_jobs.json).

    python3 tools/r69/make_r77_manifest.py --jobs r69_inputs/r66_jobs.json --out r77_manifest --results-root /root/catkin_ws/livo_recon_results

Planning agent's script; NOT run against the real host by me (only the counts / key checks in tools/r69/r77_gates.py --selftest and the check at the end of this file).
Reuses the R74 helpers (make_r74_manifest as M): the same R66 base jobs, the same overrides.

Bryce 2026-10-03: line and plane variances equivalent in kind (F-453: new key voxel_map.plane.line_residual.variance = fit_only | plane_equivalent, old kept), the line-mode ablation redone
with it; F-433 as a flag (voxel_map.plane.plane_gap_floor_en, floor at the sensor noise variance, plane kept); F-449 voxel-edge sweep 0.5, 0.6, 0.7, 0.8 m; F-451 line-row logging; F-452 gate fixes.
Batches (queues):
  r77_id   10 arms (EF cell n1600 ds off pfn3 fq, 16 s): 6 identity arms (the new build with every new key at its default reproduces the R76 arm of the same mode), ONE determinism repeat,
           3 smoke arms (one per new flag: plane_equivalent on hard absolute and on smooth, the gap floor with lines off).
  r77_ref  5 refuse arms (5 s each, MUST FAIL at startup naming the key).
  r77_fb   73 full-bag arms (eee_01, whole bag): 40 plane_equivalent line arms (5 modes x 8 cells), 8 lines-off controls (the 8 cells), 8 gap-floor arms (lines off, the 8 cells),
           16 voxel-edge sweep arms (edges 0.5 0.6 0.7 0.8 x lines off / hard absolute plane_equivalent x fixed / motion Q, every point), ONE determinism repeat.
The 8 cells are voxel {0.425, 1.0} x Q {fixed, motion} x point filter {1 (every point), 3 (every third point)}, max_layer 2.
"""
import argparse, copy, csv, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r74_manifest as M  # noqa: E402

P = "voxel_map.plane.line_residual."
VAR = P + "variance"
GAP = "voxel_map.plane.plane_gap_floor_en"
LOGKEY = {"lio.line_rows_log_en": True}   # F-451: line_rows_per_scan.csv (write-only)
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
LINE_MODES = MODES[1:]
VOXELS = (0.425, 1.0)
QS = ("fq", "mq")
PFNS = (1, 3)
SWEEP_EDGES = (0.5, 0.6, 0.7, 0.8)
CELL = {"calib.stationary.num_samples": 1600, "imu.ds.mode": "off", "cbk.lidar.point_filter_num": 3, "imu.ds.ds_leaf_size": None}
ID_DUR = int(math.ceil(1600 / 197.5 + 8.0))
# identity arm -> (mode, voxel, the R76 identity arm of the same name whose data are the reference). Every new key is left at its default in these arms.
IDENT = [("r77_id_hardabs_v0p425", "hardabs", 0.425, "r76_id_hardabs_v0p425"),
         ("r77_id_hardabs_v1p0", "hardabs", 1.0, "r76_id_hardabs_v1p0"),
         ("r77_id_smooth_v0p425", "smooth", 0.425, "r76_id_smooth_v0p425"),
         ("r77_id_off_v1p0", "off", 1.0, "r76_id_off_v1p0"),
         ("r77_id_hardrelnoise_v1p0", "hardrelnoise", 1.0, "r76_id_hardrelnoise_v1p0"),
         ("r77_id_hardshape_v1p0", "hardshape", 1.0, "r76_id_hardshape_v1p0")]
ID_DET, ID_DET_ORIG = "r77_own_det_id_hardabs_v1p0", "r77_id_hardabs_v1p0"
# smoke arms (rule 3.6, one per new flag): id, mode, voxel, which flag
SMOKE = [("r77_sm_pe_hardabs_v0p425", "hardabs", 0.425, "pe"),
         ("r77_sm_pe_smooth_v0p425", "smooth", 0.425, "pe"),
         ("r77_sm_gapfloor_off_v0p425", "off", 0.425, "gap")]
FB_DET_ORIG = "r77_fb_pe_hardabs_v1p0_pfn3_fq"
FB_DET = "r77_own_det_fb_pe_hardabs_v1p0_pfn3_fq"
# refuse arms: (id suffix, overrides, needles that must all appear in the startup failure text)
REFUSE = [("variance_while_off", {P + "enabled": False, VAR: "plane_equivalent"}, ["line_residual/variance", "is dead"]),
          ("variance_bogus", {P + "enabled": True, VAR: "bogus"}, ["line_residual/variance = 'bogus' is not one of"]),
          ("gapfloor_with_rejection", {GAP: True, "voxel_map.plane.plane_var_denom_floor_en": True}, ["plane_gap_floor_en = true together with plane_var_denom_floor_en"]),
          ("gapfloor_information", {GAP: True, "voxel_map.plane.plane_var_mode": "information"}, ["plane_gap_floor_en", "is dead"]),
          ("variance_information", {P + "enabled": True, VAR: "plane_equivalent", "voxel_map.plane.plane_var_mode": "information"}, ["plane_equivalent needs"])]
ASK_VAR = ("Bryce (2026-10-03): 'i would like plane and line variances to be equivalent in kind, let's change the code to reflect that. we'll need to redo the experiments ablating the different line mode levers'; "
           "his answers: 'New config choice, old kept (Recommended)', 'Floor at sensor noise (Recommended)', 'Yes, all line forms (Recommended)', 'Full line matrix plus F-449 sweep (Recommended)', "
           "'Fixed-Q arms' ticked")
ASK_433 = ("Bryce (2026-10-03): 'we can also try f433 as a flag, i'd like to make sure that doesn't change behavior negatively'; his answers: 'New gap floor, plane kept (Recommended)', "
           "'All eight matrix cells (Recommended)'; 'we can add f453 and f449'")
ASK_449 = ("Bryce (2026-10-03): 'we can sweep f449 voxel edges as suggested above' (suggested: 0.5, 0.6, 0.7, 0.8 m, every point, lines off and hard absolute); his answers: "
           "'plane_equivalent (Recommended)', 'Both Q (Recommended)'; 'we can add f453 and f449'")
ASK_451 = "Bryce (2026-10-03): ticked 'F-451 line-row logging (Recommended)' (per-scan counts of rows by kind, to read the effect of the new variance on line-row engagement)"
STANDING = "STANDING RULE %s (Bryce's own rule): "
EXPECTED = {"residual_dump_all.csv.gz": 25000, "r73_f417_hth_frames.csv": 2000, "pos_trace_all.csv.gz": 140000, "line_rows_per_scan_all.csv.gz": 70000}
MAXFILE = {"residual_dump_all.csv.gz": 120000, "r73_f417_hth_frames.csv": 100000, "pos_trace_all.csv.gz": 220000, "line_rows_per_scan_all.csv.gz": 250000}
DUMP_COLS = M.DUMP_COLS + ["row_kind"]
LINE_ROWS_COLS = ["scan_id", "iter", "n_rows", "n_points", "n_plane_rows", "n_line_u0", "n_line_u1", "n_smooth_u0", "n_smooth_u1", "n_smooth_u2", "n_pts_with_line_rows", "sum_fit_var_line", "sum_tilt_var",
                  "sum_sigma2_line", "sum_sigma2_plane", "gap_fits", "gap_d1", "gap_d2", "tilt_voxels", "tilt_floored_axes", "tilt_fallbacks"]


FB_POS_EXPECTED = 110000   # R74 full-bag median 104 KB (the bag is the same eee_01)
OWN = ("OWN (planning agent's validation under the standing rules; NOT asked for in Bryce's words, needs his sight): ")


def req_of(o, batch):
    r = M.requires_of(o, batch)
    if o.get("lio.line_rows_log_en"):
        r = r + ["line_rows_per_scan.csv"]
    return r


def cellarm(aid, mode, v, q, pfn, extra, group, asked, purpose, dump):
    d = dict(M.VOX(v, 2), **LINE76[mode], **extra, **LOGKEY, **{"cbk.lidar.point_filter_num": pfn})
    if dump:
        d.update(M.F1DUMP)
    return dict(id=aid, base=M.base_of("eee_01", q), group=group, tier="FB", batch="fb", dur=None, mode=mode, voxel=v, kind="fb", ref=None, delta=d, f417=1 if dump else 0, asked=asked, purpose=purpose)


def arms():
    A = []
    for aid, mode, v, ref in IDENT:
        A.append(dict(id=aid, base=M.base_of("eee_01", "fq"), group="IDN", tier="EF", batch="id", dur=ID_DUR, mode=mode, voxel=v, kind="id", ref=ref,
                      delta=dict(M.VOX(v, 2), **LINE76[mode], **LOGKEY, **M.F1DUMP, **CELL), f417=1,
                      asked=OWN + "%s, root voxel %s, EF cell n1600 ds off pfn3 fq; every new key at its default (line_rows_log_en is write-only); the state must reproduce %s exactly" % (DESC[mode], v, ref),
                      purpose="R77 identity: mode %s at voxel %s with the new keys at default vs %s" % (mode, v, ref)))
    A.append(dict(id=ID_DET, base=M.base_of("eee_01", "fq"), group="DET", tier="EF", batch="id", dur=ID_DUR, mode="hardabs", voxel=1.0, kind="det_id", ref=ID_DET_ORIG,
                  delta=dict(M.VOX(1.0, 2), **LINE76["hardabs"], **LOGKEY, **M.F1DUMP, **CELL), f417=1,
                  asked=OWN + "rule 3.7 determinism control: an exact repeat of %s; its state-trace digest must equal that arm's" % ID_DET_ORIG,
                  purpose="OWN determinism control: repeat of %s" % ID_DET_ORIG))
    for aid, mode, v, what in SMOKE:
        ex = {GAP: True} if what == "gap" else {VAR: "plane_equivalent"}
        A.append(dict(id=aid, base=M.base_of("eee_01", "fq"), group="SMK", tier="EF", batch="id", dur=ID_DUR, mode=mode, voxel=v, kind="smoke", ref=None,
                      delta=dict(M.VOX(v, 2), **LINE76[mode], **ex, **LOGKEY, **M.F1DUMP, **CELL), f417=1,
                      asked=OWN + "rule 3.6 smoke arm for the new %s, EF cell, %s, root voxel %s; the counters of line_rows_per_scan.csv must move (engagement gate)" % (
                          "gap floor (plane_gap_floor_en, lines off)" if what == "gap" else "line variance plane_equivalent", DESC[mode], v),
                      purpose="R77 smoke: %s, %s, voxel %s" % ("gap floor" if what == "gap" else "plane_equivalent", mode, v)))
    for suf, delta, needles in REFUSE:
        A.append(dict(id="r77_refuse_%s" % suf, base="00_id_baseline_60s", group="REF", tier="REF", batch="ref", dur=5, mode=None, voxel=None, kind="refuse", ref=None, needles=needles,
                      delta=dict(delta), f417=0,
                      asked=OWN + "rule 3.4 refusal: the contradictory or dead setting MUST FAIL at startup, text contains %s" % " and ".join(repr(n) for n in needles),
                      purpose="MUST FAIL: %s" % suf))
    # F-453: the line matrix redone with plane_equivalent (5 modes x 8 cells), plus the 8 lines-off controls
    for mode in MODES:
        for v in VOXELS:
            for q in QS:
                for pfn in PFNS:
                    pe = mode != "off"
                    aid = "r77_fb_%s_%s_v%s_pfn%d_%s" % ("pe" if pe else "off", mode if pe else "ctl", M.tag(v), pfn, q)
                    aid = aid.replace("_off_ctl_", "_off_")
                    ex = {VAR: "plane_equivalent"} if pe else {}
                    A.append(cellarm(aid, mode, v, q, pfn, ex, "FB",
                                     "%s: the F-447 line matrix redone with the line variance equivalent in kind to the plane variance; eee_01 whole bag, %s%s, root voxel %s max_layer 2, point filter %d, %s Q" % (
                                         ASK_VAR, DESC[mode], ", variance plane_equivalent" if pe else " (control, no line rows)", v, pfn, "fixed" if q == "fq" else "motion"),
                                     "R77 full bag: %s%s, voxel %s / max_layer 2, pfn %d, %s" % (mode, " plane_equivalent" if pe else " control", v, pfn, q), pe))
    # F-433: the gap floor, lines off, the 8 cells
    for v in VOXELS:
        for q in QS:
            for pfn in PFNS:
                A.append(cellarm("r77_fb_gap_v%s_pfn%d_%s" % (M.tag(v), pfn, q), "off", v, q, pfn, {GAP: True}, "GAP",
                                 "%s: gap floor (plane_gap_floor_en = true, lines off) on each matrix cell, to check it does not change behaviour negatively; compare with the lines-off control of the same cell; eee_01 whole bag, "
                                 "root voxel %s max_layer 2, point filter %d, %s Q" % (ASK_433, v, pfn, "fixed" if q == "fq" else "motion"),
                                 "R77 full bag: gap floor, lines off, voxel %s, pfn %d, %s" % (v, pfn, q), True))
    # F-449: voxel-edge sweep, every point
    for e in SWEEP_EDGES:
        for mode in ("off", "hardabs"):
            for q in QS:
                pe = mode != "off"
                A.append(cellarm("r77_fb_sw_%s_v%s_pfn1_%s" % ("pe_hardabs" if pe else "off", M.tag(e), q), mode, e, q, 1, {VAR: "plane_equivalent"} if pe else {}, "SWP",
                                 "%s: voxel-edge sweep at %s m, every point, %s, %s Q, max_layer 2" % (ASK_449, e, "hard absolute with plane_equivalent" if pe else "lines off", "fixed" if q == "fq" else "motion"),
                                 "R77 sweep: edge %s, %s, pfn 1, %s" % (e, "hardabs plane_equivalent" if pe else "lines off", q), False))
    d = cellarm(FB_DET, "hardabs", 1.0, "fq", 3, {VAR: "plane_equivalent"}, "DET",
                OWN + "rule 3.7 determinism control: an exact whole-bag repeat of %s; its state-trace digest must equal that arm's" % FB_DET_ORIG,
                "OWN determinism control: whole-bag repeat of %s" % FB_DET_ORIG, True)
    d["kind"] = "det_fb"; d["ref"] = FB_DET_ORIG
    A.append(d)
    return A


def required_outputs(table):
    sel = lambda f: [t["arm"] for t in table if f(t)]
    dumpers = sel(lambda t: t["f417"] == "1")
    ident = sel(lambda t: t["group"] in ("IDN", "SMK") or t["arm"] == ID_DET)
    fb_all = sel(lambda t: t["batch"] == "r77_fb")
    idfb = sel(lambda t: t["group"] != "REF")
    R = []

    def add(fu, title, arms_, file, cols, min_rows, where="return", source="arm", expected=None, maxb=None):
        eb = (expected if expected is not None else EXPECTED.get(file, 50000)) * max(1, len(arms_)) if where == "return" else 0
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=eb,
                      max_bytes_per_file=maxb or MAXFILE.get(file, 400000)))
    add("F-453", "per-residual dump of frame 1 with row_kind (0 plane, 1-2 hard line u0 u1, 3-5 smooth) for every arm that dumps", dumpers, "residual_dump_all.csv.gz", DUMP_COLS, 50)
    add("F-453", "residual report of frame 1 (HtH / Htz metrics) for every dumping arm", dumpers, "r73_f417_hth_frames.csv", ["frame", "iter", "hth_trace", "eig_min", "cond", "info_yaw"], 1)
    add("F-451", "per-scan line-row log (rows by kind, summed variances, gap and tilt counters), identity and smoke arms", ident, "line_rows_per_scan_all.csv.gz", LINE_ROWS_COLS, 50)
    add("F-451", "per-scan line-row log (thinned to scans <= 800 plus every 10th), every full-bag arm", fb_all, "line_rows_per_scan_all.csv.gz", LINE_ROWS_COLS, 100)
    add("F-453", "per-scan posterior position and yaw of every identity and smoke arm (scan_id key)", ident, "pos_trace_all.csv.gz", M.POS_COLS, 20)
    add("F-453", "per-scan posterior position and yaw of every full-bag arm", fb_all, "pos_trace_all.csv.gz", M.POS_COLS, 1000, expected=FB_POS_EXPECTED, maxb=200000)
    add("F-452", "gates: identity vs R76 (state facts, node_id as a separate INFO row), determinism digests, refusals, effective line config, row kinds, engagement of the new counters", [], "r77_gates.txt", [], 1, source="generated")
    add("F-452", "identity comparison of every identity arm against its R76 counterpart: per-scan max abs difference of position and yaw, dump signature equality, node_id equality (INFO)", [], "r77_identity_vs_old.csv",
        ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "node_id_equal", "digest_equal_if_known"], 6, source="generated")
    add("F-452", "determinism controls and their originals", [], "identity_controls.txt", [], 1, source="generated")
    add("F-409", "engagement of every dispatched arm concatenated", [], "engagement_all.txt", [], 50, source="generated")
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 50, source="generated")
    add("F-407", "engagement_imu.txt present for EVERY dispatched arm that is not a refuse arm", idfb, "engagement_imu.txt", [], 1, where="archive")
    return R


def ledger(RO):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"] or e["source"] == "generated":
            continue
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * 3 if e["file"] != "residual_dump_all.csv.gz" else e["expected_bytes"] * 2),
                    "basis": "estimate from the measured R74-R76 returns (median per file); the line_rows file has no measurement yet, the agent measures one smoke and one full arm (size_ledger.py measure); %d arms" % len(e["arms"]), "note": e["followup"]})
    for pat, mx, tot in (("r73_f417_classes.csv", 20000, 4000000), ("r73_f417_consistency.csv", 5000, 1000000), ("r77_gates.txt", 400000, 400000), ("identity_controls.txt", 150000, 150000),
                         ("r77_identity_vs_old.csv", 20000, 20000), ("engagement_all.txt", 400000, 400000), ("completeness.csv", 200000, 200000),
                         ("phase_summary.csv", 3000000, 3000000), ("arm_scan_all.csv.gz", 30000000, 30000000), ("raw_sha256.txt", 20000, 20000),
                         ("R77_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000), ("listener_*.log", 5000000, 20000000), ("*.md", 1000000, 3000000),
                         ("*.json", 2000000, 10000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000), ("*.tsv", 1000000, 3000000), ("*.csv", 1000000, 5000000), ("*.sha256", 100000, 100000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": tot, "basis": "generic ceiling (R74-R76 ledger values or a round figure); not a measurement", "note": "non-arm file"})
    # one file name required by two groups (pos_trace_all.csv.gz, line_rows_per_scan_all.csv.gz): per-file ceiling = the larger, total = the sum
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
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", default="r77_manifest")
    ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
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
            o[k] = v   # None for imu.ds.ds_leaf_size: gen_jobs.py's null-deletion strips the key the base config sets (as R72 to R76)
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/r77_%s/%s" % (a.results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out[r["batch"]].append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch="r77_" + r["batch"], sampling=sampling, requires=" ".join(req_of(o, "refuse" if r["batch"] == "ref" else "x")),
                          f417=str(r["f417"]), plane_check="0", asked=r["asked"]))
    for b, lst in out.items():
        json.dump(lst, open(os.path.join(a.out, "jobs_%s.json" % b), "w"), indent=1)
    cols = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked"]
    with open(os.path.join(a.out, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader(); w.writerows(table)
    q = lambda b, n: {"dir": "ablations/_job_queue/r77_%s" % b, "results_root": "%s/r77_%s" % (a.results_root, b), "expected_jobs": n}
    for nm, rn, bs in (("id", "R77-id", ["id", "ref"]), ("fb", "R77-fb", ["fb"])):
        sp = M.listener_spec(rn, [q(b, len(out[b])) for b in bs], a.results_root, a.min_free_gb, a.workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r77_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["line_rows_per_scan.csv"]   # reduced into the archive (BIG list of r74_score_job.py) and into line_rows_per_scan_all.csv.gz
        json.dump(sp, open(os.path.join(a.out, fn), "w"), indent=1)
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(a.out, "required_outputs.json"), "w"), indent=1)
    json.dump(ledger(RO), open(os.path.join(a.out, "r77_file_size_ledger.json"), "w"), indent=1)
    print("id %d arms (6 identity + 1 determinism + 3 smoke), ref %d refuse arms, fb %d arms (40 pe + 8 off + 8 gap + 16 sweep + 1 determinism)" % (len(out["id"]), len(out["ref"]), len(out["fb"])))


if __name__ == "__main__":
    main()
