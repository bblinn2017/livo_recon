#!/usr/bin/env python3
"""R84: job files, arm_table.tsv, listener specs, queue order, required_outputs.json, size ledger and compliance table.

    python3 tools/r69/make_r84_manifest.py --jobs r69_inputs/r66_jobs.json --grid-retime r80c_grid.json --grid-bag r84_manifest/grid_bag.json \
        --onsets r84_manifest/onset_scans.json --ref-values r84_manifest/l2_reference_values.csv --out r84_manifest --results-root /root/catkin_ws/livo_recon_results
    python3 -I tools/r69/make_r84_manifest.py --selftest

Planning agent's script. NOT run on the node by the planner; the selftest runs in the planner sandbox on a synthetic job list (generator output only, not evidence).
Successor of make_r83_manifest.py. Reuses make_r81_manifest (Ctx, rt_keys, stub jobs) and through it make_r80_manifest.

Arm families (Bryce's words are in each arm's `asked` field and in COMPLIANCE_TABLE.md):
  T3   Tier 3: 9 bags x 9 edges (0.30 .. 0.70 step 0.05) x {rt, dc} x {fq, mq} x {lines off, on} x {PCA, debiased}, pfn 3, ds first, voxel_map.map.max_layer 0 (1,296)
  L2R  layer-2 reference arms for the Tier 3 cells that no existing sweep covers: same cell, max_layer 2 (846; the covered cells are listed in l2_reference_map.tsv and NOT re-run)
  CMP  F-520 comparison: lio/point_cov/sample_window_s {0, 1, 2, 5, 10, 20} ms x imu/undistort/time_based_process_noise {var_acc, none} x 18 R83 cells (9 bags x fq / mq),
       rt, pfn 3, ds first, lines off, root 0.425, max_layer 2 (216)
  OWN  identity, determinism, smoke and refuse arms (planner rule arms; the count is asserted below)
dc = overlap_policy discard (default, no extra keys); rt = retime_trim (set per arm, never through the config default).
"""
import argparse, copy, csv, json, math, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.environ.get("R84_TREE_TOOLS", ""), os.environ.get("R83_TREE_TOOLS", ""), "/tmp/claude-0/r83b_tree/tools/r69"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
import make_r81_manifest as R81  # noqa: E402

R80, R9, M = R81.R80, R81.R9, R81.M
BAGS = list(R80.BAGS)
Q_NAME = R80.Q_NAME
OWN = R80.OWN_R80
GAP = {R9.GAP: True}
DIST_KEY = dict(R80.DIST_KEY)
REASONS = {"voxel_map.plane.plane_fit_reasons_en": True, "voxel_map.plane.plane_fit_trace_frames": 10}
K_CLASSIFY = "voxel_map.plane.line_residual.classify_at_fit"
K_EQUIV = "voxel_map.plane.debiased_line_equiv"
K_FIT = "voxel_map.plane.plane_fit_mode"
K_MAXL = "voxel_map.map.max_layer"
K_VOX = "voxel_map.map.voxel_size"
K_SW = "lio.point_cov.sample_window_s"
K_TBPN = "imu.undistort.time_based_process_noise"
K_LINE = "voxel_map.plane.line_residual.enabled"
DB_KEYS = dict(R80.DB_KEYS)         # debiased, reasons, trace frames: copied from the R82 dbctl arms (make_r82_manifest.DB_KEYS is the same dict)
EDGES_T3 = [round(0.30 + 0.05 * i, 2) for i in range(9)]       # 0.30 .. 0.70
FITS = ("pca", "deb")
LNS = ("off", "on")
OVLS = ("rt", "dc")
QS = ("fq", "mq")
SW_MS = (0, 1, 2, 5, 10, 20)
TBPNS = (("vacc", "var_acc"), ("none", "none"))
POS_COLS, MSG_COLS_C, N_SCANS = R81.POS_COLS, R81.MSG_COLS_C, R81.N_SCANS
EXPECT = {"T3": 1296, "L2R": 846, "CMP": 216}
OWN_EXPECT = 23           # 4 identity + 9 determinism + 4 smoke + 6 refuse
OWN_GROUP = "OWN"
SWEEP_GROUPS = ("T3", "L2R")
ALIGNFREE_GROUPS = ("CMP",)

BRYCE_T3 = ("Bryce (2026-10-07): 'I do want to do a sweep over 0.3 to 0.7 at max layer 0' and 'I do think that we can run tier three'; Tier 3 as approved in claude/r84-designs-pending.md: "
            "9 bags x 9 edges (0.30 to 0.70, step 0.05) x {retime_trim, discard} x {fixed-Q, motion-Q} x {lines off, on} x {PCA, debiased} = 1,296 arms, pfn 3, ds first, max_layer 0, voxel_map/map/voxel_size = the edge")
BRYCE_GO = "Bryce (2026-10-07): 'We can try this comparison design that you've proposed. You may complete all preparation and generate the handoff.'"
ASK = {
    "T3": BRYCE_T3 + "; " + BRYCE_GO,
    "L2R": "PLANNER DESIGN item of claude/r84-designs-pending.md ('layer-2 reference arms for cells the existing sweeps lack'), approved with Tier 3 (" + BRYCE_GO + "): the same cell as the Tier 3 arm named in column control with voxel_map/map/max_layer 2",
    "CMP": "Bryce (2026-10-07): 'I agree with this formulation of the window covariance. Let's try it.' (F-520, key lio/point_cov/sample_window_s) and 'i only meant to remove the experimental temporal formulations. i think we should keep it but test with it on and with it off' "
           "(imu/undistort/time_based_process_noise); " + BRYCE_GO + " Design: sample window {0, 1, 2, 5, 10, 20} ms x time_based_process_noise {var_acc, none} x the 18 R83 cells (nine bags x fixed / motion Q), rt, pfn 3, ds first, lines off, root 0.425, max_layer 2",
}
ASK_MODES = "overlap mode set per arm (dc = discard, rt = retime_trim), never through the config default"


# ---------------------------------------------------------------------------------------------------------------------------------- helpers
def tg(x):
    return M.tag(x)


def queue_of(a):
    if a["fam_kind"] in ("id", "smoke", "refuse"):
        return "r84_id"
    return "r84_%s" % a["bag"]


def derive(C, src, aid, ovl, group, fam, extra, asked, purpose, pop=(), dur=None, kind="fb", ref=None, msglog=0, flag="", **flags):
    """One arm from a source arm dict (an R80 arm or an R80.mk arm): same base job and overrides, plus extra and, for rt, the retime keys."""
    b = copy.deepcopy(src)
    b["id"] = aid
    d = dict(b["delta"])
    for k in pop:
        d.pop(k, None)
    d.update(extra)
    if ovl == "rt":
        d.update(R81.rt_keys(C.grid(b["bag"])))
    b["delta"] = d
    b.update(kind=kind, ref=ref, group=group, fam=fam, ovl=ovl, src_arm=src["id"], msglog=msglog, win=0, f417=0, dump=0, early=0, flag=flag, fam_kind=kind)
    b["tier"] = "EF" if dur else "FB"
    b["dur"] = dur if dur else None
    b.update(flags)
    b["asked"] = asked + "; same arm as %s (%s, point filter %s, %s Q) with overlap mode %s%s" % (
        src["id"], "first %s s" % b["dur"] if b["dur"] else "whole bag", b["pfn"], Q_NAME.get(b["q"], b["q"]), ovl, (", extra keys " + json.dumps(extra, sort_keys=True)) if extra else "")
    b["purpose"] = purpose
    b["batch"] = queue_of(b)
    b.setdefault("control", ""); b.setdefault("r82_control", ""); b.setdefault("dump_frames", "")
    return b


def newsrc(C, bag, q, mode, v=0.425, pfn=3, extra=None, aid="tmp"):
    return R80.mk(aid, bag, mode, v, q, "SRC", "src", "src", pfn=pfn, extra=dict(extra or {}), batch="x")


def cell_src(C, bag, q, fit, ln, e):
    """The source arm of a sweep cell. pca: the R80 sweep source arm of the same bag, line mode, edge, Q (exactly the arm the R81 SWT, R82 Tier A and R83 VSW sweeps were derived from).
    deb: the R82 D4 debiased arm keys: lines off = dbctl (DB_KEYS), lines on = dbcm (hard absolute, distance rows, gap floor, classify at fit, debiased_line_equiv mean_cov)."""
    mode = "off" if ln == "off" else "hardabs"
    if fit == "pca":
        return C.src["r80_sw_%s_%s_v%s_pfn3_%s" % (bag, mode, tg(e), q)]
    if ln == "off":
        ex = dict(DB_KEYS)
    else:
        ex = dict(DIST_KEY, **GAP, **REASONS, **{K_FIT: "debiased", K_CLASSIFY: True, K_EQUIV: "mean_cov"})
    return newsrc(C, bag, q, mode, v=e, extra=ex)


def l2_ref(bag, q, ovl, fit, ln, e):
    """(arm id, source, is_new). Existing layer-2 arms that match the cell (same pfn 3, ds first, voxel_size, overlap mode, line mode, fit, Q); everything else is a new R84 arm.
    rt pca off:  R81 SWT (0.25 .. 0.60) and R82 Tier A (0.20, 0.65 .. 1.00)       dc pca off: R83 VSW (0.20 .. 0.80)       rt pca on: R81 SWT hardabs (0.25 .. 0.60)"""
    t = tg(e)
    if fit == "pca" and ovl == "rt" and ln == "off":
        if e <= 0.60 + 1e-9:
            return "r81_rt_sw_%s_off_v%s_pfn3_%s" % (bag, t, q), "R81", False
        return "r82_rt_sw_%s_off_v%s_pfn3_%s" % (bag, t, q), "R82A", False
    if fit == "pca" and ovl == "dc" and ln == "off":
        return "r83_dc_sw_%s_off_v%s_pfn3_%s" % (bag, t, q), "R83", False
    if fit == "pca" and ovl == "rt" and ln == "on" and e <= 0.60 + 1e-9:
        return "r81_rt_sw_%s_hardabs_v%s_pfn3_%s" % (bag, t, q), "R81", False
    return "r84_%s_l2_%s_%s_%s_v%s_%s" % (ovl, bag, fit, ln, t, q), "R84", True


def cell_meta(a, fit="", ln="", edge="", layer="", sw_ms="", tbpn="", l2_ref_id="", l2_source=""):
    a.update(fit=fit, lines=ln, edge=str(edge), layer=str(layer), sw_ms=str(sw_ms), tbpn=tbpn, l2_ref=l2_ref_id, l2_source=l2_source)


# ---------------------------------------------------------------------------------------------------------------------------------- arm builders
def arms(C):
    A = []
    # ---- T3 (1,296) and L2R (846), cells in the order bag, overlap mode, fit, lines, Q, edge
    for bag in BAGS:
        for ovl in OVLS:
            for fit in FITS:
                for ln in LNS:
                    for q in QS:
                        for e in EDGES_T3:
                            src = cell_src(C, bag, q, fit, ln, e)
                            rid, rsrc, new = l2_ref(bag, q, ovl, fit, ln, e)
                            cellname = "%s %s %s lines %s edge %s %s" % (bag, ovl, fit, ln, e, q)
                            a = derive(C, src, "r84_%s_t3_%s_%s_%s_v%s_%s" % (ovl, bag, fit, ln, tg(e), q), ovl, "T3", "T3", {K_MAXL: 0}, ASK["T3"] + "; " + ASK_MODES,
                                       "R84 Tier 3: max_layer 0, %s (layer-2 reference %s, %s)" % (cellname, rid, rsrc + (" (new arm)" if new else " (existing arm, not re-run)")))
                            a["control"] = rid
                            cell_meta(a, fit, ln, e, 0, "", "", rid, rsrc)
                            A.append(a)
                            if new:
                                b = derive(C, src, rid, ovl, "L2R", "L2R", {K_MAXL: 2}, ASK["L2R"] + " (control = " + a["id"] + "); " + ASK_MODES,
                                           "R84 layer-2 reference: max_layer 2, %s" % cellname)
                                b["control"] = a["id"]
                                cell_meta(b, fit, ln, e, 2, "", "", a["id"], "R84")
                                A.append(b)
    # ---- CMP (216): the R83 baseline cell (rt, pfn 3, ds first, lines off, 0.425, max_layer 2) with the sample window and the deskew time term varied
    for bag in BAGS:
        for q in QS:
            for ms in SW_MS:
                for tname, tval in TBPNS:
                    ex = {K_MAXL: 2, K_TBPN: tval}
                    if ms:
                        ex[K_SW] = ms / 1000.0
                    a = derive(C, C.base3(bag, q), "r84_rt_cmp_%s_w%dms_%s_%s" % (bag, ms, tname, q), "rt", "CMP", "CMP", ex, ASK["CMP"] + "; " + ASK_MODES,
                               "R84 F-520 comparison: sample window %d ms, imu/undistort/time_based_process_noise %s, %s %s" % (ms, tval, bag, q), msglog=1 if (ms == 0 and tval == "var_acc") else 0)
                    a["control"] = "r83_rt_ml_%s_l2_%s" % (bag, q)               # the R83 baseline arm of the same cell (R83 return)
                    a["r82_control"] = "r82_rt_ds_%s_first_%s" % (bag, q)
                    a["same_batch_control_id"] = "r84_rt_cmp_%s_w0ms_vacc_%s" % (bag, q)
                    cell_meta(a, "pca", "off", 0.425, 2, ms, tval, a["control"], "R83")
                    A.append(a)
    A += own_arms(C, A)
    for a in A:
        a["same_batch_control"] = a.pop("same_batch_control_id", "") if a["group"] == "CMP" else ""
        if a["group"] == "CMP":      # alignfree pairing: the R82 control of the same bag and Q (as in R83); the R83 arm stays in l2_ref
            a["control"] = a["r82_control"]
        a["mode_line"] = a["mode"]
        a["alignfree"] = 1 if a["group"] in ALIGNFREE_GROUPS else 0
        a["concurrency"] = ""
        a["rss_abort_frac"] = ""
        for k in ("fit", "lines", "edge", "layer", "sw_ms", "tbpn", "l2_ref", "l2_source"):
            a.setdefault(k, "")
    return A


def _smoke(C, aid, q, extra, why, mode="off", bag="eee_01", v=0.425):
    a = R80.mk(aid, bag, mode, v, q, OWN_GROUP, OWN + "rule 3.6 smoke arm: %s; 60 s, eee_01, %s m, pfn 3, %s Q, overlap_policy default (dc)" % (why, v, Q_NAME[q]),
               "OWN smoke: %s" % why, pfn=3, extra=dict(extra), dur=R80.DP_DUR, batch="r84_id", tier="EF", kind="smoke")
    a.update(fam="SMK", ovl="dc", src_arm="(new key or key combination, no source arm; R80 smoke pattern)", flag="", fam_kind="smoke", batch="r84_id", ref=None, control="", r82_control="", dump_frames="")
    return a


def own_arms(C, A):
    S, out = C.src, []
    idinfo = (("dc", "fq", "r84_dc_id_eee_01_fq", "r82_dc_ds_eee_01_first_fq", {}), ("rt", "fq", "r84_rt_id_eee_01_fq", "r82_rt_ds_eee_01_first_fq", {}),
              ("rt", "mq", "r84_rt_id_eee_01_mq", "r82_rt_ds_eee_01_first_mq", {}), ("rt", "fq", "r84_rt_id_sw0_eee_01_fq", "r82_rt_ds_eee_01_first_fq", {K_SW: 0.0}))
    for ovl, q, aid, ref, ex in idinfo:
        a = derive(C, C.base3("eee_01", q), aid, ovl, OWN_GROUP, "ID", ex, OWN + "identity arm (the four R83 identity arms re-run on the R84 build): whole bag eee_01, %s, %s Q, every R84 key at its default%s; the state must equal R82 %s (state_trace and pos_trace within 0.005 mm); "
                   "the temporal keys removed in patch 1 were not set by it" % (ovl, Q_NAME[q], " except lio/point_cov/sample_window_s set explicitly to 0" if ex else "", ref), "OWN identity: %s %s vs %s" % (ovl, q, ref), kind="id", ref=ref, msglog=1)
        a["r82_control"] = ref
        a["control"] = ""
        out.append(a)
    for bag in BAGS:
        o = next(x for x in A if x["id"] == "r84_rt_cmp_%s_w5ms_vacc_fq" % bag)
        d = copy.deepcopy(o)
        d.update(id="r84_own_det_x_%s" % bag, kind="det_fb", fam_kind="det_fb", ref=o["id"], group=OWN_GROUP, fam="DET", msglog=0, flag="", control="", r82_control="", fit="", lines="", edge="", layer="", sw_ms="", tbpn="", l2_ref="", l2_source="")
        d.pop("same_batch_control_id", None)
        d["asked"] = OWN + "rule 3.7 determinism control (one repeat per bag queue, rule 1.16): an exact whole-bag repeat of %s (same batch; sample window 5 ms, so the new code is the code repeated); its state-trace digest must equal that arm's" % o["id"]
        d["purpose"] = "OWN determinism control: repeat of %s" % o["id"]
        d["batch"] = queue_of(d)
        out.append(d)
    sm = lambda aid, q, extra, why, mode="off", v=0.425: _smoke(C, aid, q, extra, why, mode=mode, v=v)
    out.append(sm("r84_dc_sm_sw_60s", "fq", {K_SW: 0.005}, "lio/point_cov/sample_window_s = 0.005 (F-520): the engagement line 'lio/point_cov sample_window_s=...' is present with non-zero point_evaluations and a positive mean_trace_m2; the parameter dump echoes the key"))
    out.append(sm("r84_dc_sm_l0_pca_on_60s", "fq", dict(DIST_KEY, **{K_MAXL: 0}), "max_layer 0 with hard absolute lines on the PCA fit (Tier 3 cell family pca / lines on): the job is accepted and the params echo shows voxel_map/map/max_layer 0 and line_residual/enabled true", mode="hardabs", v=0.5))
    out.append(sm("r84_dc_sm_l0_deb_off_60s", "fq", dict(DB_KEYS, **{K_MAXL: 0}), "max_layer 0 with the debiased fit and lines off (Tier 3 cell family deb / lines off): accepted, params echo shows plane_fit_mode debiased and max_layer 0", v=0.5))
    out.append(sm("r84_dc_sm_l0_deb_on_60s", "fq", dict(DIST_KEY, **GAP, **REASONS, **{K_FIT: "debiased", K_CLASSIFY: True, K_EQUIV: "mean_cov", K_MAXL: 0}),
                  "max_layer 0 with the debiased fit and hard absolute lines (R82 dbcm keys; Tier 3 cell family deb / lines on): accepted, params echo shows classify_at_fit true, debiased_line_equiv mean_cov and max_layer 0", mode="hardabs", v=0.5))
    R = S["r80_refuse_rows_distance_smooth"]
    refs = (("point_cov_mode", {"lio.point_cov.mode": "propagated"}, ["lio/point_cov/mode"]),
            ("point_cov_window_s", {"lio.point_cov.window_s": 0.01}, ["lio/point_cov/window_s"]),
            ("time_cov_mode", {"lio.time_cov.mode": "shared"}, ["lio/time_cov/mode"]),
            ("time_cov_sigma_s", {"lio.time_cov.sigma_s": 0.0001}, ["lio/time_cov/sigma_s"]),
            ("sample_window_neg", {K_SW: -0.01}, ["lio/point_cov/sample_window_s"]),
            ("sample_window_big", {K_SW: 0.2}, ["lio/point_cov/sample_window_s"]))
    for suf, delta, needles in refs:
        r = copy.deepcopy(R)
        r.update(id="r84_dc_refuse_%s" % suf, delta=dict(delta), needles=needles, group=OWN_GROUP, fam="REF", ovl="dc", src_arm="", flag="", fam_kind="refuse", batch="r84_id", kind="refuse", ref=None, control="", r82_control="", dump_frames="")
        r["asked"] = OWN + "rule 3.4 refusal: MUST FAIL at startup, text contains %s; its exit_status.txt must be non-zero" % " and ".join(repr(n) for n in needles)
        r["purpose"] = "MUST FAIL: %s" % suf
        out.append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- counts
def group_counts(ALL):
    cnt = {}
    for a in ALL:
        cnt[a["group"]] = cnt.get(a["group"], 0) + 1
    return cnt


def check_counts(ALL, die=sys.exit):
    cnt = group_counts(ALL)
    rows, bad = [], False
    for g in list(EXPECT) + [OWN_GROUP]:
        have, want = cnt.get(g, 0), (EXPECT.get(g) if g != OWN_GROUP else OWN_EXPECT)
        ok = have == want
        bad |= not ok
        rows.append("%-5s spec %-6s real %-6d %s" % (g, want, have, "" if ok else "MISMATCH"))
    extra = set(cnt) - set(EXPECT) - {OWN_GROUP}
    if extra:
        bad = True
        rows.append("UNKNOWN GROUPS: %s" % sorted(extra))
    rows.append("total spec %d + OWN %d = %d, real %d" % (sum(EXPECT.values()), OWN_EXPECT, sum(EXPECT.values()) + OWN_EXPECT, len(ALL)))
    table = "\n".join(rows)
    if bad:
        die("COUNT MISMATCH against the SPEC arm-group table. Real counts follow.\n" + table)
    return table


# ---------------------------------------------------------------------------------------------------------------------------------- required outputs and ledger
def req_of(o, aid):
    r = list(R80.req_of(o))
    if o.get("voxel_map.plane.plane_fit_reasons_en"):
        r.append("plane_fit_reasons.csv")
    return r


def required_outputs(table):
    by = {t["arm"]: t for t in table}
    sel = lambda f: [t["arm"] for t in table if f(t)]
    nonref = sel(lambda t: t["arm_kind"] != "refuse")
    nonsw = sel(lambda t: t["arm_kind"] != "refuse" and t["group"] not in SWEEP_GROUPS)
    whole = lambda t: t["duration_s"] == "full"
    R = []

    def add(fu, title, arms_, file, cols, min_rows, per=None, where="return", source="arm", eb=None, maxb=400000, **extra):
        e = eb if eb is not None else (int(sum(per(by[a]) for a in arms_)) if (where == "return" and arms_ and per) else 0)
        R.append(dict(followup=fu, title=title, arms=arms_, file=file, columns=cols, min_rows=min_rows, where=where, source=source, contains={}, expected_bytes=int(e), max_bytes_per_file=maxb, **extra))
    n = lambda t: R81.nscan(t)
    add("F-484", "per-scan position, yaw and quaternion of every whole-bag arm except the Tier 3 and layer-2 reference sweep arms (those return summaries only, plus a trace when gt_max_mm > 500; a ceiling is ledgered)",
        [a for a in nonsw if whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 1500, per=lambda t: 56 * n(t), maxb=330000)
    add("F-484", "per-scan position, yaw and quaternion of every short arm", [a for a in nonsw if not whole(by[a])], "pos_trace_all.csv.gz", POS_COLS, 5, per=lambda t: 56 * n(t), maxb=330000)
    mg = sel(lambda t: t["msglog"] == "1")
    add("F-484", "per-message LiDAR log (17 columns incl. retime_shift_s, n_trimmed) of the identity arms and the 18 CMP baseline arms (window 0, var_acc)", mg, "lidar_msg_log_all.csv.gz", MSG_COLS_C, 1500, per=lambda t: 100 * n(t), maxb=900000)
    add("F-484", "collector by-product per arm (line_rows_per_scan.csv), ledgered", nonsw, "line_rows_per_scan_all.csv.gz", [], 1, per=lambda t: 25000 if whole(t) else 10000, maxb=60000)
    ntu_whole = [a for a in nonsw if whole(by[a])]
    add("F-452", "per-scan table of every non-sweep whole-bag arm (arm_scan.csv; thinned by the collector --level to fit the ledger); the Tier 3 and reference sweep arms return phase_summary rows only", [], "arm_scan_all.csv.gz", [], 5,
        eb=min(60000000, sum(120 * n(by[a]) for a in ntu_whole)), source="generated", maxb=400000000)
    add("F-452", "phase_summary row of every arm (metrics, onset scan, collapse flags)", [], "phase_summary.csv", ["arm"], len(nonref), eb=4000 * len(nonref), source="generated", maxb=40000000)
    add("F-409", "engagement of every dispatched arm concatenated (includes the F-520 engagement line of every arm that sets the sample window)", [], "engagement_all.txt", [], 10, eb=2500 * len(table), source="generated", maxb=16000000)
    add("F-409", "per-arm completeness (arm_table 'requires' column)", [], "completeness.csv", ["arm", "requires_missing"], 10, eb=1200 * len(table), source="generated", maxb=6000000)
    add("F-326", "exit_status.txt of every dispatched arm: refuse arms non-zero, all others zero", sel(lambda t: True), "exit_status.txt", [], 1, where="archive")
    add("F-407", "engagement_imu.txt present for every dispatched arm that is not a refuse arm", nonref, "engagement_imu.txt", [], 1, where="archive")
    add("F-496", "state_reference.txt of every non-sweep arm (first-frame reference of Bryce's metric rule)", nonsw, "state_reference.txt", [], 1, per=lambda t: 400, maxb=4000)
    af = sel(lambda t: t["group"] in ALIGNFREE_GROUPS)
    add("F-496", "alignment-free stationary drift, excursion, settled offset and departure of the CMP arms ONLY (%d arms; the R84 alignfree tool compares each with the R82 control named in arm_table column control)" % len(af),
        af, "alignfree_summary.csv", [], len(af), eb=1500 * len(af), source="generated", maxb=8000000)
    add("F-452", "gates of R84: identity, determinism, grid identity per mode, key engagement per axis, refusals, exit status, completeness", [], "r84_gates.txt", [], 1, eb=6000000, source="generated", maxb=16000000)
    ids = sel(lambda t: t["arm_kind"] == "id")
    add("F-452", "identity comparison of the four identity arms against their R82 references", [], "r84_identity_vs_old.csv", ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad"], len(ids), eb=3000, source="generated", maxb=40000)
    add("F-452", "determinism controls and their originals", [], "identity_controls.txt", [], 1, eb=40000, source="generated", maxb=150000)
    t3 = sel(lambda t: t["group"] == "T3")
    cmp_ = sel(lambda t: t["group"] == "CMP")
    add("R84-T3", "one row per Tier 3 arm: cell, ATE and maximum error (gt_ate_mm, gt_max_mm), collapse flag, the layer-2 reference (arm, source, ATE, maximum error) and the paired difference in mm (reducers/r84_t3_reduce.py)", [], "r84_t3_rows.csv",
        ["arm", "bag", "overlap", "fit", "lines", "q", "edge", "gt_ate_mm", "gt_max_mm", "collapsed", "l2_ref", "l2_source", "l2_gt_ate_mm", "l2_gt_max_mm", "l2_collapsed", "d_ate_mm"], len(t3), eb=300 * len(t3), source="generated", maxb=2000000)
    add("R84-T3", "collapse count per edge and configuration, layer 0 against layer 2", [], "r84_t3_collapse_counts.csv", [], 1, eb=60000, source="generated", maxb=400000)
    add("R84-T3", "paired ATE difference (layer 0 minus layer 2, native mm): sign counts and median per edge and configuration", [], "r84_t3_paired.csv", [], 1, eb=120000, source="generated", maxb=800000)
    add("R84-T3", "best non-collapsed ATE per bag and configuration (layer 0 and layer 2) and the working range per bag", [], "r84_t3_best.csv", [], 1, eb=60000, source="generated", maxb=400000)
    add("R84-CMP", "one row per CMP arm: window, time term, ATE and maximum error, paired difference to the window-0 var_acc arm of the same cell (same batch) and to the R83 arm of the cell (native mm), engagement values of the F-520 line", [], "r84_cmp_rows.csv",
        ["arm", "bag", "q", "sw_ms", "tbpn", "gt_ate_mm", "gt_max_mm", "d_ate_vs_w0_vacc_mm", "d_ate_vs_r83_mm", "point_evaluations", "clamped_points", "mean_trace_m2", "max_trace_m2"], len(cmp_), eb=400 * len(cmp_), source="generated", maxb=1000000)
    add("R84-CMP", "per window and time term: sign counts and median of the paired difference over the 18 cells", [], "r84_cmp_summary.csv", [], 1, eb=20000, source="generated", maxb=200000)
    add("R84-F520", "F-520 engagement per arm that sets lio/point_cov/sample_window_s (value, nodes, point_evaluations, clamped_points, mean_trace_m2, max_trace_m2; parsed from engagement_imu.txt / run.log)", [], "r84_sample_window_engagement.csv", [], len(cmp_) // 2, eb=500 * len(cmp_), source="generated", maxb=1000000)
    add("R84-T3", "plane-fit rejection reasons of the debiased arms and of the smoke arms that set plane_fit_reasons_en (host archive)", sel(lambda t: "plane_fit_reasons.csv" in t["requires"].split()), "plane_fit_reasons.csv", [], 1, where="archive")
    add("R84-T3", "reference list: which layer-2 arm serves each Tier 3 cell (existing arm from R81 / R82A / R83, or the new R84 L2R arm)", [], "l2_reference_map.tsv", ["tier3_arm", "l2_ref", "l2_source"], len(t3), eb=250 * len(t3), source="generated", maxb=1000000)
    return R


MEASURED = R81.MEASURED


def ledger(RO, table):
    out = []
    for e in RO:
        if e["where"] != "return" or not e["expected_bytes"]:
            continue
        mult = 1.5 if e["file"] in MEASURED else 3.0
        out.append({"pattern": e["file"], "max_bytes_per_file": e["max_bytes_per_file"], "max_total_bytes": int(e["expected_bytes"] * mult),
                    "basis": ("measured on the R80 / R81 returns (per-arm size taken from the R82 ledger), allowance 1.5x" if e["file"] in MEASURED else "estimate (new files unmeasured), allowance 3x"), "note": e["followup"] + " (%d arms)" % len(e["arms"])})
    res, at = [], {}
    for x in out:
        if x["pattern"] in at:
            y = res[at[x["pattern"]]]
            y["max_bytes_per_file"] = max(y["max_bytes_per_file"], x["max_bytes_per_file"]); y["max_total_bytes"] += x["max_total_bytes"]; y["note"] += " + " + x["note"]
        else:
            at[x["pattern"]] = len(res); res.append(dict(x))
    out = res
    nsw = sum(1 for t in table if t["group"] in SWEEP_GROUPS)
    for x in out:
        if x["pattern"] == "pos_trace_all.csv.gz":
            x["max_total_bytes"] += nsw * 56 * 4300
            x["basis"] += "; plus a ceiling for ALL %d Tier 3 and layer-2 sweep arms whose trace is returned when gt_max_mm > 500 (R82 ceilinged 150 of its 162)" % nsw
    nn = len(table)
    for pat, mx, tot in (("summaries_all.json", 80000000, 16000 * nn), ("stop_reasons.csv", 4000000, 400 * nn), ("listener_*.log", 5000000, 80000000), ("R84_tree.tar.gz", 80000000, 80000000), ("agent_edits.patch", 2000000, 2000000),
                         ("raw_sha256.txt", 20000, 20000), ("compress_manifests.json", 100000, 100000), ("*.md", 1000000, 4000000), ("*.json", 2000000, 20000000), ("*.jsonl", 2000000, 10000000), ("*.txt", 1000000, 10000000),
                         ("*.tsv", 1000000, 3000000), ("*.csv", 3000000, 10000000), ("*.sha256", 100000, 100000), ("imu_timing/*", 4000000, 40000000)):
        out.append({"pattern": pat, "max_bytes_per_file": mx, "max_total_bytes": int(tot), "basis": "generic ceiling (R83 ledger scaled to the arm count); not a measurement", "note": "non-arm file"})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- build
COLS = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag", "win", "msglog",
        "mode", "fam", "src_arm", "arm_kind", "ref", "needles", "flag",
        "control", "r82_control", "alignfree", "concurrency", "rss_abort_frac", "dump_frames", "same_batch_control",
        "fit", "lines", "edge", "layer", "sw_ms", "tbpn", "l2_ref", "l2_source"]          # the last eight are new in R84 (appended; the R83 column set and order are unchanged)
RAW_ADD = ["plane_fit_reasons.csv"]


def build(a, J, grid_retime, grid_bag, onsets, out_dir, results_root, min_free_gb=150.0, workers=6, quiet=False, onsets_src=None, grid_bag_src=None, ref_values=None):
    for bag in BAGS:
        if bag not in grid_bag or bag not in grid_retime:
            sys.exit("grid files have no entry for %s" % bag)
        if grid_bag[bag]["seq_gaps"] != 0 or grid_retime[bag]["seq_gaps"] != 0:
            sys.exit("%s: seq_gaps != 0 in a grid file; stop" % bag)
    for bag in BAGS:
        if bag not in onsets:
            sys.exit("no onset scan for %s" % bag)
    os.makedirs(out_dir, exist_ok=True)
    C = R81.Ctx(J, grid_retime, grid_bag, onsets)
    ALL = arms(C)
    counts_txt = check_counts(ALL)
    out, table, seen = {}, [], set()
    prio = {"OWN": 0, "CMP": 1, "T3": 2, "L2R": 3}
    key = lambda i: (ALL[i]["batch"] != "r84_id", prio.get(ALL[i]["group"], 9), i)
    order = sorted(range(len(ALL)), key=key)
    for i in order:
        r = ALL[i]
        assert r["id"] not in seen, "duplicate arm id " + r["id"]
        seen.add(r["id"])
        assert r.get("asked"), "arm without an asked line: " + r["id"]
        if r["base"] not in J:
            sys.exit("base job %s not in r66_jobs.json" % r["base"])
        j = copy.deepcopy(J[r["base"]])
        j["job_id"] = r["id"]
        o = j["overrides"]
        for k, v in r["delta"].items():
            o[k] = v
        if o.get("estimator.mode") == "decoupled":
            o["estimator.mode"] = "iekf"
        o.pop("spline.mode", None)
        for k in o:
            if "perturb" in k.lower():
                sys.exit("perturbation key forbidden: " + k)
        o["eval.state_trace_run_id"] = r["id"]
        path = "%s/%s/%s" % (results_root, r["batch"], r["id"])
        o["outputs.path"] = path; o["outputs.debug_log_dir"] = path
        if r["dur"]:
            j["duration_secs"] = r["dur"]
        out.setdefault(r["batch"], []).append(j)
        sampling = "pfn%s_ds%s" % (o.get("cbk.lidar.point_filter_num", "def"), o.get("imu.ds.mode", "def"))
        req = [] if r["kind"] == "refuse" else req_of(o, r["id"])
        table.append(dict(arm=r["id"], tier=r["tier"], arch="iekf_raw", kind="closed", duration_s=str(j.get("duration_secs", "full")), seq=j["seq"], notes=r["purpose"], keep_raw="0", dup_of="",
                          r66_id=r["base"], group=r["group"], batch=r["batch"], sampling=sampling, requires=" ".join(dict.fromkeys(req)), f417=str(r["f417"]), plane_check="0",
                          asked=r["asked"], early=str(r["early"]), pfn=str(r["pfn"]), pe="0", dump=str(r["dump"]), bag=r["bag"], win=str(r["win"]), msglog=str(r["msglog"]),
                          mode=r["ovl"], fam=r["fam"], src_arm=r["src_arm"], arm_kind=r["kind"], ref=r.get("ref") or "", needles="|".join(r.get("needles") or []), flag=r.get("flag", ""),
                          control=r.get("control", ""), r82_control=r.get("r82_control", ""), alignfree=str(r["alignfree"]), concurrency="", rss_abort_frac="",
                          dump_frames=r.get("dump_frames", ""), same_batch_control=r["same_batch_control"],
                          fit=r["fit"], lines=r["lines"], edge=r["edge"], layer=r["layer"], sw_ms=r["sw_ms"], tbpn=r["tbpn"], l2_ref=r["l2_ref"], l2_source=r["l2_source"]))
    qlist = sorted(out)
    for b in qlist:
        json.dump(out[b], open(os.path.join(out_dir, "jobs_%s.json" % b), "w"), indent=1)
    with open(os.path.join(out_dir, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t"); w.writeheader(); w.writerows(table)
    qd = lambda b: {"dir": "ablations/_job_queue/%s" % b, "results_root": "%s/%s" % (results_root, b), "expected_jobs": len(out[b])}
    specs = [("id", ["r84_id"])] + [(b, ["r84_" + b]) for b in BAGS]
    order_lines = []
    for nm, qs in specs:
        qs = [q for q in qs if q in out]
        sp = M.listener_spec("R84-%s" % nm, [qd(q) for q in qs], results_root, min_free_gb, workers)
        fn = "listener_spec_%s.json" % nm
        for blk in sp["stages"] + sp["verify"]:
            blk["cmd"] = [c.replace("%s", fn).replace("r74_manifest", "r84_manifest") for c in blk["cmd"]]
        sp["raw"]["delete"] = sp["raw"]["delete"] + ["iekf_factor_iteration.csv", "lidar_msg_log.csv", "post_calibration_voxel_observation_support.csv", "line_rows_per_scan.csv"] + RAW_ADD
        json.dump(sp, open(os.path.join(out_dir, fn), "w"), indent=1)
        order_lines.append("%s\t%s\t%d arms\tdefault" % (fn, ",".join(qs), sum(len(out[q]) for q in qs)))
    open(os.path.join(out_dir, "queue_order.tsv"), "w").write("# listener spec\tqueues\tarms\tdispatch concurrency   (id FIRST and alone; then the nine bag specs: a preference, not a dependency, except id: rule 1.16 and the HOLD rule)\n" + "\n".join(order_lines) + "\n")
    if onsets_src:
        shutil.copyfile(onsets_src, os.path.join(out_dir, "onset_scans.json"))
    else:
        json.dump({"onset_scan_used": onsets, "source": "phase_summary.csv of the R81 BASE arms r81_rt_mg_<bag>_fq (onset_scan_used)"}, open(os.path.join(out_dir, "onset_scans.json"), "w"), indent=1)
    if grid_bag_src:
        shutil.copyfile(grid_bag_src, os.path.join(out_dir, "grid_bag.json"))
    else:
        json.dump(grid_bag, open(os.path.join(out_dir, "grid_bag.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "l2_reference_map.tsv"), "w") as f:
        f.write("tier3_arm\tl2_ref\tl2_source\tl2_new\n")
        for t in table:
            if t["group"] == "T3":
                f.write("%s\t%s\t%s\t%d\n" % (t["arm"], t["l2_ref"], t["l2_source"], 1 if t["l2_source"] == "R84" else 0))
    if ref_values:
        have = {row["arm"] for row in csv.DictReader(open(ref_values))}
        miss = sorted({t["l2_ref"] for t in table if t["group"] in ("T3", "CMP") and t["l2_source"] != "R84" and t["l2_ref"] not in have})
        if miss:
            sys.exit("reference values file lacks %d existing reference arms, e.g. %s" % (len(miss), miss[:5]))
        if os.path.abspath(ref_values) != os.path.abspath(os.path.join(out_dir, "l2_reference_values.csv")):
            shutil.copyfile(ref_values, os.path.join(out_dir, "l2_reference_values.csv"))
    RO = required_outputs(table)
    json.dump(RO, open(os.path.join(out_dir, "required_outputs.json"), "w"), indent=1)
    LG = ledger(RO, table)
    json.dump(LG, open(os.path.join(out_dir, "r84_file_size_ledger.json"), "w"), indent=1)
    write_compliance(out_dir, ALL, table)
    byid = {r["id"]: r for r in ALL}
    tot_exp = sum(e["expected_bytes"] for e in RO if e["where"] == "return")
    tot_led = sum(x["max_total_bytes"] for x in LG if x["note"] != "non-arm file")
    plan, plan_led = R81.part_plan(tot_exp), R81.part_plan(tot_led)
    info = dict(counts=counts_txt, total=len(table), queues={b: len(out[b]) for b in qlist}, expected_return_bytes=tot_exp, ledger_return_bytes=tot_led, plan=plan, plan_ledger=plan_led, groups=group_counts(ALL))
    if not quiet:
        print(counts_txt)
        print("arms %d; queues %d: %s; listener specs %d" % (len(table), len(qlist), ", ".join("%s=%d" % (b, len(out[b])) for b in qlist), len(specs)))
        print("expected return %.1f MB; ledger ceiling %.1f MB; exceeds 30 MB: %s; parts at 28 MiB: %d / %d" % (tot_exp / 1e6, tot_led / 1e6, plan["exceeds_30MB"], plan["parts_split_28M"], plan_led["parts_split_28M"]))
    verify(out, table, ALL, byid)
    return info


def verify(out, table, ALL, byid):
    ids = [t["arm"] for t in table]
    assert len(ids) == len(set(ids)), "arm ids not unique"
    for b, lst in out.items():
        assert len({j["job_id"] for j in lst}) == len(lst), b
    removed = ("lio.point_cov.mode", "lio.point_cov.window_s", "lio.time_cov.mode", "lio.time_cov.sigma_s")
    for t in table:
        assert t["asked"].strip(), t["arm"]
        assert t["mode"] in ("dc", "rt"), t["arm"]
        r = byid[t["arm"]]
        if t["mode"] == "rt":
            assert r["delta"].get("cbk.lidar.overlap_policy") == "retime_trim", t["arm"]
        else:
            assert "cbk.lidar.overlap_policy" not in r["delta"], t["arm"]
        assert not any("perturb" in k.lower() for k in r["delta"]), t["arm"]
        assert not any("zupt" in k.lower() for k in r["delta"]), "ZUPT key in " + t["arm"]
        if t["arm_kind"] != "refuse":
            assert not any(k in r["delta"] for k in removed), "removed key in " + t["arm"]
        if t["group"] in SWEEP_GROUPS:
            assert r["delta"][K_MAXL] == int(t["layer"]), t["arm"]
            assert abs(r["delta"][K_VOX] - float(t["edge"])) < 1e-9, t["arm"]
            assert r["delta"].get("cbk.lidar.point_filter_num") == 3 and r["delta"].get("imu.ds.mode", "first") == "first", t["arm"]
            assert (r["delta"].get(K_FIT) == "debiased") == (t["fit"] == "deb"), t["arm"]
            assert bool(r["delta"].get(K_LINE)) == (t["lines"] == "on"), t["arm"]
    assert any(t["batch"] == "r84_id" for t in table), "no r84_id queue"


def write_compliance(out_dir, ALL, table):
    with open(os.path.join(out_dir, "COMPLIANCE_TABLE.md"), "w") as f:
        f.write("# R84 compliance table (rule 1.10 item 2, 1.12)\n\nMarks: MATCHES = literally what Bryce said; OWN = the planner's rule arm (needs his sight); ASSUMPTION = a value or choice he did not give; ASSUMPTION-READING = the family itself is the planner's reading of his message (both quoted). The reason is in the last column.\n\n"
                "| arm | Bryce's words | mode | source arm | mark | note |\n|---|---|---|---|---|---|\n")
        for t in table:
            r = next(x for x in ALL if x["id"] == t["arm"])
            g, mark, note = r["group"], "MATCHES", ""
            words = "OWN rule (identity / determinism / smoke / refuse); not asked for in Bryce's words" if g == OWN_GROUP else ASK[g]
            if g == OWN_GROUP:
                mark = "OWN"
            elif g == "T3":
                if t["fit"] == "deb" and t["lines"] == "on":
                    mark, note = "ASSUMPTION", "debiased with lines on = the R82 dbcm key set (hard absolute, distance rows, gap floor, classify_at_fit, debiased_line_equiv mean_cov) because that is the only tested debiased line configuration; the PCA lines-on cell is the R80 hardabs arm (distance rows, no classify_at_fit, no floor): the two lines-on cells differ in more than the fit"
                elif t["fit"] == "deb":
                    mark, note = "ASSUMPTION", "debiased keys copied from the R82 dbctl arms (DB_KEYS: plane_fit_mode debiased, reasons, trace frames)"
                elif t["lines"] == "on":
                    mark, note = "ASSUMPTION", "lines on = hard, absolute, distance rows (the R80 sweep hardabs source arm); Bryce's wording 'hard, absolute'; rows distance and the absence of classify_at_fit follow the R81 hardabs sweep"
                else:
                    note = "source = R80 sweep arm of the cell; only max_layer differs from the layer-2 sweep arm (%s)" % t["l2_ref"]
            elif g == "L2R":
                mark, note = "ASSUMPTION", "the planner's reading of 'layer-2 reference arms for cells the existing sweeps lack' (Tier 3 design): same cell as %s with max_layer 2; existing arms are reused where the cell matches exactly (list: l2_reference_map.tsv)" % t["control"]
            elif g == "CMP":
                note = "R83 baseline cell (rt, pfn 3, ds first, lines off, 0.425, max_layer 2); time_based_process_noise written explicitly (var_acc = the default); sample_window_s omitted at 0 ms"
            f.write("| %s | %s | %s | %s | %s | %s |\n" % (t["arm"], words.replace("|", "/"), t["mode"], t["src_arm"] or "(none)", mark, note))


# ---------------------------------------------------------------------------------------------------------------------------------- selftest
R81_R82_COLS = ["arm", "tier", "arch", "kind", "duration_s", "seq", "notes", "keep_raw", "dup_of", "r66_id", "group", "batch", "sampling", "requires", "f417", "plane_check", "asked", "early", "pfn", "pe", "dump", "bag", "win", "msglog",
                "mode", "fam", "src_arm", "arm_kind", "ref", "needles", "flag"]


def selftest():
    J = R81.stub_jobs(); g, gb = R81.stub_grids()
    onsets = {b: 300 + 10 * i for i, b in enumerate(BAGS)}
    d = tempfile.mkdtemp(prefix="r84_selftest_")
    info = build(None, J, g, gb, onsets, d, "/root/catkin_ws/livo_recon_results", quiet=True)
    print(info["counts"])
    n_ok = [0]

    def ok(c, m):
        assert c, m
        n_ok[0] += 1
    tab = list(csv.DictReader(open(os.path.join(d, "arm_table.tsv")), delimiter="\t"))
    TOT = sum(EXPECT.values()) + OWN_EXPECT
    ok(info["total"] == TOT == len(tab) == 2381, "total 2381")
    ok(list(tab[0].keys())[:31] == R81_R82_COLS and list(tab[0].keys())[31:38] == ["control", "r82_control", "alignfree", "concurrency", "rss_abort_frac", "dump_frames", "same_batch_control"] and list(tab[0].keys()) == COLS, "arm_table columns")
    for gname, want in dict(EXPECT, OWN=OWN_EXPECT).items():
        ok(sum(1 for t in tab if t["group"] == gname) == want, "group count " + gname)
    ok(len({t["arm"] for t in tab}) == len(tab), "names unique")
    jobs, qjobs = {}, {}
    for fn in os.listdir(d):
        if fn.startswith("jobs_"):
            lst = json.load(open(os.path.join(d, fn)))
            qjobs[fn[5:-5]] = lst
            for j in lst:
                jobs[j["job_id"]] = j["overrides"]
    ok(len(jobs) == TOT, "jobs total")
    qc = {q: len(v) for q, v in qjobs.items()}
    ok(qc["r84_id"] == 4 + 4 + 6 and all(qc["r84_" + b] == 144 + 94 + 24 + 1 for b in BAGS) and sum(qc.values()) == TOT and len(qc) == 10, "queue sizes: id 14, nine bag queues 263")
    by = {t["arm"]: t for t in tab}
    t3 = [t for t in tab if t["group"] == "T3"]
    ok(sorted({float(t["edge"]) for t in t3}) == EDGES_T3 and EDGES_T3[0] == 0.3 and EDGES_T3[-1] == 0.7 and len(EDGES_T3) == 9, "edges 0.30 .. 0.70")
    ok(len({(t["bag"], t["mode"], t["fit"], t["lines"], t["edge"], t["arm"][-2:]) for t in t3}) == 1296 and len({t["l2_ref"] for t in t3}) == 1296, "T3 cells distinct, one reference each")
    for t in t3:
        o = jobs[t["arm"]]
        assert o[K_MAXL] == 0 and o[K_VOX] == float(t["edge"]) and o["cbk.lidar.point_filter_num"] == 3 and o["imu.ds.mode"] == "first", t["arm"]
        assert (t["mode"] == "rt") == (o.get("cbk.lidar.overlap_policy") == "retime_trim"), t["arm"]
        assert o["imu.process_noise.model"] == ("fixed" if t["arm"].endswith("_fq") else "isotropic"), t["arm"]
        assert (o.get(K_FIT) == "debiased") == (t["fit"] == "deb") and bool(o.get(K_LINE)) == (t["lines"] == "on"), t["arm"]
        if t["fit"] == "deb" and t["lines"] == "on":
            assert o[K_CLASSIFY] is True and o[K_EQUIV] == "mean_cov" and o[R9.GAP] is True and o["voxel_map.plane.line_residual.rows"] == "distance", t["arm"]
        if t["fit"] == "pca":
            assert K_CLASSIFY not in o and K_EQUIV not in o and R9.GAP not in o, t["arm"]
        assert not any(k in o for k in ("lio.point_cov.mode", "lio.point_cov.window_s", "lio.time_cov.mode", "lio.time_cov.sigma_s", K_SW)), t["arm"]
    n_ok[0] += 1
    l2r = [t for t in tab if t["group"] == "L2R"]
    ok(all(jobs[t["arm"]][K_MAXL] == 2 for t in l2r), "L2R max_layer 2")
    src = {}
    for t in t3:
        src[t["l2_source"]] = src.get(t["l2_source"], 0) + 1
    ok(src == {"R81": 252, "R82A": 36, "R83": 162, "R84": 846} and len(l2r) == 846, "reference sources " + str(src))
    ok(all(by[t["control"]]["group"] == "T3" for t in l2r) and all(by[t["arm"]]["control"] == by[t["arm"]]["l2_ref"] for t in t3), "control columns")
    for t in l2r:
        a, b = jobs[t["arm"]], jobs[by[t["control"]]["arm"]]
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"outputs.path", "outputs.debug_log_dir", "eval.state_trace_run_id"}
        assert diff == {K_MAXL}, (t["arm"], diff)
    n_ok[0] += 1
    ok(by["r84_rt_t3_eee_01_pca_off_v0p3_fq"]["l2_ref"] == "r81_rt_sw_eee_01_off_v0p3_pfn3_fq" and by["r84_rt_t3_eee_01_pca_off_v0p7_mq"]["l2_ref"] == "r82_rt_sw_eee_01_off_v0p7_pfn3_mq"
       and by["r84_dc_t3_sbs_03_pca_off_v0p45_fq"]["l2_ref"] == "r83_dc_sw_sbs_03_off_v0p45_pfn3_fq" and by["r84_rt_t3_nya_02_pca_on_v0p6_mq"]["l2_ref"] == "r81_rt_sw_nya_02_hardabs_v0p6_pfn3_mq"
       and by["r84_rt_t3_nya_02_pca_on_v0p65_mq"]["l2_ref"] == "r84_rt_l2_nya_02_pca_on_v0p65_mq" and by["r84_dc_t3_nya_02_pca_on_v0p3_mq"]["l2_ref"] == "r84_dc_l2_nya_02_pca_on_v0p3_mq"
       and by["r84_rt_t3_nya_02_deb_off_v0p5_fq"]["l2_ref"] == "r84_rt_l2_nya_02_deb_off_v0p5_fq", "reference names")
    cmp_ = [t for t in tab if t["group"] == "CMP"]
    ok(len(cmp_) == 216 and sorted({int(t["sw_ms"]) for t in cmp_}) == list(SW_MS) and {t["tbpn"] for t in cmp_} == {"var_acc", "none"}, "CMP grid")
    for t in cmp_:
        o = jobs[t["arm"]]
        ms = int(t["sw_ms"])
        assert o[K_TBPN] == t["tbpn"] and o[K_MAXL] == 2 and o[K_VOX] == 0.425 and o["cbk.lidar.overlap_policy"] == "retime_trim" and not o.get(K_LINE) and (K_SW in o) == (ms > 0), t["arm"]
        assert (ms == 0) or abs(o[K_SW] - ms / 1000.0) < 1e-12, t["arm"]
        assert t["same_batch_control"] == "r84_rt_cmp_%s_w0ms_vacc_%s" % (t["bag"], t["arm"][-2:]) and t["control"] == "r82_rt_ds_%s_first_%s" % (t["bag"], t["arm"][-2:]) and t["alignfree"] == "1", t["arm"]
        assert t["l2_ref"] == "r83_rt_ml_%s_l2_%s" % (t["bag"], t["arm"][-2:]), t["arm"]
        assert t["msglog"] == ("1" if (ms == 0 and t["tbpn"] == "var_acc") else "0"), t["arm"]
    n_ok[0] += 1
    ok(sum(1 for t in tab if t["alignfree"] == "1") == 216, "alignfree column = CMP only")
    ok(by["r84_dc_id_eee_01_fq"]["ref"] == "r82_dc_ds_eee_01_first_fq" and by["r84_rt_id_eee_01_mq"]["ref"] == "r82_rt_ds_eee_01_first_mq" and by["r84_rt_id_sw0_eee_01_fq"]["ref"] == "r82_rt_ds_eee_01_first_fq", "identity refs")
    ok(jobs["r84_rt_id_sw0_eee_01_fq"][K_SW] == 0.0 and K_SW not in jobs["r84_rt_id_eee_01_fq"] and K_TBPN not in jobs["r84_rt_id_eee_01_fq"] and by["r84_dc_id_eee_01_fq"]["batch"] == "r84_id", "sw0 identity")
    ok(by["r84_own_det_x_nya_02"]["ref"] == "r84_rt_cmp_nya_02_w5ms_vacc_fq" and by["r84_own_det_x_nya_02"]["batch"] == "r84_nya_02" and jobs["r84_own_det_x_nya_02"][K_SW] == 0.005, "determinism repeat")
    idk = {k: v for k, v in jobs["r84_dc_id_eee_01_fq"].items() if k not in ("outputs.path", "outputs.debug_log_dir", "eval.state_trace_run_id")}
    ok(not any(k.startswith(("cbk.lidar.overlap", "lio.point_cov", K_TBPN)) for k in idk), "identity arm carries no new key")
    for suf, key in (("point_cov_mode", "lio.point_cov.mode"), ("point_cov_window_s", "lio.point_cov.window_s"), ("time_cov_mode", "lio.time_cov.mode"), ("time_cov_sigma_s", "lio.time_cov.sigma_s"), ("sample_window_neg", K_SW), ("sample_window_big", K_SW)):
        ok(key in jobs["r84_dc_refuse_" + suf] and by["r84_dc_refuse_" + suf]["arm_kind"] == "refuse", "refuse arm " + suf)
    ok(jobs["r84_dc_refuse_sample_window_neg"][K_SW] == -0.01 and jobs["r84_dc_refuse_sample_window_big"][K_SW] == 0.2, "refuse values")
    for sname in ("r84_dc_sm_sw_60s", "r84_dc_sm_l0_pca_on_60s", "r84_dc_sm_l0_deb_off_60s", "r84_dc_sm_l0_deb_on_60s"):
        ok(by[sname]["batch"] == "r84_id" and by[sname]["duration_s"] == "60", "smoke " + sname)
    ok(jobs["r84_dc_sm_l0_deb_on_60s"][K_MAXL] == 0 and jobs["r84_dc_sm_l0_deb_on_60s"][K_CLASSIFY] is True and jobs["r84_dc_sm_sw_60s"][K_SW] == 0.005, "smoke keys")
    ok(not any("zupt" in k for o in jobs.values() for k in o), "no ZUPT keys")
    rm_arms = {"r84_dc_refuse_point_cov_mode", "r84_dc_refuse_point_cov_window_s", "r84_dc_refuse_time_cov_mode", "r84_dc_refuse_time_cov_sigma_s"}
    ok(not any(k in o for n_, o in jobs.items() if n_ not in rm_arms for k in ("lio.point_cov.mode", "lio.time_cov.mode", "lio.point_cov.window_s", "lio.time_cov.sigma_s")), "removed keys appear only in the four removed-key refuse arms")
    sp = json.load(open(os.path.join(d, "listener_spec_eee_01.json")))
    ok("plane_fit_reasons.csv" in sp["raw"]["delete"] and "time_cov.csv" not in sp["raw"]["delete"] and sp["queues"][0]["expected_jobs"] == 263, "listener spec")
    q = open(os.path.join(d, "queue_order.tsv")).read().splitlines()
    ok(q[1].startswith("listener_spec_id.json") and len(q) == 11, "queue_order")
    RO = json.load(open(os.path.join(d, "required_outputs.json")))
    ok({e["file"] for e in RO} >= {"r84_t3_rows.csv", "r84_cmp_rows.csv", "r84_sample_window_engagement.csv", "l2_reference_map.tsv", "r84_gates.txt", "phase_summary.csv"}, "required files")
    af = [e for e in RO if e["file"] == "alignfree_summary.csv"][0]
    ok(len(af["arms"]) == 216 and af["min_rows"] == 216, "alignfree rows 216")
    LG = json.load(open(os.path.join(d, "r84_file_size_ledger.json")))
    ok(any(x["pattern"] == "pos_trace_all.csv.gz" for x in LG) and all(x["max_total_bytes"] > 0 for x in LG), "ledger")
    for fn in ("onset_scans.json", "grid_bag.json", "COMPLIANCE_TABLE.md", "required_outputs.json", "r84_file_size_ledger.json", "queue_order.tsv", "arm_table.tsv", "l2_reference_map.tsv"):
        ok(os.path.exists(os.path.join(d, fn)), fn)
    comp = open(os.path.join(d, "COMPLIANCE_TABLE.md")).read().splitlines()
    ok(sum(1 for l in comp if l.startswith("| r84_")) == TOT and all(l.rstrip().split("|")[-3].strip() in ("MATCHES", "OWN", "ASSUMPTION", "ASSUMPTION-READING") for l in comp if l.startswith("| r84_")), "every arm has a mark")
    C = R81.Ctx(J, g, gb, onsets); ALL = arms(C)
    try:
        check_counts([a for a in ALL if a["group"] != "CMP"], die=lambda m: (_ for _ in ()).throw(RuntimeError(m)))
        raise SystemExit("selftest FAILED: a short arm list was not refused")
    except RuntimeError as e:
        ok("COUNT MISMATCH" in str(e), "count mismatch refused")
    J2 = R81.stub_jobs(); J2["00_id_baseline_full"]["overrides"]["x.perturb_y"] = 1
    try:
        build(None, J2, g, gb, onsets, tempfile.mkdtemp(), "/r", quiet=True)
        raise SystemExit("selftest FAILED: perturbation key not refused")
    except SystemExit as e:
        ok("perturbation" in str(e), "perturbation refused")
    d2 = tempfile.mkdtemp(prefix="r84_selftest2_")
    build(None, J, g, gb, onsets, d2, "/root/catkin_ws/livo_recon_results", quiet=True)
    ok(all(open(os.path.join(d, f), "rb").read() == open(os.path.join(d2, f), "rb").read() for f in sorted(os.listdir(d))), "two builds identical")
    print("reference sources:", src)
    print("selftest checks passed: %d" % n_ok[0])
    print("failures: 0")
    print("selftest OK (synthetic job list and grids; planner sandbox, generator output only, not evidence)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jobs"); ap.add_argument("--grid-retime"); ap.add_argument("--grid-bag"); ap.add_argument("--onsets"); ap.add_argument("--ref-values")
    ap.add_argument("--out", default="r84_manifest"); ap.add_argument("--results-root", default="/root/catkin_ws/livo_recon_results")
    ap.add_argument("--min-free-gb", type=float, default=150.0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.jobs and a.grid_retime and a.grid_bag and a.onsets):
        ap.error("--jobs, --grid-retime, --grid-bag and --onsets are required")
    J = {j["job_id"]: j for j in json.load(open(a.jobs))}
    onsets = json.load(open(a.onsets))["onset_scan_used"]
    build(a, J, json.load(open(a.grid_retime)), json.load(open(a.grid_bag)), onsets, a.out, a.results_root, a.min_free_gb, a.workers, onsets_src=a.onsets, grid_bag_src=a.grid_bag, ref_values=a.ref_values)


if __name__ == "__main__":
    main()
