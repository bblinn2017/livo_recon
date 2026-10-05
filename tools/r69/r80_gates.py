#!/usr/bin/env python3
"""R80 gates: every group of the R80 manifest gets its own verdicts, and one group's absence never touches another group's (rule 1.16).

Groups: DP window dumps (+ determinism DET), SMK smoke arms, REF refusals, MG / B3 / B1 BASE arms, ZD ZUPT detect_only (D7), GP gap floor (F-433), NG two-sided NIS gate (F-476), PC pose-cov / point-cov (D8),
DB debiased fit (F-469), TL coupled tail (TAIL-1), EC3 calibration samples (D3), SC step cap (D5), SW voxel sweep (D1).

    python3 tools/r69/r80_gates.py --runs <runs_dir> [--refs-fb r80_manifest/identity_refs_fb.csv.gz] [--phase-summary phase_summary.csv] [--dispatched <arm_table.tsv | arm ids | groups>]
                                   [--out r80_gates.txt] [--csv r80_identity_vs_old.csv]
    python3 tools/r69/r80_gates.py --selftest     fake arm dirs for every arm of the R80 manifest (the sweep sampled, see below): good data all PASS, every single breakage FAILs its own gate and no other,
                                                  an empty dir all MISSING, an undispatched group is NOT RUN (INFO) and touches nothing else

runs_dir/<arm>/ = the unbundled arm_bundle.zip of each arm (r69_unbundle.py); the return's compact/<arm>/ files are read from the same directory (r69_collect copies them there before the gates run).
PASS / FAIL / MISSING (absent: never a pass) / INFO (a fact reported without a verdict) / NOTRUN (INFO row: the group's queue was not dispatched at all). Exit 1 only on a FAIL.

Independence (rule 1.16): every row names the arm (or the cell / group) it concerns and is decided from that arm's own files only. A group or queue (the unit is the pair arm-table group + queue, which for the
sweep is the cell queue sw_<bag>_<q>_<line>_p<pfn>) with NO arm in the dispatched set gets one INFO row 'NOTRUN' and no other row; the dispatched set is --dispatched (default: every arm of the manifest, so an empty
directory is all MISSING). A dispatched arm whose files are absent is MISSING on its own rows only.

Own choices (disclosed): (1) identity tolerance 0.001 mm in position and 0.001 mrad in yaw on every common scan except the last 3 of the 60 s arms (the new build only adds write-only logs, so the expected difference is 0);
(2) the frame mapping gate tries an offset of -1, 0 or +1 between the scan_id of state_trace.csv and the frame number of the update logs and PASSES when one reproduces the posterior position to 1e-6 m on every window frame;
(3) message-log consistency is the pushLidar rule itself; (4) 'ground' = |world normal z| >= 0.8; (5) INFO rows (sigma^2 decomposition, gate widths, message counts, NIS engaged = deflating_scans > 0, stationary
scans, step-cap bound) are facts, not verdicts; (6) exit_status.txt: first line = the integer exit status of the node (written by the host's run_job.sh / the agent); refuse arms must be non-zero AND not DONE, every other
arm must be 0; absent -> MISSING on that arm's EXIT row only; (7) config values are read from the lines the build prints: the application's [params/...] dump lines and engagement lines first, then the
[config/effective] echo ('key = value'), then the ROS parameter dump ('/livo_recon/key: value'); a value that cannot be found anywhere is MISSING (never a pass), a value that is found and differs is FAIL; (8) sweep
(SW) arms return no pos_trace unless gt_max_mm > 500, so the SW gates never read it: they decide from run.log, arm.json, exit_status.txt and the phase_summary.csv row; PASS rows of the sweep are folded into one
row per cell and gate (FAIL / MISSING rows stay one per arm) so the report stays inside its size ledger.
"""
import argparse, io, json, os, re, shutil, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_r80_manifest as M  # noqa: E402
import r80_window_reduce as W  # noqa: E402
from r74_gates import rd, csv, jsn, digest, G, report, line  # noqa: E402
import r79_gates as R79  # noqa: E402

TOL_POS_MM = 0.001
TOL_YAW_MRAD = 0.001
LEN_MIN = 0.97
FRAMES = list(range(M.LO, M.HI + 1))
GFRAMES = list(range(M.GLO, M.GHI + 1))
SMK_DIST = "r80_sm_dist_60s"                  # the smoke arm of rows = distance (F-453)
W60 = ("DP", "DET", "SMK", "SC")              # first 60 s arms: 560 to 700 scans
LINE2_RE = re.compile(r"plane/line_residual:\s+enabled (\w+) / form (\w+) / rule (\w+) / rel_sigmas ([0-9.eE+-]+) / shape_ratio ([0-9.eE+-]+) / enum (\d+) / variance (\w+)(?: / rows (\w+))?")
FIT_RE = re.compile(r"^\s*plane/plane_fit_mode:\s+(\w+)\s*$", re.M)
POSECOV_RE = re.compile(r"^\s*residual/pose_cov_in_sigma:\s+(\w+)\s*$", re.M)
CALIB_RE = re.compile(r"\[params/calib/stationary\][^\[]*?num_samples:\s*(\d+)")
# the NIS values Bryce accepted 2026-10-05 (F-476), written here and not read from the manifest so a wrong manifest cannot agree with itself
NG_WANT = {"nis_hi": 1.3, "inflate": 1.25, "decay": 0.95, "nis_lo": 0.75, "deflate": 0.8, "min_gain": 0.25}
SC_WANT = {"cap_rot": 0.01, "cap_pos": 0.01}  # D5: 10 mrad and 10 mm
ZD_COLS = ["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"]   # the header zupt.csv is written with (src/processing/lio_iekf.cpp)
SC_COLS = ["frame", "iter", "rot_raw", "pos_raw", "rot_applied", "pos_applied", "scale", "bound", "next_converged"]
ROWS_KEY = "voxel_map.plane.line_residual.rows"
LINE_ROWS_FILES = ("line_rows_per_scan_all.csv.gz", "line_rows_per_scan.csv")
NEEDLE_FILES = ("run_log_tail.txt", "run.log")


def tmap():
    """arm table of the manifest as dicts (the same list that built the job files)."""
    out = []
    for a in M.arms():
        out.append(dict(id=a["id"], kind=a["kind"], ref=a.get("ref"), group=a["group"], batch=a["batch"], bag=a["bag"], pfn=a["pfn"], q=a["q"], dur=a["dur"], win=a["win"], msglog=a["msglog"], voxel=a["voxel"],
                        layer=a["layer"], mode=a["mode"], pe=a["pe"], needles=a.get("needles"), delta=a["delta"]))
    return out


def unit(t):
    return (t["group"], t["batch"])


def dispatched_set(spec, table):
    """-> set of arm ids that were dispatched, or None (= all). spec: a path to an arm_table.tsv / a file of arm ids, or a comma list of arm ids / group names / queue names."""
    if spec is None:
        return None
    ids = {t["id"] for t in table}
    toks = []
    if os.path.exists(spec):
        txt = open(spec).read().splitlines()
        if txt and "\t" in txt[0] and "arm" in txt[0].split("\t"):
            col = txt[0].split("\t").index("arm")
            toks = [l.split("\t")[col] for l in txt[1:] if l.strip()]
        else:
            toks = [l.split()[0] for l in txt if l.strip() and not l.startswith("#")]
    else:
        toks = [x for x in spec.split(",") if x]
    out = set()
    for tok in toks:
        if tok in ids:
            out.add(tok)
        for t in table:
            if tok in (t["group"], t["batch"]):
                out.add(t["id"])
    return out


def cfg_get(text, key, alts=()):
    """value the build printed for `key` (slash form): alternatives (dump-line regexes with one group) first, then the [config/effective] echo 'key = v', then the ROS dump '/ns/key: v'. None if nowhere."""
    k = re.escape(key)
    pats = list(alts) + [re.compile(r"^\s*%s\s*=\s*(\S+)\s*$" % k, re.M), re.compile(r"^\s*/\S*/%s:\s*(\S+)\s*$" % k, re.M)]
    for p in pats:
        m = (p if hasattr(p, "search") else re.compile(p, re.M)).search(text or "")
        if m:
            return m.group(1).strip("'\"")
    return None


def same(v, want):
    if v is None:
        return False
    if isinstance(want, bool):
        return str(v).lower() == str(want).lower()
    if isinstance(want, (int, float)):
        try:
            return abs(float(v) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
        except ValueError:
            return False
    return str(v).lower() == str(want).lower()


def kv(ln):
    return dict(re.findall(r"([A-Za-z_0-9()]+)=(\S+)", ln or ""))


def judge(items, have_text):
    """items: list of (label, found, want). found None = nowhere in the text. -> (status, detail): FAIL if any value found and different, else MISSING if any not found (or no text), else PASS."""
    bad = ["%s: found %s, wanted %s" % (l, f, w) for l, f, w in items if f is not None and not same(f, w)]
    miss = ["%s: not found" % l for l, f, w in items if f is None]
    det = "; ".join(bad + miss) if (bad or miss) else "; ".join("%s = %s" % (l, f) for l, f, w in items)
    if bad:
        return "FAIL", det
    if miss or not have_text:
        return "MISSING", det if (bad or miss) else "no log text"
    return "PASS", det


def worst(sts):
    for s in ("FAIL", "MISSING"):
        if s in sts:
            return s
    return "PASS"


def ident_row(g, arm, ref_arm, P, refs_fb, drop_last):
    rr = refs_fb[refs_fb["arm"] == ref_arm] if refs_fb is not None else None
    nm = "IDENT %s per-scan position and yaw equal %s (%.3g mm, %.3g mrad)" % (arm, ref_arm, TOL_POS_MM, TOL_YAW_MRAD)
    if P is None or rr is None or rr.empty:
        g.add(nm, "MISSING", "pos trace %s, reference rows %s" % ("absent" if P is None else "present", "absent" if rr is None or rr.empty else "present"))
        return None
    n_com, (dp, dy), (tp, ty), n_dec = R79.compare_pos(P, rr, drop_last)
    ok = n_dec >= 100 and dp <= TOL_POS_MM and dy <= TOL_YAW_MRAD
    g.add(nm, "PASS" if ok else "FAIL", "common scans %d (decided %d), max position difference %.4g mm, max yaw difference %.4g mrad" % (n_com, n_dec, dp, dy))
    if drop_last:
        g.add("IDENT %s the last %d common scans (the run stops at the duration; a fact)" % (arm, drop_last), "INFO", "max position difference %.4g mm, yaw %.4g mrad" % (tp, ty))
    return dict(arm=arm, ref_arm=ref_arm, n_common_scans=n_com, pos_max_abs_diff_mm=dp, yaw_max_abs_diff_mrad=dy, dump_equal="", node_id_equal="", digest_equal_if_known="")


def exit_status(d):
    """first line of exit_status.txt as an int; None if the file is absent; 'bad' if it is not an integer."""
    t = rd(d, "exit_status.txt")
    if t is None:
        return None
    ln = (t.splitlines() or [""])[0].strip()
    return int(ln) if re.fullmatch(r"-?\d+", ln) else "bad"


def eng_text(d):
    return (rd(d, "engagement.txt") or "") + "\n" + (rd(d, "run.log") or "")


class SwAgg:
    """sweep arms: FAIL / MISSING rows stay one per arm; the PASS rows of a (gate, cell) fold into one row."""

    def __init__(self, g):
        self.g, self.c = g, {}

    def put(self, kind, cell, name, st, det=""):
        if st != "PASS":
            self.g.add(name, st, det)
        self.c.setdefault((kind, cell), {"PASS": 0, "FAIL": 0, "MISSING": 0})[st] += 1

    def flush(self):
        n = len(M.EDGES)
        for (kind, cell), c in self.c.items():
            if c["FAIL"] == 0 and c["MISSING"] == 0 and c["PASS"] == n:
                self.g.add("SWCELL %s %s: all %d arms PASS" % (kind, cell, n), "PASS")
            elif c["PASS"] > 0:
                self.g.add("SWCELL %s %s: arms PASS / FAIL / MISSING of the %d of the cell (the FAIL and MISSING arms have their own rows)" % (kind, cell, n), "INFO", "%d / %d / %d" % (c["PASS"], c["FAIL"], c["MISSING"]))


def ref_gates(g, d, t, pres, sent, ex):
    arm = t["id"]
    nm = "REFUSE %s dies at startup, text contains %s" % (arm, " and ".join(repr(n) for n in t["needles"]))
    if not pres:
        g.add(nm, "MISSING", "arm.json absent")
    else:
        tail = "".join(rd(d, f) or "" for f in NEEDLE_FILES)
        has = [n in tail for n in t["needles"]]
        g.add(nm, "PASS" if sent == "FAILED" and all(has) else "FAIL", "sentinel=%s needles_found=%s" % (sent, has))
    nm = "EXIT %s exit_status.txt exists and is non-zero, and the job did not reach DONE (the deliberately refused job of the F-326 check)" % arm
    if ex is None:
        g.add(nm, "MISSING", "exit_status.txt absent")
    elif ex == "bad":
        g.add(nm, "FAIL", "first line of exit_status.txt is not an integer")
    elif not pres:
        g.add(nm, "MISSING", "exit status %d but arm.json absent: cannot tell whether the job reached DONE" % ex)
    else:
        g.add(nm, "PASS" if ex != 0 and sent != "DONE" else "FAIL", "exit status %d, sentinel %s" % (ex, sent))


def cfg_gates(t, log, et):
    """-> list of (gate name, status, detail): the effective configuration of ONE arm, from its own log lines. Never reads another arm."""
    arm, dl, grp = t["id"], t["delta"], t["group"]
    out = []
    have = bool(log.strip())
    stl = lambda ok: "PASS" if ok else ("MISSING" if not have else "FAIL")
    mv, ml = re.search(r"map/voxel_size:\s+([0-9.eE+-]+)", log), re.search(r"map/max_layer:\s+(\d+)", log)
    ok = bool(mv and ml) and abs(float(mv.group(1)) - t["voxel"]) < 1e-12 and int(ml.group(1)) == t["layer"]
    out.append(("VOX %s root voxel %s max_layer %s in the effective config" % (arm, t["voxel"], t["layer"]), stl(ok), "log: voxel %s max_layer %s" % (mv and mv.group(1), ml and ml.group(1))))
    want = M.R9.WANT[t["mode"]]
    wvar = "plane_equivalent" if t["pe"] else "fit_only"
    wrows = "distance" if dl.get(ROWS_KEY) == "distance" else "per_normal"
    l = LINE2_RE.search(log)
    ok = bool(l) and (l.group(1), l.group(2), l.group(3)) == want and abs(float(l.group(4)) - 3.0) < 1e-12 and abs(float(l.group(5)) - 0.1) < 1e-12 and int(l.group(6)) == M.R9.ENUM[t["mode"]] and l.group(7) == wvar and l.group(8) == wrows
    out.append(("LINE %s effective config enabled %s / form %s / rule %s (enum %d) / variance %s / rows %s" % (arm, want[0], want[1], want[2], M.R9.ENUM[t["mode"]], wvar, wrows), stl(ok), l.group(0) if l else "no line"))
    pm = R79.PFN_RE.search(log)
    out.append(("PFN %s map point filter %s in the effective config" % (arm, t["pfn"]), "PASS" if pm and int(pm.group(1)) == t["pfn"] else ("MISSING" if not pm else "FAIL"), "log: %s" % (pm and pm.group(0))))
    gm = R79.GAP_RE.search(log)
    wg = "true" if dl.get(M.R9.GAP) else "false"
    out.append(("GAPCFG %s effective config plane_gap_floor_en %s (application line only)" % (arm, wg), "PASS" if gm and gm.group(1) == wg else ("MISSING" if not gm else "FAIL"), gm.group(0).strip() if gm else "no line"))
    if t["msglog"]:
        mk = re.search(r"data_queues/log_lidar_msgs:\s*(\w+)", log)
        out.append(("MSGKEY %s effective config data_queues/log_lidar_msgs true" % arm, "PASS" if mk and mk.group(1) == "true" else ("MISSING" if not mk else "FAIL"), "log: %s" % (mk and mk.group(0))))
    # plane fit mode (DB: debiased, every other arm: pca) and the pose-covariance switch (PC cells ps / pb: true, every other arm: false)
    wfit = dl.get("voxel_map.plane.plane_fit_mode", "pca")
    f = cfg_get(log, "voxel_map/plane/plane_fit_mode", [FIT_RE])
    s, dt = judge([("plane_fit_mode", f, wfit)], have)
    out.append(("FITCFG %s plane_fit_mode %s" % (arm, wfit), s, dt))
    wpc = bool(dl.get("voxel_map.residual.pose_cov_in_sigma", False))
    items = [("pose_cov_in_sigma", cfg_get(log, "voxel_map/residual/pose_cov_in_sigma", [POSECOV_RE]), wpc)]
    if grp == "PC":
        wpt = dl.get("lio.point_cov.mode", "end_pose")
        items.append(("lio/point_cov/mode", cfg_get(log, "lio/point_cov/mode"), wpt))
        s, dt = judge(items, have)
        out.append(("PCCFG %s D8 cell: pose_cov_in_sigma %s, lio/point_cov/mode %s" % (arm, str(wpc).lower(), wpt), s, dt))
    else:
        s, dt = judge(items, have)
        out.append(("POSECFG %s pose_cov_in_sigma %s (default)" % (arm, str(wpc).lower()), s, dt))
    # ZUPT detect_only (D7)
    zl = line(et, "lio/zupt mode=")
    if dl.get("lio.zupt.mode"):
        z = kv(zl)
        sc = z.get("scans")
        items = [("mode", z.get("mode"), dl["lio.zupt.mode"]), ("gyro_thresh", z.get("gyro_thresh"), dl["lio.zupt.gyro_thresh"]), ("acc_thresh", z.get("acc_thresh"), dl["lio.zupt.acc_thresh"]),
                 ("enter_scans", z.get("enter_scans"), dl["lio.zupt.enter_scans"]), ("evaluated (scans > 0)", None if sc is None else ("yes" if int(sc) > 0 else "no"), "yes")]
        if dl["lio.zupt.mode"] == "detect_only":
            items.append(("applied (detect_only never touches the state)", z.get("applied"), 0))
        s, dt = judge(items, bool(zl))
        out.append(("ZDCFG %s lio/zupt mode %s as the engagement line states it" % (arm, dl["lio.zupt.mode"]), s, dt))
        if zl and z.get("stationary") is not None:
            out.append(("ZDINFO %s scans evaluated, stationary, applied (a fact)" % arm, "INFO", "scans %s stationary %s applied %s" % (sc, z.get("stationary"), z.get("applied"))))
    # two-sided NIS gate (F-476)
    gl = line(et, "imu/process_noise/gain mode=")
    if dl.get("imu.process_noise.gain.mode") == "nis_gate":
        z = kv(gl)
        sc = z.get("scans")
        items = [("mode", z.get("mode"), "nis_gate"), ("evaluated (scans > 0)", None if sc is None else ("yes" if int(sc) > 0 else "no"), "yes")]
        items += [("imu/process_noise/gain/" + k, cfg_get(log, "imu/process_noise/gain/" + k), v) for k, v in NG_WANT.items()]
        if gl:
            items.append(("summary line prints deflating_scans and min_gain_seen", "yes" if ("deflating_scans" in z and "min_gain_seen" in z) else "no", "yes"))
        s, dt = judge(items, bool(gl) or have)
        if not gl and have and et.strip():
            s, dt = "FAIL", "no imu/process_noise/gain engagement line; " + dt
        out.append(("NGCFG %s nis_gate: nis_hi 1.3, inflate 1.25, decay 0.95, nis_lo 0.75, deflate 0.8, min_gain 0.25, engaged by the gain line" % arm, s, dt))
        if gl and "deflating_scans" in z:
            out.append(("NGINFO %s engaged = deflating_scans > 0 (a fact: a gate that never deflates is a result, not a defect)" % arm, "INFO",
                        "engaged %s; deflating_scans %s inflating_scans %s min_gain_seen %s scans_with_gain_ne_1 %s" % ("yes" if int(z["deflating_scans"]) > 0 else "no", z["deflating_scans"], z.get("inflating_scans"), z.get("min_gain_seen"), z.get("scans_with_gain_ne_1"))))
    # step cap (D5)
    cl = line(et, "lio/ekf/step_cap mode=")
    if dl.get("lio.ekf.step_cap.mode"):
        z = kv(cl)
        it = z.get("iterations")
        items = [("mode", z.get("mode"), "hard"), ("cap_rot", z.get("cap_rot"), SC_WANT["cap_rot"]), ("cap_pos", z.get("cap_pos"), SC_WANT["cap_pos"]), ("evaluated (iterations > 0)", None if it is None else ("yes" if int(it) > 0 else "no"), "yes")]
        s, dt = judge(items, bool(cl))
        out.append(("SCCFG %s step cap hard 0.01 rad / 0.01 m as the engagement line states it" % arm, s, dt))
        if cl and z.get("bound") is not None:
            out.append(("SCINFO %s step-cap iterations and iterations where the cap bound (a fact)" % arm, "INFO", "iterations %s bound %s" % (it, z.get("bound"))))
    # coupled tail (TAIL-1 and its smoke arm)
    if dl.get("estimator.mode") == "coupled":
        el = line(et, "[engagement] estimator=")
        z = kv(el)
        wit = dl.get("lio.ekf.max_iterations")
        items = [("estimator", z.get("estimator"), "coupled_joint_knots"), ("residual_evaluation_time", z.get("residual_evaluation_time"), dl["estimator.coupled.residual_evaluation_time"]),
                 ("lio/ekf/max_iterations", cfg_get(log, "lio/ekf/max_iterations"), wit)]
        s, dt = judge(items, bool(el) or have)
        out.append(("TLCFG %s estimator coupled, residual_evaluation_time %s, max_iterations %s" % (arm, dl["estimator.coupled.residual_evaluation_time"], wit), s, dt))
    # calibration samples (D3)
    if "calib.stationary.num_samples" in dl:
        ns = dl["calib.stationary.num_samples"]
        s, dt = judge([("calib/stationary/num_samples", cfg_get(log, "calib/stationary/num_samples", [CALIB_RE]), ns)], have)
        out.append(("ECCFG %s calibration samples %d (point filter %s is the PFN row)" % (arm, ns, t["pfn"]), s, dt))
    # nothing the arm did not ask for
    extra = []
    if zl and not dl.get("lio.zupt.mode"):
        extra.append("ZUPT")
    if gl and dl.get("imu.process_noise.gain.mode") != "nis_gate":
        extra.append("process-noise gain")
    if cl and not dl.get("lio.ekf.step_cap.mode"):
        extra.append("step cap")
    out.append(("NOEXTRA %s no ZUPT, process-noise gain or step-cap engagement line that the arm did not ask for" % arm, "FAIL" if extra else ("PASS" if et.strip() else "MISSING"), ", ".join(extra)))
    return out


def rdist_gate(g, arm, d):
    """F-453 smoke: line_rows_per_scan has n_line_dist (last column) with sum > 0 and n_line_u0 + n_line_u1 == 0; the residual dump (if returned) has no row_kind 1 or 2 and some row_kind 6."""
    L = next((x for x in (csv(d, f) for f in LINE_ROWS_FILES) if x is not None), None)
    D = csv(d, "residual_dump_all.csv.gz")
    nm = "RDIST %s rows = distance engaged: n_line_dist sum > 0, n_line_u0 + n_line_u1 == 0, dump (if returned) has no row_kind 1 or 2 and some row_kind 6" % arm
    bad, miss, det = [], [], []
    if L is None:
        miss.append("line_rows_per_scan file absent")
    elif any(c not in L.columns for c in ("n_line_dist", "n_line_u0", "n_line_u1")):
        bad.append("line_rows_per_scan lacks n_line_dist / n_line_u0 / n_line_u1")
    else:
        nd, n01 = float(L["n_line_dist"].sum()), float(L["n_line_u0"].sum() + L["n_line_u1"].sum())
        det.append("n_line_dist %d, u0+u1 %d, n_line_dist is the last column: %s" % (nd, n01, L.columns[-1] == "n_line_dist"))
        if not nd > 0:
            bad.append("n_line_dist sum is 0")
        if n01 != 0:
            bad.append("u0+u1 rows %d, the distance row must replace them" % n01)
        if L.columns[-1] != "n_line_dist":
            bad.append("n_line_dist is not the last column")
    if D is not None:
        if "row_kind" not in D.columns:
            bad.append("dump has no row_kind")
        else:
            k = D["row_kind"].value_counts().to_dict()
            det.append("dump row kinds %s" % dict(sorted(k.items())))
            if k.get(1, 0) or k.get(2, 0):
                bad.append("dump has row_kind 1 or 2 rows")
            if not k.get(6, 0):
                bad.append("dump has no row_kind 6 row")
    else:
        det.append("dump not returned (not required)")
    g.add(nm, "FAIL" if bad else ("MISSING" if miss else "PASS"), "; ".join(bad + miss + det))


def run_gates(runs, refs_fb=None, phase=None, dispatched=None):
    g = G()
    A = lambda a: os.path.join(runs, a)
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    table = tmap()
    disp = dispatched_set(dispatched, table) if (dispatched is None or isinstance(dispatched, str)) else set(dispatched)
    live = {unit(t) for t in table if disp is None or t["id"] in disp}
    units = []
    for t in table:
        if unit(t) not in units:
            units.append(unit(t))
    for u in units:
        if u not in live:
            g.add("NOTRUN group %s queue %s: no arm of it was dispatched; no other gate is affected" % u, "INFO")
    id_rows = []
    sw = SwAgg(g)
    # ---- determinism (DET arm against the pfn 1 fixed-Q dump arm)
    du = next(unit(t) for t in table if t["id"] == M.DET_ARM)
    if du in live:
        a, b = digest(runs, M.DET_ARM), digest(runs, M.DET_ORIG)
        miss = [n for n, x in ((M.DET_ARM, a), (M.DET_ORIG, b)) if x is None]
        g.add("DET %s digest == %s" % (M.DET_ARM, M.DET_ORIG), "MISSING" if miss else ("PASS" if a == b else "FAIL"), ("digest file absent for: " + ", ".join(miss)) if miss else "%s vs %s" % (a[:16], b[:16]))
    for t in table:
        if unit(t) not in live:
            continue
        arm, grp, d = t["id"], t["group"], A(t["id"])
        isw = grp == "SW"
        pres = present(arm)
        sent = (jsn(d, "arm.json") or {}).get("sentinel") if pres else None
        ex = exit_status(d) if os.path.isdir(d) else None
        put = (lambda kind, name, st, det="": sw.put(kind, t["batch"], name, st, det)) if isw else (lambda kind, name, st, det="": g.add(name, st, det))
        if grp == "REF":
            ref_gates(g, d, t, pres, sent, ex)
            continue
        nm = "RUN %s completed (arm.json sentinel DONE)" % arm
        put("RUN", nm, "MISSING" if not pres else ("PASS" if sent == "DONE" else "FAIL"), "" if not pres else "sentinel=%s" % sent)
        nm = "EXIT %s exit_status.txt exists and equals 0 (the node's real exit status; first line)" % arm
        put("EXIT", nm, "MISSING" if ex is None else ("PASS" if ex == 0 else "FAIL"), "exit_status.txt absent" if ex is None else "exit status %s" % ex)
        if not pres:
            continue
        log = rd(d, "run.log") or ""
        cg = cfg_gates(t, log, eng_text(d))
        if isw:
            sts = [s for n, s, dt in cg if s != "INFO"]
            bad = ["%s: %s" % (n.split(" ")[0], dt) for n, s, dt in cg if s in ("FAIL", "MISSING")]
            put("CFG", "SWCFG %s effective config: edge %s, max_layer 2, lines %s, rows %s, point filter %s, plane fit, gap floor, nothing extra" % (arm, t["voxel"], t["mode"], "distance" if t["mode"] == "hardabs" else "per_normal", t["pfn"]), worst(sts), "; ".join(bad))
            continue
        for n, s, dt in cg:
            g.add(n, s, dt)
        # ---- pos trace, identity, length
        P = csv(d, "pos_trace_all.csv.gz")
        if t["kind"] == "id":
            r = ident_row(g, arm, t["ref"], P, refs_fb, M.DX_LAST_EXCLUDED if grp == "DP" else 0)
            if r:
                id_rows.append(r)
        if grp in W60:
            g.add("DPLEN %s first 60 s: between 560 and 700 scans (about 595 expected)" % arm, "PASS" if P is not None and 560 <= len(P) <= 700 else ("MISSING" if P is None else "FAIL"), "rows %s" % (None if P is None else len(P)))
        elif grp == "EC3":
            g.add("ECLEN %s early arm: at least 5 scans in the pos trace (the manifest's min_rows)" % arm, "PASS" if P is not None and len(P) >= 5 else ("MISSING" if P is None else "FAIL"), "rows %s" % (None if P is None else len(P)))
        else:
            n = M.N_SCANS[t["bag"]]
            g.add("LEN %s whole bag: at least %d percent of the %d scans of %s" % (arm, int(LEN_MIN * 100), n, t["bag"]), "PASS" if P is not None and len(P) >= LEN_MIN * n else ("MISSING" if P is None else "FAIL"), "rows %s" % (None if P is None else len(P)))
        if t["win"]:
            window_gates(g, arm, d, t)
        if t["msglog"]:
            msglog_gates(g, arm, d, t, P)
        if arm == SMK_DIST:
            rdist_gate(g, arm, d)
        if grp == "ZD":
            zupt_file_gate(g, arm, d)
        if grp == "SC":
            step_cap_file_gate(g, arm, d)
    sw.flush()
    # ---- D1 sweep completeness: summaries only (the sweep arms return no pos_trace unless gt_max_mm > 500; never required here)
    P = phase if isinstance(phase, pd.DataFrame) else (pd.read_csv(phase) if phase and os.path.exists(phase) else None)
    have = set(P["arm"]) if (P is not None and "arm" in P.columns) else None
    for u in units:
        if u[0] != "SW" or u not in live:
            continue
        cell = u[1]
        arms_c = [t["id"] for t in table if unit(t) == u]
        if have is None:
            g.add("SWCOMP %s phase_summary.csv rows of the %d arms of the cell" % (cell, len(arms_c)), "MISSING", "phase_summary.csv absent or without an arm column")
            continue
        miss = [a for a in arms_c if a not in have]
        g.add("SWCOMP %s arms with a phase_summary row (a fact)" % cell, "INFO", "%d of %d" % (len(arms_c) - len(miss), len(arms_c)))
        for a in miss:
            g.add("SWCOMP %s has a phase_summary row" % a, "MISSING")
        if not miss:
            g.add("SWCOMP %s all %d arms have a phase_summary row" % (cell, len(arms_c)), "PASS")
    return g, id_rows


def zupt_file_gate(g, arm, d):
    """D7: zupt_all.csv.gz present, the writer's columns, at least 100 rows (the manifest's min_rows), applied never 1 (detect_only never touches the state)."""
    Z = csv(d, "zupt_all.csv.gz")
    nm = "ZDFILE %s zupt_all.csv.gz with columns %s, at least 100 rows, applied always 0" % (arm, ", ".join(ZD_COLS))
    if Z is None:
        g.add(nm, "MISSING"); return
    miss = [c for c in ZD_COLS if c not in Z.columns]
    if miss:
        g.add(nm, "FAIL", "missing columns %s" % miss); return
    napp = int((pd.to_numeric(Z["applied"], errors="coerce") != 0).sum())
    ok = len(Z) >= 100 and napp == 0
    g.add(nm, "PASS" if ok else "FAIL", "rows %d, rows with applied != 0: %d" % (len(Z), napp))
    st = pd.to_numeric(Z["stationary"], errors="coerce")
    g.add("ZDFACT %s stationary scans (a fact)" % arm, "INFO", "%d of %d (%.1f percent)" % (int((st == 1).sum()), len(Z), 100.0 * float((st == 1).mean())))


def step_cap_file_gate(g, arm, d):
    S = csv(d, "step_cap.csv.gz")
    nm = "SCFILE %s step_cap.csv.gz with columns %s and at least 20 rows" % (arm, ", ".join(SC_COLS))
    if S is None:
        g.add(nm, "MISSING"); return
    miss = [c for c in SC_COLS if c not in S.columns]
    if miss:
        g.add(nm, "FAIL", "missing columns %s" % miss); return
    g.add(nm, "PASS" if len(S) >= 20 else "FAIL", "rows %d" % len(S))
    g.add("SCFACT %s iterations where the cap bound (a fact)" % arm, "INFO", "%d of %d" % (int((pd.to_numeric(S["bound"], errors="coerce") == 1).sum()), len(S)))


def window_gates(g, arm, d, t):
    ST, IT = csv(d, "window_state_all.csv.gz"), csv(d, "window_iter_all.csv.gz")
    UD, FS = csv(d, "window_update_diag_all.csv.gz"), csv(d, "window_frame_stats_all.csv.gz")
    DS, SM, GR = csv(d, "window_dump_state_all.csv.gz"), csv(d, "window_row_summary_all.csv"), csv(d, "window_ground_rows_all.csv.gz")
    need = {"window_state_all.csv.gz": (ST, M.STATE_COLS), "window_iter_all.csv.gz": (IT, M.ITER_COLS), "window_update_diag_all.csv.gz": (UD, M.UPD_COLS), "window_frame_stats_all.csv.gz": (FS, M.FS_COLS),
            "window_dump_state_all.csv.gz": (DS, M.DSTATE_COLS), "window_row_summary_all.csv": (SM, M.SUM_COLS), "window_ground_rows_all.csv.gz": (GR, M.GROUND_COLS)}
    bad = [n for n, (df, cols) in need.items() if df is None or any(c not in df.columns for c in cols)]
    if any(df is None for df, _ in need.values()):
        g.add("WINFILES %s the seven window files with their columns" % arm, "MISSING", "absent or lacking columns: %s" % bad); return
    g.add("WINFILES %s the seven window files with their columns" % arm, "PASS" if not bad else "FAIL", "problems: %s" % bad)
    cov = lambda df, key: sorted(set(int(x) for x in df[key].unique()))
    ok = cov(IT, "scan_id") == FRAMES and cov(UD, "frame") == FRAMES and cov(FS, "frame_idx") == FRAMES and cov(DS, "frame") == FRAMES and cov(SM, "frame") == FRAMES
    g.add("WINFR %s per-iteration, update-diag, frame-stats, dump-state and summary files cover every frame %d to %d" % (arm, M.LO, M.HI), "PASS" if ok else "FAIL",
          "iter %s..%s (%d frames), update %d, stats %d, dump state %d, summary %d" % (IT.scan_id.min(), IT.scan_id.max(), IT.scan_id.nunique(), UD.frame.nunique(), FS.frame_idx.nunique(), DS.frame.nunique(), SM.frame.nunique()))
    # state window: both phases, scans lo-1 .. hi+1
    ok = set(ST["phase"]) == {"post_imu", "post_lio"} and int(ST.scan_id.min()) <= M.LO - 1 and int(ST.scan_id.max()) >= M.HI + 1
    g.add("WINST %s state window has both phases and scans %d..%d" % (arm, M.LO - 1, M.HI + 1), "PASS" if ok else "FAIL", "phases %s scans %s..%s" % (sorted(set(ST['phase'])), ST.scan_id.min(), ST.scan_id.max()))
    # frame mapping: final iteration's position after the update == post_lio position of scan (frame + offset)
    last = IT.sort_values(["scan_id", "iter"]).groupby("scan_id").tail(1).set_index("scan_id")
    lio = ST[ST["phase"] == "post_lio"].drop_duplicates("scan_id").set_index("scan_id")
    res = {}
    for off in (-1, 0, 1):
        common = [f for f in FRAMES if (f + off) in lio.index and f in last.index]
        if len(common) == len(FRAMES):
            res[off] = float(max(abs(last.loc[f, "p_after_x"] - lio.loc[f + off, "px"]) for f in common))
    good = [o for o, v in res.items() if v < 1e-6]
    g.add("FRAMEMAP %s the posterior position after the last iteration of frame f equals state_trace post_lio at scan f + offset (1e-6 m on all %d frames)" % (arm, len(FRAMES)), "PASS" if good else "FAIL", "max differences by offset %s" % {k: "%.3g" % v for k, v in res.items()})
    if good:
        g.add("FRAMEMAP %s offset between the update-log frame number and the state_trace scan_id (a fact)" % arm, "INFO", "offset %+d" % good[0])
    # dump: iteration 0 block present for every frame; ground rows for every ground frame
    cnt = SM.groupby(["frame", "iter"])["n_rows"].sum()
    have0 = all(((f, 0) in cnt.index and cnt[(f, 0)] > 0) for f in FRAMES)
    g.add("DUMP %s the residual dump holds rows at iteration 0 for every frame %d to %d" % (arm, M.LO, M.HI), "PASS" if have0 else "FAIL", "rows at iter 0 by frame %s" % {f: int(cnt.get((f, 0), 0)) for f in FRAMES})
    grows = GR.groupby("frame").size()
    ok = all(grows.get(f, 0) >= 20 for f in GFRAMES) and set(GR.frame.unique()) <= set(GFRAMES) and bool((GR["hpos_z"].abs() >= W.NZ_GROUND).all())
    g.add("GROUND %s at least 20 ground rows for every frame %d to %d and only those frames, all with |normal z| >= %.1f" % (arm, M.GLO, M.GHI, W.NZ_GROUND), "PASS" if ok else "FAIL", "rows by frame %s" % {f: int(grows.get(f, 0)) for f in GFRAMES})
    ok = bool((GR["s_prior_pose"] >= -1e-12).all()) and bool((GR["sigma_squared"] > 0).all()) and bool(np.isfinite(GR[["r", "sigma_squared", "plane_var_term", "floor_term", "sigma_diag_squared", "s_prior_pose", "sigma_gate_squared"]].to_numpy()).all())
    g.add("ROWSANE %s ground rows: finite, sigma^2 > 0, prior-pose term >= 0 (a quadratic form H P H^T)" % arm, "PASS" if ok else "FAIL", "min s_prior_pose %.3g, min sigma^2 %.3g" % (GR["s_prior_pose"].min(), GR["sigma_squared"].min()))
    err = (GR["sigma_squared"] - (GR["floor_term"] + GR["sigma_diag_squared"] + GR["plane_var_term"])).abs() / GR["sigma_squared"]
    g.add("SIGMADEC %s sigma^2 = floor + sensor part + plane variance on the ground rows (a fact: the max relative error; a large value means the weight has a term the planner has not found)" % arm, "INFO", "max relative error %.3g, median %.3g" % (err.max(), err.median()))
    gs = SM[SM["class"] == "ground"]
    over = int(gs.n_r2_over_9sigma2.sum()); tot = int(gs.n_rows.sum())
    gate_over = int(SM.n_r2_over_9gate.sum())
    g.add("GATEOK %s every dumped row (ground and other, all frames, both iterations) satisfies r^2 <= 9 sigma_gate^2, the test the code applies (the dump reproduces the admission)" % arm, "PASS" if gate_over == 0 else "FAIL", "rows over the gate variance: %d of %d" % (gate_over, int(SM.n_rows.sum())))
    g.add("GATEFACT %s ground rows with r^2 > 9 sigma^2 where sigma^2 is the WEIGHT variance (a fact: such rows pass the gate only because sigma_gate^2 is wider than sigma^2)" % arm, "INFO", "%d of %d (%.2f percent)" % (over, tot, 100.0 * over / max(tot, 1)))
    ratio = (GR["sigma_gate_squared"] / GR["sigma_squared"])
    g.add("GATEWIDTH %s sigma_gate^2 / sigma^2 on the ground rows: median, p90, max (a fact; 1 means the gate used the weight variance, more means the pose covariance widened the gate)" % arm, "INFO", "%.3g, %.3g, %.3g" % (ratio.median(), ratio.quantile(.9), ratio.max()))
    rej = FS[["frame_idx", "n_residuals", "n_miss_coverage", "n_miss_mismatch"]].sort_values("frame_idx")
    g.add("MISSFACT %s rows used and misses by reason, frames 519 to 526 (a fact)" % arm, "INFO", "; ".join("%d: used %d, coverage %d, mismatch %d" % (r.frame_idx, r.n_residuals, r.n_miss_coverage, r.n_miss_mismatch) for r in rej.itertuples() if 519 <= r.frame_idx <= 526))


def msglog_gates(g, arm, d, t, P):
    L = csv(d, "lidar_msg_log_all.csv.gz")
    nm = "MSGLOG %s lidar_msg_log_all.csv.gz with the %d columns, at least 100 rows, msg_index 0..n-1 in order, verdicts in {0,1,2,3}" % (arm, len(M.MSG_COLS))
    if L is None:
        g.add(nm, "MISSING"); return
    miss = [c for c in M.MSG_COLS if c not in L.columns]
    if miss:
        g.add(nm, "FAIL", "missing columns %s" % miss); return
    n = len(L)
    ok = n >= (100 if t["group"] == "MG" else 100) and list(L["msg_index"].astype(int)) == list(range(n)) and set(L["verdict"].astype(int)) <= {0, 1, 2, 3}
    g.add(nm, "PASS" if ok else "FAIL", "rows %d, verdicts %s" % (n, sorted(set(L['verdict'].astype(int)))))
    v = L["verdict"].astype(int)
    ov = pd.to_numeric(L["overlap_s"], errors="coerce")
    drop_bad = int(((v == 1) & ~(ov > 0)).sum()); acc_bad = int(((v == 0) & (ov > 0)).sum())
    g.add("MSGRULE %s every pushLidar drop has overlap_s > 0 and no accepted message has (the rule the queue applies, checked row by row)" % arm, "PASS" if drop_bad == 0 and acc_bad == 0 else "FAIL",
          "drops without overlap %d, accepted with overlap %d" % (drop_bad, acc_bad))
    if P is not None:
        na = int((v == 0).sum())
        g.add("MSGSCANS %s accepted messages >= scans in the pos trace (a scan cannot exist without its message)" % arm, "PASS" if na >= len(P) else "FAIL", "accepted %d, scans %d" % (na, len(P)))
    num = lambda c: pd.to_numeric(L[c], errors="coerce")
    g.add("MSGFACT %s messages by verdict (accepted, dropped by pushLidar, dropped old, no points) (a fact)" % arm, "INFO", "%d, %d, %d, %d of %d" % (int((v == 0).sum()), int((v == 1).sum()), int((v == 2).sum()), int((v == 3).sum()), n))
    g.add("MSGFACT %s header to first point (off_first_s) and header to last point (off_last_s), seconds: median, min, max (a fact)" % arm, "INFO",
          "first %.4f %.4f %.4f; last %.4f %.4f %.4f" % (num("off_first_s").median(), num("off_first_s").min(), num("off_first_s").max(), num("off_last_s").median(), num("off_last_s").min(), num("off_last_s").max()))
    g.add("MSGFACT %s span (last minus first point) and header gap to the previous message, seconds: median, max (a fact)" % arm, "INFO",
          "span %.4f %.4f; gap %.4f %.4f" % (num("span_s").median(), num("span_s").max(), num("gap_header_s").median(), num("gap_header_s").max()))




# ------------------------------------------------------------------------------------------------ selftest
def fake_window(dest, bad=None):
    """the seven reduced window files of a DP arm (the reducer's own output on the reducer's own fake arm), with the one breakage `bad` if it is a window breakage."""
    tmp = tempfile.mkdtemp(); W.fake_arm(tmp)
    st = pd.read_csv(os.path.join(tmp, "state_trace.csv")); it = pd.read_csv(os.path.join(tmp, "iekf_factor_iteration.csv"))
    it["p_after_x"] = it["scan_id"] * 0.01; it["p_before_x"] = it["scan_id"] * 0.01 - 0.001      # consistent posterior: post_lio px(scan) == p_after_x(last iter of frame)
    it["v_before_z"] = 0.1; it["dtheta_norm"] = 0.0; it["b0"] = 0.0
    if bad == "badmap":
        it["p_after_x"] += 0.5
    st["px"] = st["scan_id"] * 0.01
    for c in ("py", "pz", "vx", "vy", "t_abs"):
        st[c] = 0.0
    st.to_csv(os.path.join(tmp, "state_trace.csv"), index=False); it.to_csv(os.path.join(tmp, "iekf_factor_iteration.csv"), index=False)
    for fn, cols, v in (("lio_update_diag.csv", M.UPD_COLS, 0.1), ("frame_stats.txt", M.FS_COLS, 1.0), ("residual_dump_state.csv", M.DSTATE_COLS, 0.0)):
        x = pd.read_csv(os.path.join(tmp, fn))
        for c in cols:
            if c not in x.columns:
                x[c] = v
        x.to_csv(os.path.join(tmp, fn), index=False)
    dd = pd.read_csv(os.path.join(tmp, "residual_dump.csv"))
    if bad == "noground":
        dd["hpos_z"] = 0.1
    if bad == "gatebad":
        dd["sigma_gate_squared"] = dd["sigma_squared"] * 0.5          # gate narrower than the weight: the dump no longer reproduces the admission
    if bad == "negprior":
        dd.loc[dd.index[::3], "s_prior_pose"] = -1.0
    dd.to_csv(os.path.join(tmp, "residual_dump.csv"), index=False)
    try:
        W.reduce_window(tmp, dest)
    except SystemExit:
        pass          # noground: the reducer itself refuses; the arm dir then lacks the files (MISSING)
    if bad == "nowin":
        os.remove(os.path.join(dest, "window_iter_all.csv.gz"))
    if bad == "winframes":
        x = pd.read_csv(os.path.join(dest, "window_update_diag_all.csv.gz")); x[x.frame != 520].to_csv(os.path.join(dest, "window_update_diag_all.csv.gz"), index=False, compression="gzip")
    shutil.rmtree(tmp, ignore_errors=True)


def mq_log(t, o):
    """the lines a build prints for ONE arm, in the formats the real code uses (see the source greps in the module docstring); o = the breakage switches that apply to this arm."""
    dl, grp = t["delta"], t["group"]
    en, form, rule = M.R9.WANT[t["mode"]]
    var = "plane_equivalent" if t["pe"] else "fit_only"
    rows = "distance" if dl.get(ROWS_KEY) == "distance" else "per_normal"
    rows = o.get("rows", rows); var = o.get("var", var)
    gap = o.get("gap", "true" if dl.get(M.R9.GAP) else "false")
    fit = o.get("fit", dl.get("voxel_map.plane.plane_fit_mode", "pca"))
    pose = o.get("pose", "true" if dl.get("voxel_map.residual.pose_cov_in_sigma") else "false")
    L = ["/livo_recon/voxel_map/plane/plane_gap_floor_en: True",          # a ROS dump line BEFORE the application line (the R78 false-FAIL)
         "[params/voxel_map]", "  map/voxel_size:                  %s" % o.get("vox", t["voxel"]), "  map/max_layer:                   %d" % o.get("layer", t["layer"]),
         "  plane/plane_fit_mode:            %s" % fit,
         "  plane/line_residual:             enabled %s / form %s / rule %s / rel_sigmas 3.0 / shape_ratio 0.1 / enum %d / variance %s / rows %s" % (en, form, rule, M.R9.ENUM[t["mode"]], var, rows),
         "  plane/plane_gap_floor_en:        %s" % gap, "  residual/pose_cov_in_sigma:  %s" % pose,
         "cbk/lidar/point_filter_num: %d" % o.get("pfn", t["pfn"]), "data_queues/log_lidar_msgs: %s" % o.get("msgkey", "true"),
         "[config/effective]  what this run actually does"]
    if grp == "PC":
        L.append("  lio/point_cov/mode = %s" % o.get("ptcov", dl.get("lio.point_cov.mode", "end_pose")))
    if dl.get("imu.process_noise.gain.mode") == "nis_gate":
        ng = dict(NG_WANT, **o.get("ng", {}))
        L += ["  imu/process_noise/gain/mode = nis_gate"] + ["  imu/process_noise/gain/%s = %s" % (k, v) for k, v in ng.items()]
    if dl.get("estimator.mode") == "coupled":
        L += ["  estimator/mode = coupled", "/livo_recon/lio/ekf/max_iterations: %d" % o.get("maxit", dl["lio.ekf.max_iterations"])]    # max_iterations has no echo of its own: the ROS dump line
    if "calib.stationary.num_samples" in dl:
        L += ["[params/calib/stationary]", "  required: true", "  num_samples: %d" % o.get("samples", dl["calib.stationary.num_samples"]), "  stall_calls_max: 5"]
    return "\n".join(L) + "\n"


def mq_eng(t, o, good_ng_deflating=True):
    dl = t["delta"]
    L = ["[engagement]  did each flag actually do anything?"]
    if dl.get("lio.zupt.mode"):
        L.append("  lio/zupt mode=%s gyro_thresh=0.02 acc_thresh=0.15 enter_scans=3 sigma_v=0.02 scans=590 stationary=%d applied=%d mean_vnorm_before(applied)=0 max_vnorm_before=0.01" % (o.get("zmode", dl["lio.zupt.mode"]), 40, o.get("zapplied", 0)))
    if dl.get("imu.process_noise.gain.mode") == "nis_gate" and not o.get("nogainline"):
        defl = 12 if good_ng_deflating else 0
        tail = "" if o.get("oldgain") else " inflating_scans=7 deflating_scans=%d min_gain_seen=%s" % (defl, "0.4" if defl else "1")
        L.append("  imu/process_noise/gain mode=nis_gate apply=both scans=590 scans_with_gain_ne_1=%d max_gain=1.9 mean_nis=1.1 mean_rho=0.1 mean_abs_lag1=0.2%s whiteness_blocked=0" % (19 if defl else 0, tail))
    if o.get("extra_gain"):
        L.append("  imu/process_noise/gain mode=nis_gate apply=both scans=590 scans_with_gain_ne_1=3 max_gain=1.9 mean_nis=1.1 mean_rho=0.1 mean_abs_lag1=0.2 inflating_scans=7 deflating_scans=1 min_gain_seen=0.9 whiteness_blocked=0")
    if dl.get("lio.ekf.step_cap.mode"):
        L.append("  lio/ekf/step_cap mode=hard cap_rot=%s rad cap_pos=0.01 m iterations=%d bound=%d mean_scale(bound)=0.5 min_scale=0.1 max_raw_rot=0.02 max_raw_pos=0.03" % (o.get("caprot", 0.01), 900, 4))
    if dl.get("estimator.mode") == "coupled":
        L.append("[engagement] estimator=coupled_joint_knots knot_count=4 residual_evaluation_time=%s gating_state_uncertainty=false iterations=3000" % o.get("ret", dl["estimator.coupled.residual_evaluation_time"]))
    return "\n".join(L) + "\n"


def cases(table):
    """breakage name -> (target arm, the row-name prefix that must FAIL / MISSING)."""
    first = lambda f: next(t["id"] for t in table if f(t))
    FIRSTDP = M.SMOKE_ARM
    c = dict(
        det=(M.DET_ARM, "DET r80_own_det"), arm_failed=(FIRSTDP, "RUN " + FIRSTDP), vox=(FIRSTDP, "VOX " + FIRSTDP), pfn=(FIRSTDP, "PFN " + FIRSTDP), msgkey=(FIRSTDP, "MSGKEY " + FIRSTDP),
        ident=(FIRSTDP, "IDENT " + FIRSTDP), badmap=(FIRSTDP, "FRAMEMAP " + FIRSTDP), nowin=(FIRSTDP, "WINFILES " + FIRSTDP), winframes=(FIRSTDP, "WINFR " + FIRSTDP), noground=(FIRSTDP, "WINFILES " + FIRSTDP),
        negprior=(FIRSTDP, "ROWSANE " + FIRSTDP), gatebad=(FIRSTDP, "GATEOK " + FIRSTDP), msgrule=(FIRSTDP, "MSGRULE " + FIRSTDP), msgcount=(FIRSTDP, "MSGSCANS " + FIRSTDP),
        msgorder=(FIRSTDP, "MSGLOG " + FIRSTDP), msgcol=(FIRSTDP, "MSGLOG " + FIRSTDP))
    B3T, ZDT, MGT, B1T = "r80_b3_eee_01_mq", "r80_zd_eee_01_fq", "r80_mg_eee_01_fq", "r80_b1_eee_01_fq"
    GPT, NGT, PCT, PCT2, DBT, TLT = "r80_gp_eee_01_pfn3_fq", "r80_ng_eee_01_fq", "r80_pc_ps_fq", "r80_pc_pp_fq", "r80_db_fq", "r80_tl_it10"
    ECT, SCT, SMD, SMT = "r80_ec_n300_pfn1_fq", "r80_sc_fq", SMK_DIST, "r80_sm_tl_60s"
    SWH = first(lambda t: t["group"] == "SW" and t["mode"] == "hardabs")
    SWO = first(lambda t: t["group"] == "SW" and t["mode"] == "off")
    R1, R2, R3 = [t["id"] for t in table if t["group"] == "REF"]
    c.update(
        ident_mq=(B3T, "IDENT " + B3T), ident_zd=(ZDT, "IDENT " + ZDT), ident_mg=(MGT, "IDENT " + MGT),
        zd_mode=(ZDT, "ZDCFG " + ZDT), zd_applied_eng=(ZDT, "ZDCFG " + ZDT), zd_file_applied=(ZDT, "ZDFILE " + ZDT), zd_file_cols=(ZDT, "ZDFILE " + ZDT), zd_file_none=(ZDT, "ZDFILE " + ZDT),
        sw_rows_hard=(SWH, "SWCFG " + SWH), sw_rows_off=(SWO, "SWCFG " + SWO), sw_layer=(SWH, "SWCFG " + SWH), sw_edge=(SWO, "SWCFG " + SWO), sw_pe=(SWH, "SWCFG " + SWH),
        sw_exit=(SWO, "EXIT " + SWO), sw_exit_absent=(SWO, "EXIT " + SWO), sw_run=(SWO, "RUN " + SWO), sw_nosummary=(SWH, "SWCOMP " + SWH),
        gp_off=(GPT, "GAPCFG " + GPT), gp_ctrl_on=(B1T, "GAPCFG " + B1T),
        ng_thr=(NGT, "NGCFG " + NGT), ng_noline=(NGT, "NGCFG " + NGT), ng_oldline=(NGT, "NGCFG " + NGT), ng_on_ctrl=(B3T, "NOEXTRA " + B3T),
        pc_cell=(PCT, "PCCFG " + PCT), pc_pt=(PCT2, "PCCFG " + PCT2), pose_ctrl=(B3T, "POSECFG " + B3T), db_mode=(DBT, "FITCFG " + DBT), fit_ctrl=(B3T, "FITCFG " + B3T),
        tl_iters=(TLT, "TLCFG " + TLT), tl_mode=(TLT, "TLCFG " + TLT), tl_smoke=(SMT, "TLCFG " + SMT), ec_samples=(ECT, "ECCFG " + ECT),
        sc_cap=(SCT, "SCCFG " + SCT), sc_nofile=(SCT, "SCFILE " + SCT), sc_cols=(SCT, "SCFILE " + SCT),
        rd_zero=(SMD, "RDIST " + SMD), rd_u0=(SMD, "RDIST " + SMD), rd_kind1=(SMD, "RDIST " + SMD), rd_nokind6=(SMD, "RDIST " + SMD), rd_col=(SMD, "RDIST " + SMD), rd_none=(SMD, "RDIST " + SMD),
        rd_rows_cfg=(SMD, "LINE " + SMD),
        ref_zero=(R1, "EXIT " + R1), ref_done=(R1, "EXIT " + R1), ref_needle=(R2, "REFUSE " + R2), ref_noexit=(R3, "EXIT " + R3),
        exit_nonzero=(MGT, "EXIT " + MGT), exit_absent=(B1T, "EXIT " + B1T), exit_bad=(B1T, "EXIT " + B1T), len_short=(B1T, "LEN " + B1T))
    return c


def mkfake(root, table, CS, bad=None, only=None, winpool=None):
    """one fake arm dir per manifest arm (or per arm of `only`); `bad` names ONE breakage, applied to its own arm only (CS[bad][0])."""
    os.makedirs(root, exist_ok=True)
    for t in table:
        arm = t["id"]
        if only is not None and arm not in only:
            continue
        hit = lambda b: bad == b and arm == CS[b][0]
        d = os.path.join(root, arm); os.makedirs(d, exist_ok=True)
        w = lambda name, text: open(os.path.join(d, name), "w").write(text)
        grp, dl = t["group"], t["delta"]
        if grp == "REF":
            w("arm.json", json.dumps({"sentinel": "DONE" if hit("ref_done") else "FAILED"}))
            w("run_log_tail.txt", "terminate called: " + ("something else" if hit("ref_needle") else " and ".join(t["needles"])) + "\n")
            if not hit("ref_noexit"):
                w("exit_status.txt", "0\n" if hit("ref_zero") else "1\n")
            continue
        w("arm.json", json.dumps({"sentinel": "FAILED" if (hit("arm_failed") or hit("sw_run")) else "DONE"}))
        if not (hit("exit_absent") or hit("sw_exit_absent")):
            w("exit_status.txt", "abc\n" if hit("exit_bad") else ("3\n" if (hit("exit_nonzero") or hit("sw_exit")) else "0\n"))
        w("state_trace_blank_runid.sha256", ("b" if hit("det") else "a") * 64 + "\n")
        o = {}
        for b, key, val in (("vox", "vox", 0.35), ("pfn", "pfn", 1 if t["pfn"] != 1 else 3), ("msgkey", "msgkey", "false"), ("sw_rows_hard", "rows", "per_normal"), ("sw_rows_off", "rows", "distance"),
                            ("rd_rows_cfg", "rows", "per_normal"), ("sw_layer", "layer", 3), ("sw_edge", "vox", 0.35), ("sw_pe", "var", "fit_only"), ("gp_off", "gap", "false"), ("gp_ctrl_on", "gap", "true"),
                            ("ng_thr", "ng", {"nis_hi": 1.5}), ("pc_cell", "pose", "false"), ("pose_ctrl", "pose", "true"), ("pc_pt", "ptcov", "end_pose"), ("db_mode", "fit", "pca"), ("fit_ctrl", "fit", "debiased"),
                            ("tl_iters", "maxit", 5), ("tl_smoke", "maxit", 5), ("tl_mode", "ret", "measurement"), ("ec_samples", "samples", 200), ("sc_cap", "caprot", 0.02),
                            ("zd_mode", "zmode", "apply"), ("zd_applied_eng", "zapplied", 5), ("ng_noline", "nogainline", True), ("ng_oldline", "oldgain", True), ("ng_on_ctrl", "extra_gain", True)):
            if hit(b):
                o[key] = val
        if hit("tl_mode"):
            o["ret"] = "measurement"
        w("run.log", mq_log(t, o))
        w("engagement.txt", mq_eng(t, o, good_ng_deflating=arm != "r80_ng_nya_03_mq"))
        if grp == "SW":
            continue                 # the sweep returns no pos_trace unless gt_max_mm > 500: none is written and none is required
        if grp in W60:
            n_scans = 595
        elif grp == "EC3":
            n_scans = 120
        else:
            n_scans = M.N_SCANS[t["bag"]]
        if hit("len_short"):
            n_scans = 100
        sid = np.arange(1, n_scans + 1)
        P = pd.DataFrame({"scan_id": sid, "px": np.cumsum(np.full(n_scans, 1e-3)), "py": 0.0, "pz": 0.0, "yaw_rad": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0})
        if t["ref"] and (hit("ident") or hit("ident_mq") or hit("ident_zd") or hit("ident_mg")):
            P["px"] += 5e-3
        P.to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        if t["win"]:
            if bad in ("badmap", "nowin", "winframes", "noground", "negprior", "gatebad") and hit(bad):
                fake_window(d, bad)
            else:
                for fn in os.listdir(winpool):
                    shutil.copy(os.path.join(winpool, fn), d)
        if t["msglog"]:
            n_msg = n_scans + 7
            L = pd.DataFrame({"msg_index": np.arange(n_msg), "seq": np.arange(n_msg), "header_stamp": 100 + np.arange(n_msg) * 0.1, "scan_time": 99.9 + np.arange(n_msg) * 0.1, "gap_header_s": 0.1,
                              "n_raw": 20000, "n_kept": 6600, "t_first": 99.9 + np.arange(n_msg) * 0.1, "t_last": 100.0 + np.arange(n_msg) * 0.1, "off_first_s": 0.0, "off_last_s": 0.0999, "span_s": 0.0999,
                              "prev_accepted_last_t": 99.8 + np.arange(n_msg) * 0.1, "overlap_s": -0.1, "verdict": 0})
            L.loc[L.index[:7], "verdict"] = 1; L.loc[L.index[:7], "overlap_s"] = 0.002        # 7 dropped messages: accepted = scans
            if hit("msgrule"):
                L.loc[L.index[3], "overlap_s"] = -0.01
            if hit("msgcount"):
                L.loc[L.index[10:20], "verdict"] = 1; L.loc[L.index[10:20], "overlap_s"] = 0.002
            if hit("msgorder"):
                L["msg_index"] = L["msg_index"].iloc[::-1].to_numpy()
            if hit("msgcol"):
                L = L.drop(columns=["off_last_s"])
            L.to_csv(os.path.join(d, "lidar_msg_log_all.csv.gz"), index=False, compression="gzip")
        if grp == "ZD" and not hit("zd_file_none"):
            n = 300
            Z = pd.DataFrame({"frame": np.arange(1, n + 1), "gyro_mean": 0.01, "acc_mean": 0.05, "run": np.arange(n) % 5, "stationary": (np.arange(n) % 3 == 0).astype(int), "applied": 0, "vnorm_before": 0.001})
            if hit("zd_file_applied"):
                Z.loc[5, "applied"] = 1
            if hit("zd_file_cols"):
                Z = Z.drop(columns=["run"])
            Z.to_csv(os.path.join(d, "zupt_all.csv.gz"), index=False, compression="gzip")
        if grp == "SC" and not hit("sc_nofile"):
            n = 400
            S = pd.DataFrame({"frame": np.arange(n) // 4, "iter": np.arange(n) % 4, "rot_raw": 0.005, "pos_raw": 0.005, "rot_applied": 0.005, "pos_applied": 0.005, "scale": 1.0, "bound": (np.arange(n) % 9 == 0).astype(int), "next_converged": 0})
            if hit("sc_cols"):
                S = S.drop(columns=["next_converged"])
            S.to_csv(os.path.join(d, "step_cap.csv.gz"), index=False, compression="gzip")
        if arm == SMK_DIST:
            nsc = 300
            Lr = pd.DataFrame({"scan_id": np.arange(nsc), "iter": 0, "n_rows": 1000, "n_points": 900, "n_plane_rows": 800, "n_line_u0": 0, "n_line_u1": 0, "n_smooth_u0": 0, "n_smooth_u1": 0, "n_smooth_u2": 0})
            for c_ in ("n_pts_with_line_rows", "sum_fit_var_line", "sum_tilt_var", "sum_sigma2_line", "sum_sigma2_plane", "gap_fits", "gap_d1", "gap_d2", "tilt_voxels", "tilt_floored_axes", "tilt_fallbacks"):
                Lr[c_] = 0
            Lr["n_line_dist"] = 0 if hit("rd_zero") else 60
            if hit("rd_u0"):
                Lr.loc[3, "n_line_u0"] = 4
            if hit("rd_col"):
                Lr = Lr.drop(columns=["n_line_dist"])
            if not hit("rd_none"):
                Lr.to_csv(os.path.join(d, "line_rows_per_scan_all.csv.gz"), index=False, compression="gzip")
            kinds = np.array([0] * 30 + [6] * 20)
            if hit("rd_kind1"):
                kinds[3] = 1
            if hit("rd_nokind6"):
                kinds[kinds == 6] = 0
            pd.DataFrame({"frame": 1, "row_kind": kinds}).to_csv(os.path.join(d, "residual_dump_all.csv.gz"), index=False, compression="gzip")


def fake_refs(root, table):
    """references: the exact traces the (good) arms wrote, under the R79 arm names; one block per R79 arm."""
    rows, seen = [], set()
    for new, old in M.IDREF.items():
        if old in seen or not os.path.exists(os.path.join(root, new, "pos_trace_all.csv.gz")):
            continue
        seen.add(old)
        Pn = pd.read_csv(os.path.join(root, new, "pos_trace_all.csv.gz"))[["scan_id", "px", "py", "pz", "yaw_rad"]]
        Pn.insert(0, "arm", old); rows.append(Pn)
    return pd.concat(rows) if rows else None


def phase_of(table, drop=()):
    return pd.DataFrame({"arm": [t["id"] for t in table if t["group"] != "REF" and t["id"] not in drop], "gt_ate_mm": 1.0})


def selftest():
    fails = []
    S = {}
    saved = M.EDGES
    M.EDGES = [0.2, 1.0]            # SAMPLED SWEEP: all 72 cells, 2 edges per cell (the fake dirs would otherwise be 1224); the full 17-edge manifest is exercised in part 2 below
    try:
        table = tmap()
        CS = cases(table)
        winpool = tempfile.mkdtemp(); fake_window(winpool)
        root = tempfile.mkdtemp()
        mkfake(root, table, CS, winpool=winpool)
        refs = fake_refs(root, table)
        ph = phase_of(table)
        g, _ = run_gates(root, refs, ph)
        bad_rows = [(n, s, dd) for n, s, dd in g.rows if s in ("FAIL", "MISSING")]
        if bad_rows:
            fails.append("good data: %d FAIL/MISSING, first %s" % (len(bad_rows), bad_rows[:3]))
        nrows = len(g.rows)
        for need in ("NGINFO r80_ng_eee_01_fq", "NGINFO r80_ng_nya_03_mq", "ZDINFO", "SCINFO", "ZDFACT", "SCFACT", "SWCELL CFG sw_eee_01_fq_hardabs_p3", "SWCOMP sw_eee_01_fq_hardabs_p3 all 2", "RDIST r80_sm_dist_60s", "EXIT r80_refuse_nis_lo_without_deflate"):
            if not any(need in n for n, s, _ in g.rows):
                fails.append("good data: no row named like %r" % need)
        ngi = [dd for n, s, dd in g.rows if n.startswith("NGINFO r80_ng_nya_03_mq")]
        if not ngi or not ngi[0].startswith("engaged no"):
            fails.append("good data: the NG arm with deflating_scans 0 must be an INFO row 'engaged no': %s" % ngi)
        if not [1 for n, s, _ in g.rows if n.startswith("NGINFO") and s == "INFO"]:
            fails.append("good data: NGINFO rows are not INFO")
        # ---- every single breakage FAILs (or is MISSING for an absence) on its own arm's gate, and on no other arm's row
        ok_names = []
        for bad, (arm, needle) in CS.items():
            r = tempfile.mkdtemp()
            shutil.rmtree(r); shutil.copytree(root, r, copy_function=os.link)     # hard links: the target arm dir is removed and rebuilt, the others are never opened for writing
            ph_b = phase_of(table, drop=(arm,) if bad == "sw_nosummary" else ())
            if bad != "sw_nosummary":
                shutil.rmtree(os.path.join(r, arm), ignore_errors=True)
                mkfake(r, table, CS, bad=bad, only={arm}, winpool=winpool)
            gg, _ = run_gates(r, refs, ph_b)
            hit = [(n, s) for n, s, _ in gg.rows if n.startswith(needle) and s in ("FAIL", "MISSING")]
            if not hit:
                fails.append("breakage %s did not trip a gate named %s" % (bad, needle))
            other = [(n[:90], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING") and arm not in n]
            if other:
                fails.append("breakage %s tripped rows that do not concern %s: %s" % (bad, arm, other[:2]))
            if bad.endswith(("_absent", "_none", "_noexit", "nofile", "nosummary", "_none")) and not all(s == "MISSING" for n, s in hit):
                fails.append("breakage %s (an absence) must be MISSING, got %s" % (bad, hit[:2]))
            if bad in ("exit_nonzero", "sw_exit", "ref_zero", "exit_bad", "ng_thr", "sc_cap", "zd_mode", "tl_iters") and not all(s == "FAIL" for n, s in hit):
                fails.append("breakage %s must be FAIL, got %s" % (bad, hit[:2]))
            shutil.rmtree(r, ignore_errors=True)
        # ---- a whole group absent: only its own rows are MISSING (and none FAIL); dispatched-away: NOTRUN INFO and nothing else
        gone = [t["id"] for t in table if t["group"] in ("NG", "ZD")] + [t["id"] for t in table if t["batch"] == "sw_eee_01_fq_off_p3"]
        r = tempfile.mkdtemp(); shutil.rmtree(r); shutil.copytree(root, r, copy_function=os.link)
        for a in gone:
            shutil.rmtree(os.path.join(r, a), ignore_errors=True)
        gg, _ = run_gates(r, refs, phase_of(table, drop=gone))
        stray = [(n[:90], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING") and not any(a in n or "sw_eee_01_fq_off_p3" in n for a in gone)]
        if stray or any(s == "FAIL" for n, s, _ in gg.rows) or not any(s == "MISSING" for n, s, _ in gg.rows):
            fails.append("absent groups NG, ZD and one sweep cell: rows outside them are not PASS/INFO, or no MISSING at all: %s" % stray[:2])
        gg, _ = run_gates(r, refs, phase_of(table, drop=gone), dispatched=set(t["id"] for t in table) - set(gone))
        left = [(n[:90], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING")]
        nr = [n for n, s, _ in gg.rows if n.startswith("NOTRUN")]
        if left or not nr or any(s != "INFO" for n, s, _ in gg.rows if n.startswith("NOTRUN")):
            fails.append("undispatched groups NG, ZD and a sweep cell: expected only NOTRUN INFO rows and no FAIL/MISSING, got %s NOTRUN rows, left %s" % (len(nr), left[:2]))
        shutil.rmtree(r, ignore_errors=True)
        # ---- an empty directory is all MISSING
        e = tempfile.mkdtemp()
        ge, _ = run_gates(e, None, None)
        if any(s != "MISSING" for n, s, _ in ge.rows):
            fails.append("empty dir: a row other than MISSING: %s" % [(n[:80], s) for n, s, _ in ge.rows if s != "MISSING"][:2])
        n_sampled = len([t for t in table if t["group"] == "SW"])
        shutil.rmtree(root, ignore_errors=True)
    finally:
        M.EDGES = saved
    # ---- part 2: the FULL manifest (1363 arms, the real 17 edges): sizes, an empty directory, one whole sweep cell with 17 of 17, and one with 16 of 17
    full = tmap()
    cnt = {}
    for t in full:
        cnt[t["group"]] = cnt.get(t["group"], 0) + 1
    if len(full) != 1363 or cnt.get("SW") != 1224 or len([u for u in {unit(t) for t in full} if u[0] == "SW"]) != 72 or len(M.EDGES) != 17:
        fails.append("full manifest shape: %d arms, groups %s, edges %d" % (len(full), cnt, len(M.EDGES)))
    e = tempfile.mkdtemp()
    ge, _ = run_gates(e, None, None)
    if any(s != "MISSING" for n, s, _ in ge.rows):
        fails.append("full manifest, empty dir: a row other than MISSING: %s" % [(n[:80], s) for n, s, _ in ge.rows if s != "MISSING"][:2])
    cell = "sw_eee_01_fq_hardabs_p3"
    ids = [t["id"] for t in full if t["batch"] == cell]
    CS2 = cases(full)
    r = tempfile.mkdtemp()
    mkfake(r, full, CS2, only=set(ids), winpool=winpool if os.path.isdir(winpool) else None)
    g2, _ = run_gates(r, None, pd.DataFrame({"arm": ids, "gt_ate_mm": 1.0}), dispatched=set(ids))
    if any(s in ("FAIL", "MISSING") for n, s, _ in g2.rows) or not any("SWCOMP %s all 17 arms" % cell in n and s == "PASS" for n, s, _ in g2.rows) or not any("SWCELL CFG %s: all 17" % cell in n and s == "PASS" for n, s, _ in g2.rows):
        fails.append("full cell of 17: %s" % [(n[:80], s) for n, s, _ in g2.rows if s in ("FAIL", "MISSING")][:3])
    g3, _ = run_gates(r, None, pd.DataFrame({"arm": ids[:-1], "gt_ate_mm": 1.0}), dispatched=set(ids))
    if [(n[-40:]) for n, s, _ in g3.rows if s in ("FAIL", "MISSING")] != [("SWCOMP %s has a phase_summary row" % ids[-1])[-40:]] or not any(n.startswith("SWCOMP %s arms with" % cell) and d == "16 of 17" for n, s, d in g3.rows):
        fails.append("cell with 16 of 17 phase_summary rows: %s" % [(n[:90], s) for n, s, _ in g3.rows if s in ("FAIL", "MISSING")][:3])
    for x in (e, r):
        shutil.rmtree(x, ignore_errors=True)
    shutil.rmtree(winpool, ignore_errors=True)
    print("SELFTEST r80_gates", "FAIL " + "; ".join(fails) if fails else
          "PASS (%d gate rows on good data (sweep sampled to 2 edges per cell: %d sweep arms), none FAIL or MISSING; %d single breakages each trip their own gate and no other arm's; absent group = its own MISSING only; "
          "undispatched group = NOTRUN INFO only; empty directory all MISSING; full manifest 1363 arms / 17 edges: empty dir all MISSING, a 17 of 17 cell PASS, a 16 of 17 cell one MISSING)" % (nrows, n_sampled, len(CS)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refs-fb", default=os.path.join(HERE, "..", "..", "r80_manifest", "identity_refs_fb.csv.gz"))
    ap.add_argument("--phase-summary", default=None, help="phase_summary.csv of the return (default: next to --out, else in --runs or its parent)")
    ap.add_argument("--dispatched", default=None, help="arm_table.tsv / a file of arm ids / a comma list of arm ids, groups or queue names that were dispatched (default: all arms of the manifest)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    refs_fb = pd.read_csv(a.refs_fb) if os.path.exists(a.refs_fb) else None
    ph = a.phase_summary
    if ph is None:
        cands = ([os.path.join(os.path.dirname(os.path.abspath(a.out)), "phase_summary.csv")] if a.out else []) + [os.path.join(a.runs, "phase_summary.csv"), os.path.join(a.runs, "..", "phase_summary.csv")]
        ph = next((c for c in cands if os.path.exists(c)), None)
    g, id_rows = run_gates(a.runs, refs_fb, ph, a.dispatched)
    txt = report(g)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    if a.csv:
        cols = ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "node_id_equal", "digest_equal_if_known"]
        pd.DataFrame(id_rows, columns=cols).to_csv(a.csv, index=False)
    return 1 if "FAIL" in [l.split()[0] for l in txt.splitlines() if l] else 0


if __name__ == "__main__":
    sys.exit(main())
