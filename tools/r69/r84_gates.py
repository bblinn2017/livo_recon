#!/usr/bin/env python3
"""R84 gates (planning agent's script, derived from r83_gates.py; selftest on SYNTHETIC fixtures only, planner sandbox, not evidence).

    python3 tools/r69/r84_gates.py --runs <runs_dir> --manifest r84_manifest [--grid-retime r80c_grid.json] [--grid-bag r84_manifest/grid_bag.json] \
        [--ref-return R82_RETURN] [--ref-runs <old archive dir>] [--phase-summary phase_summary.csv] [--ret-dir <return dir>] [--expect "T3=1296,L2R=846,CMP=216"] \
        [--dispatched <arm ids file | arm_table.tsv | comma list of arm ids / groups / families / queues>] [--out r84_gates.txt] [--csv r84_identity_vs_old.csv] [--hold-file HOLD_FROM_GATES.txt]
    python3 tools/r69/r84_gates.py --selftest

Row vocabulary: PASS, FAIL, MISSING (never a pass), INFO, NOTRUN. The expectation of every arm is read from the MANIFEST ONLY (arm_table.tsv + jobs_*.json): every effective config key found in the
params echo is compared with the job's override (a key the job does not set is compared with its compiled default), and every family row below takes its expected value from the job of the arm, never from a
per-arm constant in this file. The only constants are the SPEC group counts of EXPECT (override with --expect; OWN is a fact) and the four REMOVED keys of R84 patch 1.
Groups: T3 (1,296 Tier-3 arms, max_layer 0), L2R (846 layer-2 reference arms), CMP (216 comparison arms), OWN. T3 and L2R are SWEEP groups: summaries only; their PASS rows fold into CELL rows and they need no
pos_trace unless gt_max_mm > 500 in the phase summary (then TRACE). Rows of R84:
  T3ENG / L2RENG  params echo shows voxel_map/map/max_layer, voxel_map/map/voxel_size, plane_fit_mode (pca when the job has none) and line_residual/enabled (false when none) equal to the job; the job's voxel_size
                  equals the edge in the arm name (v0p35 = 0.35) and the arm_table columns edge / layer / fit / lines; deb+lines-on arms: classify_at_fit and debiased_line_equiv as the job states them (echo or engagement);
                  T3: l2_ref non-empty, l2_source in {R81,R82A,R83,R84}, an R84 l2_source needs the twin L2R arm (its control column = the T3 arm) whose job differs from the T3 job in max_layer only (output paths and
                  run id aside); L2R: control is a T3 arm of the manifest
  SWENG           job sample_window_s > 0: the engagement line `lio/point_cov sample_window_s=<v> nodes=5 point_evaluations=<n> clamped_points=<n> mean_trace_m2=<x> max_trace_m2=<x>` with v equal to the job (1e-9), nodes 5,
                  point_evaluations > 0, mean_trace_m2 finite and > 0, no `<-- INERT` / `<-- NOT ENGAGED` marker (clamped_points is INFO), and the params echo shows the key; key explicitly 0: no engagement line.
                  Key absent or 0: no sample_window_s= line anywhere (NOEXTRA).
  TBPNENG         job sets imu.undistort.time_based_process_noise: the echo equals it; no job key: the echo shows the default var_acc if it prints the key, else INFO
  REMOVED         refuse arms (the four removed keys of patch 1 and the two sample_window_s range arms): exit status non-zero and the run.log / stderr text contains every needle of the arm_table column needles; the
                  removed-key arms also say `removed` (case-insensitive; else INFO). Every other arm: none of lio.point_cov.mode / window_s, lio.time_cov.mode / sigma_s in its job, params echo or engagement text
  ZUPTGONE        no lio/zupt key in any job, params echo or engagement text
  SMOKE / L0SMOKE the 4 smoke arms: exit 0 and their specific check (sw: SWENG rows; layer-0 smokes: echoed max_layer and the whole key set of the job); a layer-0 smoke that did not finish with exit 0 is a FAIL row
                  L0SMOKE naming the Tier-3 cell family it blocks and the count of T3 arms in it (reported, never writes the hold file)
  CMPPAIR         INFO, PRESENT / ABSENT: the same-batch control of a CMP arm in the dispatched set, its R83 arm (column l2_ref) in the dispatched set or in a reference file
  CFG themes      OVLCFG, MSGKEY, PNCFG, SWCFG (sample_window_s), TBPNCFG, TRCCFG, D4CFG, DSCFG, PFN, VOX: effective config of every key against the job's override
  RTGRID / OVLCFG rt arms: the job's retime keys equal the grid of the bag and the policy is retime_trim; dc arms: no retime key, no overlap_policy key
  NOEXTRA, PFT, MSGLOG/MSGRULE/MSGGRID/MSGSCANS, LEN, EXIT, RUN, REFUSE, DET, DETPOS, IDENT (the identity hold of r84_identity_hold.py against the R82 return), FAMCOUNT, SWEEPCOMP
A FAIL of a DET, DETPOS or IDENT row writes --hold-file (HOLD_FROM_GATES.txt): the agent places it as HOLD.txt (r84_hold_check.sh). Every other FAIL (incl. L0SMOKE) is reported only. The gate never deletes or cancels anything.
"""
import argparse, csv, glob, gzip, io, json, os, re, shutil, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r84_identity_hold as IH  # noqa: E402

LEN_MIN = 0.97
REF_OFFSET = -0.1
N_SCANS = {"eee_01": 3987, "eee_02": 3210, "eee_03": 1814, "nya_01": 3950, "nya_02": 4287, "nya_03": 4094, "sbs_01": 3542, "sbs_02": 3732, "sbs_03": 3893}   # message / scan counts of the bags (make_r79_manifest.N_SCANS)
EXPECT = {"T3": (1296,), "L2R": (846,), "CMP": (216,)}   # SPEC table (planner, 2026-10-07); OWN is a fact, not checked
SWEEP = ("T3", "L2R")
L2_SOURCES = ("R81", "R82A", "R83", "R84")
MSG_COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict"]
MSG_COLS_C = MSG_COLS + ["retime_shift_s", "n_trimmed"]
SWEEP_RE = re.compile(r"^r84_(?P<ovl>rt|dc)_(?P<kind>t3|l2)_(?P<bag>[a-z]+_\d\d)_(?P<fit>pca|deb)_(?P<lines>off|on)_v(?P<edge>[0-9p]+)_(?P<q>fq|mq)$")
RETIME_KEYS = ("cbk.lidar.retime_t0_s", "cbk.lidar.retime_period_s", "cbk.lidar.retime_seq0", "cbk.lidar.retime_ref_offset_s")
K_SW, K_TBPN = "lio.point_cov.sample_window_s", "imu.undistort.time_based_process_noise"
K_MAXL, K_VOX, K_FIT, K_LINE = "voxel_map.map.max_layer", "voxel_map.map.voxel_size", "voxel_map.plane.plane_fit_mode", "voxel_map.plane.line_residual.enabled"
K_CLS, K_EQV, K_TRACE = "voxel_map.plane.line_residual.classify_at_fit", "voxel_map.plane.debiased_line_equiv", "voxel_map.plane.plane_fit_trace_frames"
REMOVED_KEYS = ("lio.point_cov.mode", "lio.point_cov.window_s", "lio.time_cov.mode", "lio.time_cov.sigma_s")      # removed by R84 patch 1 (F-521): refused at startup
REMOVED_RE = re.compile(r"lio[/.](point_cov[/.](mode|window_s)\b|time_cov\b)")
ZUPT_RE = re.compile(r"lio[/.]zupt")
SW_LINE = re.compile(r"lio/point_cov\s+sample_window_s=")
IGN_PAIR = ("outputs.path", "outputs.debug_log_dir", "eval.state_trace_run_id")
STRICT = re.compile(r"^(voxel_map\.map\.(voxel_size|max_layer)|cbk\.lidar\.(point_filter_num|overlap_policy|retime_.*)|imu\.process_noise\.model|imu\.process_noise\.motion\.(gyro|acc)_scale|imu\.ds\.mode|"
                    r"lio\.point_cov\.sample_window_s|imu\.undistort\.time_based_process_noise|"
                    r"voxel_map\.plane\.(plane_gap_floor_en|debiased_line_equiv|plane_fit_mode|plane_fit_reasons_en|plane_fit_trace_frames)|"
                    r"voxel_map\.plane\.line_residual\.(enabled|form|rule|rows|variance|classify_at_fit)|data_queues\.log_lidar_msgs)$")
THEMES = [("OVLCFG", re.compile(r"^cbk\.lidar\.(overlap_policy|retime_.*)$")), ("MSGKEY", re.compile(r"^data_queues\.log_lidar_msgs$")), ("SWCFG", re.compile(r"^lio\.point_cov\.sample_window_s$")),
          ("TBPNCFG", re.compile(r"^imu\.undistort\.time_based_process_noise$")), ("PNCFG", re.compile(r"^imu\.process_noise\.(model|motion\..*)$")),
          ("TRCCFG", re.compile(r"^voxel_map\.plane\.plane_fit_(reasons_en|trace_frames)$")),
          ("D4CFG", re.compile(r"^voxel_map\.plane\.(line_residual\.(classify_at_fit|enabled|rows|variance)|debiased_line_equiv|plane_gap_floor_en|plane_fit_mode)$")),
          ("DSCFG", re.compile(r"^imu\.ds\.mode$")), ("PFN", re.compile(r"^cbk\.lidar\.point_filter_num$")), ("VOX", re.compile(r"^voxel_map\.map\.(voxel_size|max_layer)$"))]
DEFAULTS = {"voxel_map.plane.line_residual.classify_at_fit": False, "voxel_map.plane.plane_gap_floor_en": False, "voxel_map.plane.debiased_line_equiv": "off", "cbk.lidar.overlap_policy": "discard",
            "voxel_map.plane.line_residual.enabled": False, "voxel_map.map.max_layer": 2, "voxel_map.plane.plane_fit_mode": "pca", "lio.point_cov.sample_window_s": 0.0,
            "imu.undistort.time_based_process_noise": "var_acc"}   # a key the job does not set is at its compiled default; a build that prints it must print the default
SKIP_KEYS = re.compile(r"^(outputs\.|eval\.|evo\.|vio\.|calib\.p0\.|state\.cov\.|diagnostics\.|estimator\.mode$|spline\.)")


# ------------------------------------------------------------------------------------------------ small helpers
def rd(d, n):
    p = os.path.join(d, n)
    if os.path.exists(p):
        return gzip.open(p, "rt", errors="replace").read() if p.endswith(".gz") else open(p, errors="replace").read()
    if os.path.exists(p + ".gz"):
        return gzip.open(p + ".gz", "rt", errors="replace").read()
    return None


def csvf(d, n):
    t = rd(d, n)
    try:
        return pd.read_csv(io.StringIO(t)) if t else None
    except Exception:
        return None


def jsn(d, n):
    p = os.path.join(d, n)
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except Exception:
        return None


def line(txt, tag):
    for l in (txt or "").splitlines():
        if tag in l:
            return l
    return None


def kv(ln):
    return dict(re.findall(r"([A-Za-z_0-9()]+)=(\S+)", ln or ""))


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class G:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))


def report(g):
    out = ["%-8s %s%s" % (s, n, ("   [" + d + "]") if d else "") for n, s, d in g.rows]
    out += ["", "PASS %d  FAIL %d  MISSING %d  INFO %d  NOTRUN %d" % tuple(sum(1 for r in g.rows if r[1] == k) for k in ("PASS", "FAIL", "MISSING", "INFO", "NOTRUN"))]
    return "\n".join(out)


def worst(sts):
    for s in ("FAIL", "MISSING"):
        if s in sts:
            return s
    return "PASS"


# ------------------------------------------------------------------------------------------------ effective configuration lookup
LINE_RE = re.compile(r"^\s*(/?[A-Za-z0-9_./\-]+?)\s*(=|:)\s*(\S+)(.*)$")


def parse_log(text):
    """-> {leaf: [(keypath, value, priority, tail)]}; priority 0 = `key = v` (the [config/effective] echo), 1 = `key: v` of the application's [params] dump, 2 = ROS dump `/ns/key: v`."""
    idx = {}
    for ln in (text or "").splitlines():
        m = LINE_RE.match(ln)
        if not m:
            continue
        kp, sep, val, tail = m.group(1), m.group(2), m.group(3).strip("'\""), m.group(4)
        pr = 2 if kp.startswith("/") else (0 if sep == "=" else 1)
        kp = kp.lstrip("/")
        if pr == 2 and "/" in kp:
            kp = kp.split("/", 1)[1]                 # drop the namespace (/livo_recon/)
        idx.setdefault(kp.split("/")[-1], []).append((kp, val, pr, tail))
    return idx


def lookup(idx, dotted):
    """-> list of (value, tail) the build printed for the key, best source first. A [params] line may print a suffix of the key (plane/plane_var_min for voxel_map/plane/plane_var_min)."""
    slash = dotted.replace(".", "/")
    out = []
    for kp, val, pr, tail in idx.get(slash.split("/")[-1], []):
        if kp == slash or slash.endswith("/" + kp) and kp.count("/") >= 1 or kp.endswith("/" + slash):
            out.append((pr, val, tail))
    return [(v, t) for _, v, t in sorted(out, key=lambda x: x[0])]


def sig_digits(s):
    d = re.sub(r"[eE][+-]?\d+$", "", s.lstrip("+-")).replace(".", "").lstrip("0")
    return max(len(d), 1)


def val_match(cand, want, key=""):
    c = str(cand).strip().strip("'\"")
    if isinstance(want, bool):
        return c.lower() in (("true", "1", "yes", "on") if want else ("false", "0", "no", "off"))
    if isinstance(want, int) and not isinstance(want, bool):
        v = num(c)
        return v is not None and v == want
    if isinstance(want, float):
        v = num(c)
        if v is None:
            return False
        tol = 1e-6 if key.endswith("retime_t0_s") else (1e-9 if key.endswith("retime_period_s") else 1e-9 * max(1.0, abs(want)))
        if abs(v - want) <= tol:
            return True
        k = sig_digits(c)
        return k >= 3 and num(format(want, ".%dg" % k)) == v
    return c.lower() == str(want).lower()


def judge_key(idx, key, want):
    """-> (state, detail): state PASS / FAIL (found, none matches) / NF (not printed)."""
    c = lookup(idx, key)
    if not c:
        return "NF", "%s not printed" % key
    if any(val_match(v, want, key) for v, _ in c):
        return "PASS", "%s = %s" % (key, c[0][0])
    return "FAIL", "%s: printed %s, wanted %s" % (key, "/".join(v for v, _ in c[:3]), want)


LINE2_RE = re.compile(r"plane/line_residual:\s+enabled (\w+) / form (\w+) / rule (\w+)(?: / rel_sigmas [0-9.eE+-]+)?(?: / shape_ratio [0-9.eE+-]+)?(?: / enum \d+)?(?: / variance (\w+))?(?: / rows (\w+))?")


def line_alt(log, key):
    """the one-line line-residual dump as a fallback for the per-key lines (cut by an interleaved [WARN] in R80)."""
    m = LINE2_RE.search(log or "")
    if not m:
        return None
    leaf = key.rsplit(".", 1)[1]
    return {"enabled": m.group(1), "form": m.group(2), "rule": m.group(3), "variance": m.group(4), "rows": m.group(5)}.get(leaf)


def exit_status(d):
    """-> (code, signal) from `exit_code=<N> signal=<NAME|none>` (or a bare integer); None if the file is absent; 'bad' if unreadable."""
    t = rd(d, "exit_status.txt")
    if t is None:
        return None
    ln = (t.splitlines() or [""])[0].strip()
    m = re.fullmatch(r"exit_code=(-?\d+)(?:\s+signal=(\S+))?", ln)
    if m:
        return int(m.group(1)), (m.group(2) or "none")
    return (int(ln), "none") if re.fullmatch(r"-?\d+", ln) else "bad"




# ------------------------------------------------------------------------------------------------ manifest
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_manifest(mdir):
    rows = list(csv.DictReader(open(os.path.join(mdir, "arm_table.tsv"), newline=""), delimiter="\t"))
    jobs = {}
    for fp in sorted(glob.glob(os.path.join(mdir, "jobs_*.json"))):
        for j in json.load(open(fp)):
            jobs[j["job_id"]] = j
    table = []
    for r in rows:
        j = jobs.get(r["arm"], {})
        dur = r.get("duration_s", "full")
        g_ = lambda k: (r.get(k) or "").strip()
        table.append(dict(id=r["arm"], group=r.get("group", ""), fam=r.get("fam", ""), mode=r.get("mode", ""), kind=r.get("arm_kind") or r.get("kind", ""), ref=r.get("ref") or None, bag=r.get("bag") or r.get("seq"),
                          batch=r.get("batch", ""), dur=None if dur in ("full", "", None) else float(dur), msglog=r.get("msglog") == "1", needles=[x for x in (r.get("needles") or "").split("|") if x],
                          fit=g_("fit"), lines=g_("lines"), edge=_f(g_("edge")), layer=_f(g_("layer")), sw_ms=_f(g_("sw_ms")), tbpn=g_("tbpn"), l2_ref=g_("l2_ref"), l2_source=g_("l2_source"), control=g_("control"),
                          sbc=g_("same_batch_control"), ov=j.get("overrides"), has_job=r["arm"] in jobs))
    return table


def dispatched_set(spec, table):
    if spec is None:
        return None
    toks = []
    if os.path.exists(spec):
        txt = open(spec).read().splitlines()
        if txt and "\t" in txt[0] and "arm" in txt[0].split("\t"):
            col = txt[0].split("\t").index("arm"); toks = [l.split("\t")[col] for l in txt[1:] if l.strip()]
        else:
            toks = [l.split()[0] for l in txt if l.strip() and not l.startswith("#")]
    else:
        toks = [x for x in spec.split(",") if x]
    ids = {t["id"] for t in table}; out = set()
    for tok in toks:
        if tok in ids:
            out.add(tok)
        for t in table:
            if tok in (t["group"], t["fam"], t["batch"]):
                out.add(t["id"])
    return out


def grid_of(bag, grid_retime, grid_bag):
    """the R80D rule, written here and not read from the manifest: eee_03 from the bag fit, the eight R80C bags exactly as shipped."""
    if bag == "eee_03":
        g = (grid_bag or {}).get("eee_03")
        return None if not g else dict(g["retime"], seq0=g["seq0"], n=g["n"])
    g = (grid_retime or {}).get(bag)
    return None if not g else dict(g, n=g["n"])


def cell_of(t):
    m = SWEEP_RE.match(t["id"])
    return "%s_%s_%s_%s" % (t["group"], t["mode"], t["bag"], m.group("q") if m else "x")


# ------------------------------------------------------------------------------------------------ per-arm gates (each returns a list of (name, status, detail))
def theme_of(key):
    for nm, rx in THEMES:
        if rx.match(key):
            return nm
    return "CFG"


def cfg_gates(t, ov, log):
    """effective-configuration rows of ONE arm from its own run.log text."""
    arm, out = t["id"], []
    have = bool(log.strip())
    idx = parse_log(log)
    want = {k: v for k, v in (ov or {}).items() if v is not None and not SKIP_KEYS.match(k) and isinstance(v, (bool, int, float, str))}
    for k, dv in DEFAULTS.items():
        want.setdefault(k, dv)
    by = {}
    for k, v in sorted(want.items()):
        st, det = judge_key(idx, k, v)
        if st == "NF" and k.startswith("voxel_map.plane.line_residual."):
            alt = line_alt(log, k)
            if alt is not None:
                st, det = ("PASS", "%s = %s (one-line dump)" % (k, alt)) if val_match(alt, v, k) else ("FAIL", "%s: one-line dump says %s, wanted %s" % (k, alt, v))
        explicit = k in (ov or {}) or k == "cbk.lidar.overlap_policy"
        if st == "NF":
            st = "MISSING" if (STRICT.match(k) and explicit) else "INFO"
        by.setdefault(theme_of(k), []).append((k, st, det))
    for th, items in by.items():
        sts = [s for _, s, _ in items if s != "INFO"]
        bad = ["%s: %s" % (k, d) for k, s, d in items if s in ("FAIL", "MISSING")]
        if not sts:
            if th in ("CFG",):
                continue
            out.append(("%s %s effective config (nothing printed to compare: facts only)" % (th, arm), "INFO", "; ".join(d for _, _, d in items)[:300]))
            continue
        if not have:
            out.append(("%s %s effective config" % (th, arm), "MISSING", "no run.log text"))
            continue
        out.append(("%s %s effective config: %d key(s) against the job's overrides and the defaults of the unset keys" % (th, arm, len(items)), worst(sts), "; ".join(bad) if bad else "; ".join(d for _, s, d in items if s == "PASS")[:300]))
    # overlap policy engagement
    pol = line(log, "[lidar_overlap_policy]")
    if t["mode"] == "rt":
        okp = bool(re.search(r"\[lidar_overlap_policy\]\s+(?:retime_trim\s+)?policy=retime_trim", log)) or bool(pol and "retime_trim" in pol)
        inert = bool(re.search(r"\[lidar_overlap_policy\].*inert when on", log))
        out.append(("OVLENG %s rt arm prints [lidar_overlap_policy] policy=retime_trim and no 'inert when on' error" % arm, "FAIL" if inert or (have and not okp) else ("PASS" if okp else "MISSING"), "line: %s" % (pol or "none")))
    elif t["mode"] == "dc":
        out.append(("OVLENG %s dc arm prints no [lidar_overlap_policy] line (the discard path prints what R80 printed)" % arm, "FAIL" if pol else ("PASS" if have else "MISSING"), "line: %s" % (pol or "none")))
    return out


def grid_gate(t, grid_retime, grid_bag):
    """rt: RTGRID, the job's retime keys equal the bag's grid (the R80D make() rule) and the policy is retime_trim with ref offset -0.1. dc: OVLCFG, no retime key and no overlap_policy key in the job."""
    arm, ov = t["id"], t["ov"] or {}
    if t["ov"] is None:
        return [("RTGRID %s job's overlap keys" % arm, "MISSING", "no job for the arm in jobs_*.json")]
    if t["mode"] == "dc":
        bad = [k for k in ov if k in RETIME_KEYS or k == "cbk.lidar.overlap_policy"]
        return [("OVLCFG %s overlap mode dc: the job has no retime key and no overlap_policy key" % arm, "FAIL" if bad else "PASS", ", ".join(bad))]
    nm = "RTGRID %s job's retime keys equal the grid of %s (t0_s, period_s, seq0 exactly, ref offset %.1f) and the policy is retime_trim" % (arm, t["bag"], REF_OFFSET)
    gr = grid_of(t["bag"], grid_retime, grid_bag)
    if gr is None:
        return [(nm, "MISSING", "no grid for %s (--grid-retime / --grid-bag)" % t["bag"])]
    bad = []
    for k, gv in (("cbk.lidar.retime_t0_s", gr["t0_s"]), ("cbk.lidar.retime_period_s", gr["period_s"]), ("cbk.lidar.retime_seq0", gr["seq0"]), ("cbk.lidar.retime_ref_offset_s", REF_OFFSET)):
        if k not in ov:
            bad.append("%s absent" % k)
        elif float(ov[k]) != float(gv):
            bad.append("%s = %r, grid %r" % (k, ov[k], gv))
    if ov.get("cbk.lidar.overlap_policy") != "retime_trim":
        bad.append("overlap_policy %r" % ov.get("cbk.lidar.overlap_policy"))
    return [(nm, "FAIL" if bad else "PASS", "; ".join(bad) if bad else "t0 %.7f period %.12f seq0 %s" % (gr["t0_s"], gr["period_s"], gr["seq0"]))]


def len_gate(g, t, P):
    arm = t["id"]
    n = None if P is None else len(P)
    if t["dur"] is None:
        want = N_SCANS.get(t["bag"])
        g.add("LEN %s whole bag: at least %d percent of the %s scans of %s" % (arm, int(LEN_MIN * 100), want, t["bag"]), "MISSING" if P is None else ("PASS" if want and n >= LEN_MIN * want else "FAIL"), "rows %s" % n)
    elif t["dur"] == 60:
        g.add("LEN %s first 60 s: between 560 and 700 scans (about 595 expected)" % arm, "MISSING" if P is None else ("PASS" if 560 <= n <= 700 else "FAIL"), "rows %s" % n)
    else:
        hi = 10 * t["dur"] + 20
        g.add("LEN %s short arm of %g s: 5 to %d scans in the pos trace" % (arm, t["dur"], hi), "MISSING" if P is None else ("PASS" if 5 <= n <= hi else "FAIL"), "rows %s" % n)


def msglog_gates(g, t, d, P, grid_retime, grid_bag):
    """BASE arms only. 15 columns (dc) or 17 (rt) ... the pushLidar rule row by row; for rt the grid, the shift, the row count, no verdict-1 rows under retime_trim."""
    arm = t["id"]
    L = csvf(d, "lidar_msg_log_all.csv.gz")
    need = MSG_COLS_C if t["mode"] == "rt" else MSG_COLS
    nm = "MSGLOG %s lidar_msg_log_all.csv.gz with %d columns, msg_index 0..n-1 in order, verdicts in {0,1,2,3}" % (arm, len(need))
    if L is None:
        g.add(nm, "MISSING"); return
    miss = [c for c in need if c not in L.columns]
    if miss:
        g.add(nm, "FAIL", "missing columns %s" % miss); return
    n = len(L)
    ok = n >= 100 and list(L["msg_index"].astype(int)) == list(range(n)) and set(L["verdict"].astype(int)) <= {0, 1, 2, 3}
    g.add(nm, "PASS" if ok else "FAIL", "rows %d, verdicts %s" % (n, sorted(set(L["verdict"].astype(int)))))
    v = L["verdict"].astype(int); ov_ = pd.to_numeric(L["overlap_s"], errors="coerce")
    drop_bad = int(((v == 1) & ~(ov_ > 0)).sum()); acc_bad = int(((v == 0) & (ov_ > 0)).sum())
    g.add("MSGRULE %s every pushLidar drop has overlap_s > 0 and no accepted message has (row by row)" % arm, "PASS" if drop_bad == 0 and acc_bad == 0 else "FAIL", "drops without overlap %d, accepted with overlap %d" % (drop_bad, acc_bad))
    if P is not None:
        na = int((v == 0).sum())
        if t["mode"] == "rt":
            g.add("MSGSCANS %s accepted messages >= scans in the pos trace" % arm, "PASS" if na >= len(P) else "FAIL", "accepted %d, scans %d" % (na, len(P)))
        else:   # R82 applied the rt rule to dc arms and FAILED the dc identity arm (accepted 3600, scans 3980, 387 verdict-1 rows): where the trace holds more scans than accepted messages is a fact, not a verdict
            g.add("MSGSCANS %s dc arm: accepted messages and scans in the pos trace (a fact; the rt rule accepted >= scans does not hold for the discard policy)" % arm, "INFO", "accepted %d, scans %d, verdict-1 rows %d" % (na, len(P), int((v == 1).sum())))
    if t["mode"] != "rt":
        return
    gr = grid_of(t["bag"], grid_retime, grid_bag)
    nm = "MSGGRID %s rt arm: rows = messages of the bag, scan_time on the grid (2e-6 s), retime_shift_s <= 1e-6, no verdict 1 row under retime_trim" % arm
    if gr is None:
        g.add(nm, "MISSING", "no grid for %s" % t["bag"]); return
    sh = pd.to_numeric(L["retime_shift_s"], errors="coerce").to_numpy(); sc = pd.to_numeric(L["scan_time"], errors="coerce").to_numpy(); sq = pd.to_numeric(L["seq"], errors="coerce").to_numpy()
    dev = float(np.nanmax(np.abs(sc - (gr["t0_s"] + (sq - gr["seq0"]) * gr["period_s"])))) if n else float("nan")
    bad = []
    if n != gr["n"]:
        bad.append("rows %d, bag has %d messages" % (n, gr["n"]))
    if not dev < 2e-6:
        bad.append("max |scan_time - grid| = %.3e s" % dev)
    if not np.nanmax(sh) <= 1e-6:
        bad.append("max shift %.9f s" % np.nanmax(sh))
    if int((v == 1).sum()):
        bad.append("verdict 1 rows %d" % int((v == 1).sum()))
    g.add(nm, "FAIL" if bad else "PASS", "; ".join(bad) if bad else "rows %d, max grid deviation %.2e s, max shift %.2e s" % (n, dev, float(np.nanmax(sh))))


def _eng_line(et, pat):
    return next((l for l in et.splitlines() if re.search(pat, l)), None)


def find_any(d, names):
    """first existing <d>/<name> or <d>/<name>.gz."""
    for n in names:
        for suf in ("", ".gz"):
            p = os.path.join(d, n + suf)
            if os.path.isfile(p):
                return p
    return None


def echo_row(idx, key, want, have_log):
    """-> (status, detail) of one params-echo comparison: PASS / FAIL / MISSING (not printed, or no log)."""
    if not have_log:
        return "MISSING", "no run.log text"
    st, det = judge_key(idx, key, want)
    return ("MISSING" if st == "NF" else st), det


def echo_or_alt(idx, log, key, want, have):
    st, det = echo_row(idx, key, want, have)
    if st == "MISSING" and have and key.startswith("voxel_map.plane.line_residual."):
        alt = line_alt(log, key)
        if alt is not None:
            return ("PASS", "%s = %s (one-line dump)" % (key, alt)) if val_match(alt, want, key) else ("FAIL", "%s: one-line dump says %s, wanted %s" % (key, alt, want))
    return st, det


def eng_text(d, log):
    return "\n".join([rd(d, "engagement.txt") or "", rd(d, "engagement_imu.txt") or "", log])


def sw_value(ov):
    v = (ov or {}).get(K_SW)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def sweep_static(t, by_id, ctl):
    """T3ENG / L2RENG rows that need the manifest only: the job against the arm name and the arm_table columns; the l2 reference and the twin pair."""
    arm, grp, ov = t["id"], t["group"], t["ov"]
    P = grp + "ENG"
    if ov is None:
        return [("%s %s manifest consistency" % (P, arm), "MISSING", "no job for the arm in jobs_*.json")]
    out, bad = [], []
    m = SWEEP_RE.match(arm)
    maxl, vox = ov.get(K_MAXL, DEFAULTS[K_MAXL]), ov.get(K_VOX)
    fit_j, on_j = ov.get(K_FIT, DEFAULTS[K_FIT]), bool(ov.get(K_LINE, DEFAULTS[K_LINE]))
    if not m:
        bad.append("arm name is not of the R84 sweep form")
    else:
        if m.group("kind") != {"T3": "t3", "L2R": "l2"}[grp]:
            bad.append("group %s but the name says %s" % (grp, m.group("kind")))
        if m.group("ovl") != t["mode"]:
            bad.append("mode column %s but the name says %s" % (t["mode"], m.group("ovl")))
        if vox is not None and abs(float(vox) - float(m.group("edge").replace("p", "."))) > 1e-9:
            bad.append("job voxel_size %r but the arm name says %s" % (vox, m.group("edge").replace("p", ".")))
        if fit_j != {"deb": "debiased", "pca": "pca"}[m.group("fit")]:
            bad.append("job plane_fit_mode %r but the arm name says %s" % (fit_j, m.group("fit")))
        if on_j != (m.group("lines") == "on"):
            bad.append("job line_residual/enabled %s but the arm name says %s" % (on_j, m.group("lines")))
    if vox is None:
        bad.append("the job has no voxel_size")
    elif t["edge"] is not None and abs(float(vox) - t["edge"]) > 1e-9:
        bad.append("job voxel_size %r but arm_table edge %g" % (vox, t["edge"]))
    if t["layer"] is not None and t["layer"] != maxl:
        bad.append("arm_table layer %g but the job max_layer is %r" % (t["layer"], maxl))
    if t["fit"] and fit_j != {"deb": "debiased", "pca": "pca"}.get(t["fit"], t["fit"]):
        bad.append("arm_table fit %s but the job plane_fit_mode is %r" % (t["fit"], fit_j))
    if t["lines"] and on_j != (t["lines"] == "on"):
        bad.append("arm_table lines %s but the job line_residual/enabled is %s" % (t["lines"], on_j))
    out.append(("%s %s manifest: the job's voxel_size, max_layer, plane_fit_mode and line_residual/enabled agree with the arm name and the arm_table columns edge, layer, fit, lines" % (P, arm), "FAIL" if bad else "PASS",
                "; ".join(bad) if bad else "edge %s, max_layer %s, fit %s, lines %s" % (vox, maxl, fit_j, on_j)))
    if grp == "T3":
        b2 = []
        if not t["l2_ref"]:
            b2.append("l2_ref empty")
        if t["l2_source"] not in L2_SOURCES:
            b2.append("l2_source %r not in %s" % (t["l2_source"], "/".join(L2_SOURCES)))
        out.append(("T3ENG %s l2_ref is non-empty and l2_source is one of %s" % (arm, ", ".join(L2_SOURCES)), "FAIL" if b2 else "PASS", "; ".join(b2) or "%s (%s)" % (t["l2_ref"], t["l2_source"])))
        if t["l2_source"] == "R84":
            twins = ctl.get(arm, [])
            b3 = []
            tw = by_id.get(t["l2_ref"])
            if not twins:
                b3.append("no L2R arm of the manifest has control = this arm")
            elif t["l2_ref"] not in twins:
                b3.append("l2_ref %s is not the L2R arm whose control is this arm (%s)" % (t["l2_ref"], ", ".join(twins)))
            if tw is None:
                b3.append("l2_ref %s is not in the manifest" % t["l2_ref"])
            elif tw["group"] != "L2R" or tw["ov"] is None:
                b3.append("l2_ref %s is not an L2R arm with a job" % t["l2_ref"])
            else:
                a_, b_ = dict(ov), dict(tw["ov"])
                a_.setdefault(K_MAXL, DEFAULTS[K_MAXL]); b_.setdefault(K_MAXL, DEFAULTS[K_MAXL])
                ab = object()
                diffs = sorted(k for k in (set(a_) | set(b_)) - set(IGN_PAIR) if a_.get(k, ab) != b_.get(k, ab))
                if diffs != [K_MAXL]:
                    b3.append("the jobs of the pair differ in %s (must be max_layer only)" % (", ".join(diffs) or "nothing"))
            out.append(("T3ENG %s twin L2R arm exists (its control column = this arm) and the pair's job overrides differ only in max_layer" % arm, "FAIL" if b3 else "PASS", "; ".join(b3) or t["l2_ref"]))
    else:
        c = by_id.get(t["control"])
        b4 = []
        if c is None or c["group"] != "T3":
            b4.append("control %r is not a T3 arm of the manifest" % t["control"])
        elif c["l2_source"] == "R84" and c["l2_ref"] != arm:
            b4.append("its T3 control names %s as l2_ref" % c["l2_ref"])
        out.append(("L2RENG %s control is the T3 arm of the same cell" % arm, "FAIL" if b4 else "PASS", "; ".join(b4) or t["control"]))
    return out


def sweep_echo(t, log, et):
    """T3ENG / L2RENG rows from the params echo of one present arm; the expectation is the job's value (pca / false / the code default 2 when the job has none)."""
    arm, grp, ov = t["id"], t["group"], t["ov"]
    P = grp + "ENG"
    have = bool(log.strip())
    idx = parse_log(log)
    maxl, vox = ov.get(K_MAXL, DEFAULTS[K_MAXL]), ov.get(K_VOX)
    fit_j, on_j = ov.get(K_FIT, DEFAULTS[K_FIT]), bool(ov.get(K_LINE, DEFAULTS[K_LINE]))
    parts = [(k, *echo_or_alt(idx, log, k, w, have)) for k, w in ((K_MAXL, maxl), (K_VOX, vox), (K_FIT, fit_j), (K_LINE, on_j)) if w is not None]
    out = [("%s %s params echo max_layer, voxel_size, plane_fit_mode and line_residual/enabled equal the job's" % (P, arm), worst([s for _, s, _ in parts]),
            "; ".join(d for _, s, d in parts if s != "PASS") or "; ".join(d for _, _, d in parts)[:240])]
    if fit_j == "debiased" and on_j:
        wc, we = ov.get(K_CLS), ov.get(K_EQV)
        nm = "%s %s deb + lines on: classify_at_fit and debiased_line_equiv as the job states them (echo, else engagement line)" % (P, arm)
        if wc is not True or we is None:
            out.append((nm, "FAIL", "the job lacks classify_at_fit true / debiased_line_equiv (classify %r, equiv %r)" % (wc, we)))
        else:
            ps = []
            for key, want, rx in ((K_CLS, True, r"classify_at_fit\s*=\s*(\w+)"), (K_EQV, we, r"debiased_line_equiv\s*=\s*(\w+)")):
                st, det = echo_row(idx, key, want, have)
                if st == "MISSING":
                    m_ = re.search(rx, et)
                    if m_:
                        st, det = ("PASS" if val_match(m_.group(1), want, key) else "FAIL"), "engagement line says %s, wanted %s" % (m_.group(1), want)
                ps.append((st, det))
            out.append((nm, worst([s for s, _ in ps]), "; ".join(d for _, d in ps)))
    return out


def sweng_rows(t, ov, log, et):
    """SWENG: engagement line and params echo of the sample window. Arms with the key absent print nothing and are judged by NOEXTRA."""
    arm, w = t["id"], sw_value(ov)
    have_et, have = bool(et.strip()), bool(log.strip())
    lines = [l for l in et.splitlines() if SW_LINE.search(l)]
    out = []
    if w is None:
        return out
    if w > 0:
        nm = "SWENG %s engagement line lio/point_cov sample_window_s=%g nodes=5 with point_evaluations > 0, mean_trace_m2 > 0 and no INERT / NOT ENGAGED marker" % (arm, w)
        if not lines:
            out.append((nm, "FAIL" if have_et else "MISSING", "no `lio/point_cov sample_window_s=` line in run.log / engagement_imu.txt" if have_et else "no engagement text"))
        else:
            z, bad = kv(lines[0]), []
            v = num(z.get("sample_window_s"))
            if v is None or abs(v - w) > 1e-9:
                bad.append("sample_window_s=%s, job %g" % (z.get("sample_window_s"), w))
            if num(z.get("nodes")) != 5:
                bad.append("nodes=%s (want 5)" % z.get("nodes"))
            pe, mt = num(z.get("point_evaluations")), num(z.get("mean_trace_m2"))
            if pe is None or pe <= 0:
                bad.append("point_evaluations=%s" % z.get("point_evaluations"))
            if mt is None or mt <= 0:
                bad.append("mean_trace_m2=%s (want finite and > 0)" % z.get("mean_trace_m2"))
            for mk in ("<-- INERT", "<-- NOT ENGAGED"):
                ml = next((l for l in lines if mk in l), None)
                if ml:
                    bad.append("marker %s: %s" % (mk, ml.split(mk, 1)[1].strip()[:80]))
            out.append((nm, "FAIL" if bad else "PASS", "; ".join(bad) or lines[0].strip()[:200]))
            out.append(("SWENG %s clamped_points (a fact)" % arm, "INFO", "clamped_points=%s" % z.get("clamped_points")))
        out.append(("SWENG %s params echo lio/point_cov/sample_window_s equals the job's" % arm, *echo_row(parse_log(log), K_SW, ov[K_SW], have)))
    else:
        out.append(("SWENG %s key explicitly 0: the engagement text has no sample_window_s= line" % arm, "FAIL" if lines else ("PASS" if have_et else "MISSING"), lines[0].strip()[:200] if lines else ""))
    return out


def tbpn_rows(t, ov, log):
    arm, v, have = t["id"], (ov or {}).get(K_TBPN), bool(log.strip())
    idx = parse_log(log)
    if v is not None:
        return [("TBPNENG %s params echo imu/undistort/time_based_process_noise equals the job's %s" % (arm, v), *echo_row(idx, K_TBPN, v, have))]
    if not have:
        return [("TBPNENG %s no job key: the default var_acc" % arm, "MISSING", "no run.log text")]
    c = lookup(idx, K_TBPN)
    if not c:
        return [("TBPNENG %s no job key: the echo does not print the key (default var_acc, a fact)" % arm, "INFO", "")]
    return [("TBPNENG %s no job key: the echo shows the default var_acc" % arm, "PASS" if val_match(c[0][0], "var_acc", K_TBPN) else "FAIL", "printed %s" % c[0][0])]


def cmp_static(t):
    """CMPCFG: the arm_table columns sw_ms and tbpn agree with the job (sample_window_s x 1000, time_based_process_noise)."""
    ov, bad = t["ov"], []
    if ov is None:
        return [("CMPCFG %s manifest" % t["id"], "MISSING", "no job")]
    w = sw_value(ov) or 0.0
    if t["sw_ms"] is None or abs(t["sw_ms"] - 1000.0 * w) > 1e-6:
        bad.append("arm_table sw_ms %s, job sample_window_s %g s" % (t["sw_ms"], w))
    if t["tbpn"] != str(ov.get(K_TBPN)):
        bad.append("arm_table tbpn %r, job %r" % (t["tbpn"], ov.get(K_TBPN)))
    return [("CMPCFG %s manifest: arm_table sw_ms equals 1000 x the job's sample_window_s and tbpn equals the job's time_based_process_noise" % t["id"], "FAIL" if bad else "PASS", "; ".join(bad) or "sw_ms %g, tbpn %s" % (t["sw_ms"], t["tbpn"]))]


def cmp_pair(t, live, ref_ids):
    sb, l2 = t["sbc"], t["l2_ref"]
    a = "PRESENT" if sb and sb in live else "ABSENT"
    b = "PRESENT" if l2 and (l2 in live or l2 in ref_ids) else "ABSENT"
    return [("CMPPAIR %s same-batch control %s and R83 arm %s in the dispatched set / reference file" % (t["id"], sb or "-", l2 or "-"), "INFO", "same-batch control %s; R83 arm %s" % (a, b))]


def removed_absent(t, ov, et, have):
    """REMOVED + ZUPTGONE for every arm that is not a refuse arm."""
    arm = t["id"]
    jk = [k for k in (ov or {}) if k in REMOVED_KEYS]
    hits = sorted({m.group(0) for m in REMOVED_RE.finditer(et)})
    out = [("REMOVED %s none of the four removed keys (lio/point_cov/mode, window_s, lio/time_cov/mode, sigma_s) in the job, the params echo or the engagement text" % arm, "FAIL" if (jk or hits) else ("PASS" if have else "MISSING"),
            ("job keys: %s; " % ", ".join(jk) if jk else "") + ("text: %s" % ", ".join(hits) if hits else ""))]
    zj = [k for k in (ov or {}) if ZUPT_RE.search(k)]
    zl = [l.strip() for l in et.splitlines() if ZUPT_RE.search(l)]
    out.append(("ZUPTGONE %s no lio/zupt key in the job, the params echo or the engagement text" % arm, "FAIL" if (zj or zl) else ("PASS" if have else "MISSING"), ("job keys: %s; " % ", ".join(zj) if zj else "") + "; ".join(zl[:2])[:200]))
    return out


def refuse_removed(t, d, ex, ov):
    """REMOVED rows of a refuse arm: exit non-zero and every needle in the refusal text; the removed-key arms also say `removed`."""
    arm = t["id"]
    tail = "".join(rd(d, f) or "" for f in ("run_log_tail.txt", "run.log", "stderr.txt", "stderr.log"))
    nm = "REMOVED %s refused: exit status non-zero and the run.log / stderr text contains %s" % (arm, " and ".join(repr(n) for n in t["needles"]) or "<no needle in the arm table>")
    if not t["needles"]:
        r = ("FAIL", "the arm table has no needles for the refuse arm")
    elif ex is None and not tail.strip():
        r = ("MISSING", "exit_status.txt and run.log / stderr text absent")
    elif ex == "bad":
        r = ("FAIL", "first line of exit_status.txt is not `exit_code=<N> signal=<NAME|none>`")
    elif ex is not None and ex[0] == 0:
        r = ("FAIL", "exit status 0: the build started with the refused configuration")
    elif not tail.strip():
        r = ("MISSING", "exit status %s but no run.log / stderr text" % (ex,))
    else:
        has = [n in tail for n in t["needles"]]
        r = ("PASS" if ex is not None and all(has) else ("MISSING" if ex is None else "FAIL"), "exit %s, needles found %s" % (ex[0] if ex else "unknown", has))
    out = [(nm, *r)]
    if any(k in REMOVED_KEYS for k in (ov or {})):
        nm2 = "REMOVED %s the refusal text says the key was removed (the word `removed`, case-insensitive; the exact text is the code's)" % arm
        out.append((nm2, "MISSING" if not tail.strip() else ("PASS" if re.search("removed", tail, re.I) else "INFO"), "" if tail.strip() else "no text"))
    return out


def smoke_row(t, ov, ex, cg, swr, log):
    """SMOKE: exit 0 and the arm's specific check (sw: the SWENG rows; layer 0: echoed max_layer and every key of the job)."""
    parts = []
    if ex is None:
        parts.append(("MISSING", "exit_status.txt absent"))
    elif ex == "bad":
        parts.append(("FAIL", "exit_status.txt unreadable"))
    else:
        parts.append(("PASS" if ex[0] == 0 else "FAIL", "exit_code %s" % ex[0]))
    w = sw_value(ov)
    if w and w > 0:
        sts = [s for n, s, _ in swr if s != "INFO"]
        parts.append((worst(sts) if sts else "MISSING", "SWENG rows: %s" % ("/".join(sts) or "none")))
    if ov.get(K_MAXL) == 0:
        st, det = echo_row(parse_log(log), K_MAXL, 0, bool(log.strip()))
        parts.append((st, det))
        bad = ["%s %s" % (n.split(" ")[0], s) for n, s, _ in cg if s in ("FAIL", "MISSING")]
        parts.append(("FAIL" if any(s == "FAIL" for n, s, _ in cg) else ("MISSING" if bad else "PASS"), "job key set echoed: %s" % ("; ".join(bad) or "all keys")))
    return ("SMOKE %s exit_code 0 and the arm's specific check (%s)" % (t["id"], "SWENG rows" if (w and w > 0) else "layer-0 echo and the job's key set"), worst([s for s, _ in parts]), "; ".join(d for _, d in parts))


def l0smoke_rows(table, live, runs):
    """L0SMOKE: a layer-0 smoke arm that did not finish with exit 0 blocks its Tier-3 cell family (fit, lines). Reported; it does not write the hold file."""
    sm = [t for t in table if t["kind"] == "smoke" and (t["ov"] or {}).get(K_MAXL) == 0]
    if not sm:
        return []
    out = []
    t3 = [x for x in table if x["group"] == "T3"]
    for t in sm:
        if t["id"] not in live:
            out.append(("L0SMOKE %s not dispatched" % t["id"], "NOTRUN", ""))
            continue
        ov = t["ov"]
        fit = t["fit"] or ("deb" if ov.get(K_FIT) == "debiased" else "pca")
        lines = t["lines"] or ("on" if ov.get(K_LINE) else "off")
        d = os.path.join(runs, t["id"])
        ex = exit_status(d) if os.path.isdir(d) else None
        if ex is not None and ex != "bad" and ex[0] == 0:
            continue
        n = sum(1 for x in t3 if x["fit"] == fit and x["lines"] == lines)
        out.append(("L0SMOKE %s did not finish with exit 0: blocks the Tier 3 cells fit=%s lines=%s (%d T3 arms)" % (t["id"], fit, lines, n), "MISSING" if (ex is None and not os.path.isdir(d)) else "FAIL",
                    "exit status %s" % (ex,)))
    if not any(r[1] in ("FAIL", "MISSING") for r in out) and any(t["id"] in live for t in sm):
        out.append(("L0SMOKE all %d layer-0 smoke arms finished with exit 0 (the Tier 3 families are not blocked)" % len(sm), "PASS", ""))
    return out


def engagement_gates(t, ov, et, have):
    arm = t["id"]; ov = ov or {}; out = []
    smoke = t["kind"] == "smoke"
    vl = next((l for l in et.splitlines() if re.search(r"plane_var_min\s*=.*\brows\s*=", l)), None)
    dl = _eng_line(et, r"plane_var_diag_en rows=")
    cl = _eng_line(et, r"line_residual/classify_at_fit\s*=.*line_classified=")
    swl = _eng_line(et, SW_LINE.pattern)
    cls_on = bool(ov.get(K_CLS))
    eq = ov.get(K_EQV)
    fl_deb = bool(ov.get("voxel_map.plane.plane_gap_floor_en")) and ov.get(K_FIT) == "debiased"
    if cls_on or eq or fl_deb:
        z = kv(cl or ""); bad = []
        if cl is None:
            bad.append("no D4 engagement line although a D4 key is on")
        else:
            if cls_on and "classify_at_fit = true" not in cl:
                bad.append("classify_at_fit not printed true")
            if eq and ("debiased_line_equiv = %s" % eq) not in cl:
                bad.append("debiased_line_equiv not printed %s" % eq)
            if "<-- INERT" in cl:
                bad.append("the build printed INERT")
            if smoke:
                num_ = lambda k: int(float(z.get(k, 0) or 0))
                if cls_on and num_("line_classified") <= 0:
                    bad.append("smoke arm: line_classified = %s" % z.get("line_classified"))
                if eq == "mean_cov" and num_("tilt_voxels_meancov") <= 0:
                    bad.append("smoke arm: tilt_voxels_meancov = %s" % z.get("tilt_voxels_meancov"))
                if eq == "moment" and num_("tilt_voxels_moment") <= 0:
                    bad.append("smoke arm: tilt_voxels_moment = %s" % z.get("tilt_voxels_moment"))
                if fl_deb and num_("gaps_floored") <= 0:
                    bad.append("smoke arm: gaps_floored = %s" % z.get("gaps_floored"))
        out.append(("D4ENG %s D4 engagement line (classify_at_fit, debiased_line_equiv, gap floor, counters%s)" % (arm, "; smoke arm: each engaged key shows a non-zero counter" if smoke else ""), "FAIL" if bad else ("MISSING" if cl is None and not have else "PASS"),
                    "; ".join(bad) or (cl or "").strip()[:300]))
    extra = []
    if vl and not ((ov.get("voxel_map.plane.plane_var_min") or 0) > 0 or (ov.get("voxel_map.plane.plane_var_scan_s2") or 0) > 0):
        extra.append("plane variance keys")
    if dl and not ov.get("voxel_map.plane.plane_var_diag_en"):
        extra.append("plane_var_diag")
    if cl and "classify_at_fit = true" in cl and not cls_on:
        extra.append("D4 classify_at_fit printed true although the job does not set it")
    w = sw_value(ov)
    if swl and not (w and w > 0):
        extra.append("sample_window_s= engagement line although the job's key is %s" % ("absent" if w is None else "0"))
    out.append(("NOEXTRA %s nothing engaged that the arm did not ask for (plane variance, plane_var_diag, D4 classify, sample_window_s engagement line)" % arm, "FAIL" if extra else ("PASS" if have else "MISSING"), ", ".join(extra)))
    return out


def ref_gates(g, t, d, present, sent, ex):
    arm = t["id"]
    nm = "REFUSE %s dies at startup, text contains %s" % (arm, " and ".join(repr(n) for n in t["needles"]))
    if not present:
        g.add(nm, "MISSING", "arm.json absent")
    else:
        tail = "".join(rd(d, f) or "" for f in ("run_log_tail.txt", "run.log"))
        has = [n in tail for n in t["needles"]]
        g.add(nm, "PASS" if sent == "FAILED" and all(has) else "FAIL", "sentinel=%s needles_found=%s" % (sent, has))
    nm = "EXIT %s exit_status.txt exists and is non-zero, and the job did not reach DONE (the deliberately refused job)" % arm
    if ex is None:
        g.add(nm, "MISSING", "exit_status.txt absent")
    elif ex == "bad":
        g.add(nm, "FAIL", "first line of exit_status.txt is not `exit_code=<N> signal=<NAME|none>`")
    elif not present:
        g.add(nm, "MISSING", "exit status %s but arm.json absent: cannot tell whether the job reached DONE" % (ex,))
    else:
        g.add(nm, "PASS" if ex[0] != 0 and sent != "DONE" else "FAIL", "exit status %s signal %s, sentinel %s" % (ex[0], ex[1], sent))


class SwAgg:
    """Sweep arms (T3, L2R): FAIL / MISSING rows stay one per arm; PASS and INFO rows of a (gate, cell) fold into one CELL row (the report stays inside its size ledger)."""

    def __init__(self, g):
        self.g, self.c = g, {}

    def put(self, kind, cell, name, st, det=""):
        if st not in ("PASS", "INFO"):
            self.g.add(name, st, det)
        self.c.setdefault((kind, cell), {"PASS": 0, "INFO": 0, "FAIL": 0, "MISSING": 0, "NOTRUN": 0})[st] += 1

    def flush(self):
        for (kind, cell), c in self.c.items():
            if c["FAIL"] == 0 and c["MISSING"] == 0:
                self.g.add("CELL %s %s: all %d checks PASS%s" % (kind, cell, c["PASS"] + c["INFO"], "" if not c["INFO"] else " or INFO (%d INFO: facts only)" % c["INFO"]), "PASS" if not c["INFO"] else "INFO")
            else:
                self.g.add("CELL %s %s: checks PASS / INFO / FAIL / MISSING (the FAIL and MISSING checks have their own rows)" % (kind, cell), "INFO", "%d / %d / %d / %d" % (c["PASS"], c["INFO"], c["FAIL"], c["MISSING"]))


def find_return_file(name, dirs):
    for d in dirs:
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


# ------------------------------------------------------------------------------------------------ the gate
def run_gates(runs, manifest, grid_retime=None, grid_bag=None, ref_returns=(), ref_runs=(), phase=None, dispatched=None, ret_dirs=(), expect=EXPECT, ref_manifests=(), tol_mm=IH.TOL_MM):
    g = G()
    table = load_manifest(manifest)
    A = lambda a: os.path.join(runs, a)
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    disp = dispatched_set(dispatched, table) if (dispatched is None or isinstance(dispatched, str)) else set(dispatched)
    live = {t["id"] for t in table if disp is None or t["id"] in disp}
    by_id = {t["id"]: t for t in table}
    ctl = {}
    for t in table:
        if t["group"] == "L2R" and t["control"]:
            ctl.setdefault(t["control"], []).append(t["id"])
    queues = []
    for t in table:
        if t["batch"] not in queues:
            queues.append(t["batch"])
    for q in queues:
        if not any(t["id"] in live for t in table if t["batch"] == q):
            g.add("NOTRUN queue %s: no arm of it was dispatched; no other gate is affected" % q, "NOTRUN")
    # ---- manifest counts against the SPEC table
    cnt = {}
    for t in table:
        cnt[t["group"]] = cnt.get(t["group"], 0) + 1
    if expect:
        for grp, want in expect.items():
            have = cnt.get(grp, 0)
            g.add("FAMCOUNT %s manifest holds %d arms (SPEC %s)" % (grp, have, " or ".join(str(x) for x in want)), "PASS" if have in want else "FAIL", "")
        extra = sorted(set(cnt) - set(expect) - {"OWN"})
        if extra:
            g.add("FAMCOUNT manifest has groups the SPEC table does not list", "FAIL", ", ".join(extra))
    g.add("FAMCOUNT OWN arms in the manifest (a fact; the planner reports the total)", "INFO", "OWN %d, all %d" % (cnt.get("OWN", 0), len(table)))
    Pn = phase if isinstance(phase, pd.DataFrame) else (pd.read_csv(phase) if phase and os.path.exists(phase) else None)
    gtmax = {}
    if Pn is not None and "arm" in Pn.columns and "gt_max_mm" in Pn.columns:
        gtmax = {a: num(v) for a, v in zip(Pn["arm"], Pn["gt_max_mm"])}
    ref_ids = set()
    for d_ in [manifest] + list(ref_returns):
        for fn in ("l2_reference_values.csv", "phase_summary.csv"):
            try:
                p_ = os.path.join(d_, fn)
                if os.path.exists(p_):
                    ref_ids |= set(pd.read_csv(p_, usecols=["arm"])["arm"])
            except Exception:
                pass
    id_rows, sw = [], SwAgg(g)
    P_cache = {}
    popen = lambda a: P_cache.setdefault(a, csvf(A(a), "pos_trace_all.csv.gz"))
    swcells = {}
    for t in table:
        if t["id"] not in live:
            continue
        arm, grp, d = t["id"], t["group"], A(t["id"])
        is_sw = grp in SWEEP
        cell = cell_of(t) if is_sw else None
        if is_sw:
            swcells.setdefault(cell, []).append(arm)

        def put(rows, fold=None):
            for n_, s_, dt_ in rows:
                if is_sw:
                    sw.put(fold or n_.split(" ")[0], cell, n_, s_, dt_)
                else:
                    g.add(n_, s_, dt_)
        pres = present(arm)
        sent = (jsn(d, "arm.json") or {}).get("sentinel") if pres else None
        ex = exit_status(d) if os.path.isdir(d) else None
        # manifest-only rows (no run needed)
        if is_sw:
            put(sweep_static(t, by_id, ctl))
        if grp == "CMP":
            put(cmp_static(t) + cmp_pair(t, live, ref_ids))
        if t["kind"] == "refuse":
            ref_gates(g, t, d, pres, sent, ex)
            put(refuse_removed(t, d, ex, t["ov"]))
            continue
        put([("RUN %s completed (arm.json sentinel DONE)" % arm, "MISSING" if not pres else ("PASS" if sent == "DONE" else "FAIL"), "" if not pres else "sentinel=%s" % sent)])
        nm = "EXIT %s exit_status.txt `exit_code=<N> signal=<NAME|none>` with N = 0" % arm
        if ex is None:
            put([(nm, "MISSING", "exit_status.txt absent")])
        elif ex == "bad":
            put([(nm, "FAIL", "first line is not `exit_code=<N> signal=<NAME|none>` nor an integer")])
        else:
            put([(nm, "PASS" if ex[0] == 0 else "FAIL", "exit status %s signal %s" % ex)])
        if not pres:
            if t["kind"] == "smoke":
                put([("SMOKE %s exit_code 0 and the arm's specific check" % arm, "MISSING", "arm.json absent")])
            continue
        log = rd(d, "run.log") or ""
        et = eng_text(d, log)
        have = bool(log.strip()) or bool(et.strip())
        ov = t["ov"]
        if ov is None:
            put([("CFG %s effective config" % arm, "MISSING", "no job for the arm in jobs_*.json")])
            continue
        cg = cfg_gates(t, ov, log)
        if is_sw:
            sts = [s for n, s, dt in cg if s != "INFO"]
            bad = ["%s: %s" % (n.split(" ")[0], dt) for n, s, dt in cg if s in ("FAIL", "MISSING")]
            put([("SWEEPCFG %s effective config: edge, layer, fit, lines, point filter, overlap policy, sample window, nothing extra" % arm, worst(sts) if sts else "MISSING", "; ".join(bad))])
        else:
            put(cg)
        put(grid_gate(t, grid_retime, grid_bag))
        if is_sw:
            put(sweep_echo(t, log, et))
        swr = sweng_rows(t, ov, log, et)
        put(swr)
        put(tbpn_rows(t, ov, log))
        put(removed_absent(t, ov, et, have))
        put(engagement_gates(t, ov, et, have))
        if t["kind"] == "smoke":
            put([smoke_row(t, ov, ex, cg, swr, log)])
        if is_sw:
            gm = gtmax.get(arm)
            if gm is not None and gm > 500:
                Pt = popen(arm)
                put([("TRACE %s collapsed (gt_max_mm %.0f > 500): pos_trace_all.csv.gz present with rows" % (arm, gm), "PASS" if Pt is not None and len(Pt) else "MISSING", "")])
            continue                    # T3 / L2R: summaries only; no pos_trace is required unless gt_max_mm > 500
        P = popen(arm)
        len_gate(g, t, P)
        if t["kind"] == "id":
            rows, _ = ident_rows(t, runs, ref_returns, ref_runs, manifest, ref_manifests, tol_mm)
            for n, s, dt in rows:
                g.add("IDENT %s %s" % (arm, n), s, dt)
            id_rows.append(ident_csv_row(t, runs, ref_returns, tol_mm))
        if t["kind"] == "det_fb":
            det_gates(g, t, runs, by_id, popen)
        if t["msglog"]:
            msglog_gates(g, t, d, P, grid_retime, grid_bag)
    sw.flush()
    for n_, s_, dt_ in l0smoke_rows(table, live, runs):
        g.add(n_, s_, dt_)
    # ---- sweep completeness: summaries only
    have = set(Pn["arm"]) if (Pn is not None and "arm" in Pn.columns) else None
    full = {}
    for t in table:
        if t["group"] in SWEEP:
            full[cell_of(t)] = full.get(cell_of(t), 0) + 1
    for cell, arms_ in sorted(swcells.items()):
        if have is None:
            g.add("SWEEPCOMP %s phase_summary.csv rows of the %d arms of the cell" % (cell, len(arms_)), "MISSING", "phase_summary.csv absent or without an arm column"); continue
        miss = [a for a in arms_ if a not in have]
        g.add("SWEEPCOMP %s arms with a phase_summary row (a fact)" % cell, "INFO", "%d of %d" % (len(arms_) - len(miss), len(arms_)))
        for a in miss:
            g.add("SWEEPCOMP %s has a phase_summary row" % a, "MISSING")
        if not miss and len(arms_) == full[cell]:
            g.add("SWEEPCOMP %s all %d arms have a phase_summary row" % (cell, len(arms_)), "PASS")
        elif len(arms_) != full[cell]:
            g.add("SWEEPCOMP %s dispatched arms of the cell: %d of the %d of the manifest" % (cell, len(arms_), full[cell]), "INFO")
    group_files(g, table, live, ret_dirs, runs)
    return g, id_rows


def det_gates(g, t, runs, by_id, popen):
    arm, ref = t["id"], t["ref"]
    dg = lambda a: (rd(os.path.join(runs, a), "state_trace_blank_runid.sha256") or "").strip() or None
    if ref is None:
        g.add("DET %s names no original in the manifest (ref column)" % arm, "FAIL"); return
    if ref not in by_id:
        g.add("DET %s digest == %s" % (arm, ref), "INFO", "reference arm is not in this manifest"); return
    a, b = dg(arm), dg(ref)
    miss = [n for n, x in ((arm, a), (ref, b)) if x is None]
    g.add("DET %s digest == %s" % (arm, ref), "MISSING" if miss else ("PASS" if a == b else "FAIL"), ("digest file absent for: " + ", ".join(miss)) if miss else "%s vs %s" % (a[:16], b[:16]))
    Pa, Pb = popen(arm), popen(ref)
    nm = "DETPOS %s pos trace identical to %s (full precision, rule 3.7)" % (arm, ref)
    if Pa is None or Pb is None:
        g.add(nm, "MISSING", "pos trace absent for: %s" % ", ".join(n for n, x in ((arm, Pa), (ref, Pb)) if x is None)); return
    cols = [c for c in ("scan_id", "px", "py", "pz", "yaw_rad") if c in Pa.columns and c in Pb.columns]
    same_ = len(Pa) == len(Pb) and Pa[cols].reset_index(drop=True).equals(Pb[cols].reset_index(drop=True))
    g.add(nm, "PASS" if same_ else "FAIL", "rows %d vs %d" % (len(Pa), len(Pb)))


def ident_rows(t, runs, ref_returns, ref_runs, manifest, ref_manifests, tol_mm):
    if not ref_returns:
        return [("per-scan position, excursion, settled offset and digest equal %s" % t["ref"], "MISSING", "no --ref-return given: the R82 reference return is required for the identity rows")], None
    r = IH.check_arm(t["id"], t["ref"], runs, list(ref_returns), list(ref_runs), [manifest], list(ref_manifests), tol_mm)
    return [(n, s, dt) for n, s, dt in r["rows"]], r


def ident_csv_row(t, runs, ref_returns, tol_mm):
    d = os.path.join(runs, t["id"])
    P, _ = IH.pos_of(d) if os.path.isdir(d) else (None, None)
    R, _ = IH.ref_pos(list(ref_returns), t["ref"]) if ref_returns else (None, None)
    row = dict(arm=t["id"], ref_arm=t["ref"], n_common_scans="", pos_max_abs_diff_mm="", yaw_max_abs_diff_mrad="", dump_equal="", node_id_equal="", digest_equal_if_known="")
    if P is not None and R is not None:
        c = IH.pos_compare(P, R, tol_mm)
        row.update(n_common_scans=c["n_common"], pos_max_abs_diff_mm=c["max_mm"], yaw_max_abs_diff_mrad="" if c["yaw_mrad"] is None else c["yaw_mrad"])
    da, db = IH.arm_digest(d), IH.ref_digest(list(ref_returns), t["ref"]) if ref_returns else None
    if da and db:
        row["digest_equal_if_known"] = "yes" if da == db else "no"
    return row


def group_files(g, table, live, ret_dirs, runs):
    """PFT: group-level completeness from files of the return (found in ret_dirs)."""
    dirs = list(ret_dirs) + [runs, os.path.join(runs, "..")]
    pft = [t for t in table if t["id"] in live and (t["ov"] or {}).get(K_TRACE) not in (None, 0, False, "0")]
    if pft:
        p = find_return_file("plane_fit_trace_counts_all.csv", dirs)
        C = pd.read_csv(p) if p else None
        if C is None or "arm" not in C.columns:
            g.add("PFT plane_fit_trace_counts_all.csv has rows for the %d arms whose job sets %s (written by the collector)" % (len(pft), K_TRACE), "MISSING", "file absent")
        else:
            bad = 0
            for t in pft:
                n = int(C[C["arm"] == t["id"]]["rows"].sum()) if "rows" in C.columns else 0
                if n <= 0:
                    bad += 1; g.add("PFT %s has rows in plane_fit_trace_counts_all.csv" % t["id"], "MISSING", "no rows")
            if not bad:
                g.add("PFT all %d arms whose job sets %s have rows in plane_fit_trace_counts_all.csv" % (len(pft), K_TRACE), "PASS")


def write_hold(g, hold_file):
    bad = [(n, s, d) for n, s, d in g.rows if s == "FAIL" and (n.startswith("DET ") or n.startswith("DETPOS ") or n.startswith("IDENT "))]
    if not bad:
        return False
    import time
    body = ["HOLD (from r84_gates.py, %s): a determinism or identity gate FAILED. Do not dispatch the unstarted queues; let running jobs finish; cancel nothing; delete nothing; change nothing in the tree." % time.strftime("%Y-%m-%d %H:%M:%S"),
            "Investigate (config dump diff, run.log, first-divergence scan, tree diff), report to the planner. The planner's written word lifts the hold (rename HOLD.txt to HOLD.released_<date>.txt; never delete it).", ""]
    body += ["%s   [%s]" % (n, d) for n, s, d in bad]
    with open(hold_file, "a" if os.path.exists(hold_file) else "w") as f:
        f.write("\n".join(body) + "\n")
    return True


# ------------------------------------------------------------------------------------------------ selftest fixtures
def selftest():
    """Synthetic: a manifest written here (arm_table.tsv + jobs_x.json; no generator is imported) and fabricated arm folders. Planner sandbox, not evidence about the node."""
    root = tempfile.mkdtemp(prefix="r84_gates_selftest_")
    fails, nck = [], [0]

    def ck(c, m):
        nck[0] += 1
        if not c:
            fails.append(m)
    cols = ["arm", "tier", "group", "fam", "mode", "kind", "arm_kind", "ref", "bag", "seq", "batch", "pfn", "duration_s", "msglog", "needles", "flag", "fit", "lines", "edge", "layer", "sw_ms", "tbpn", "l2_ref", "l2_source", "control",
            "same_batch_control"]
    ARMS = []
    BASE = {"cbk.lidar.point_filter_num": 3, "imu.ds.mode": "first"}
    GRID = {"eee_01": {"t0_s": 100.0, "period_s": 0.1, "seq0": 5, "n": 3987}}
    RT = {"cbk.lidar.overlap_policy": "retime_trim", "cbk.lidar.retime_t0_s": 100.0, "cbk.lidar.retime_period_s": 0.1, "cbk.lidar.retime_seq0": 5, "cbk.lidar.retime_ref_offset_s": -0.1}
    DEBON = {K_FIT: "debiased", K_LINE: True, "voxel_map.plane.line_residual.variance": "plane_equivalent", "voxel_map.plane.line_residual.rows": "distance", "voxel_map.plane.plane_gap_floor_en": True, K_CLS: True, K_EQV: "mean_cov"}

    def arm(name, group, ov, mode="dc", kind="fb", ref="", bag="eee_01", dur="full", msglog=0, needles="", fam="", **extra):
        base = dict(BASE, **(RT if mode == "rt" else {}))
        ARMS.append((dict(dict(arm=name, tier="FB", group=group, fam=fam or group, mode=mode, kind=kind, arm_kind=kind, ref=ref, bag=bag, seq=bag, batch="q_" + bag, pfn=3, duration_s=dur, msglog=msglog, needles=needles, flag=""), **extra),
                     dict(base, **ov)))
    sn = lambda kind, fit, lines, tag, q="fq", ovl="dc": "r84_%s_%s_eee_01_%s_%s_v%s_%s" % (ovl, kind, fit, lines, tag, q)

    def pair(fit, lines, tag, edge, extra_t3=None, extra_l2=None, t3_extra_row=None, src="R84"):
        ov = dict(DEBON) if (fit == "deb" and lines == "on") else {}
        t3, l2 = sn("t3", fit, lines, tag), sn("l2", fit, lines, tag)
        arm(t3, "T3", dict(ov, **{K_VOX: edge, K_MAXL: 0}, **(extra_t3 or {})), fit=fit, lines=lines, edge=edge, layer=0, l2_ref=l2, l2_source=src, control=l2, **(t3_extra_row or {}))
        arm(l2, "L2R", dict(ov, **{K_VOX: edge}, **(extra_l2 or {})), fit=fit, lines=lines, edge=edge, layer=2, l2_ref=t3, l2_source="R84", control=t3)
        return t3, l2
    # ---- sweep arms
    T3a, L2a = pair("pca", "off", "0p35", 0.35)
    T3b, L2b = pair("deb", "on", "0p4", 0.4, extra_t3={"voxel_map.plane.plane_fit_reasons_en": True, K_TRACE: 10}, extra_l2={"voxel_map.plane.plane_fit_reasons_en": True, K_TRACE: 10})
    T3c, L2c = pair("pca", "off", "0p45", 0.45)                  # echo of the T3 arm shows max_layer 2
    T3d = sn("t3", "pca", "off", "0p5")                           # job edge 0.55 against the name v0p5
    arm(T3d, "T3", {K_VOX: 0.55, K_MAXL: 0}, fit="pca", lines="off", edge=0.5, layer=0, l2_ref="r81_rt_sw_x", l2_source="R81", control="r81_rt_sw_x")
    T3e = sn("t3", "pca", "off", "0p55")                          # l2_source outside the set
    arm(T3e, "T3", {K_VOX: 0.55, K_MAXL: 0}, fit="pca", lines="off", edge=0.55, layer=0, l2_ref="r99_x", l2_source="R99", control="r99_x")
    T3f = sn("t3", "pca", "off", "0p6")                           # l2_ref empty
    arm(T3f, "T3", {K_VOX: 0.6, K_MAXL: 0}, fit="pca", lines="off", edge=0.6, layer=0, l2_ref="", l2_source="R83")
    T3g = sn("t3", "pca", "off", "0p65")                          # R84 source whose twin does not exist
    arm(T3g, "T3", {K_VOX: 0.65, K_MAXL: 0}, fit="pca", lines="off", edge=0.65, layer=0, l2_ref=sn("l2", "pca", "off", "0p65"), l2_source="R84")
    T3h, L2h = pair("pca", "off", "0p7", 0.7, extra_l2={"cbk.lidar.point_filter_num": 4})      # the pair differs in one more key
    T3i, L2i = pair("deb", "on", "0p3", 0.3)                       # echo prints neither classify_at_fit nor debiased_line_equiv
    T3j, L2j = pair("deb", "on", "0p75", 0.75)                     # classify / equiv only in the engagement line
    T3k, L2k = pair("deb", "on", "0p8", 0.8)                       # equiv printed off
    L2m = sn("l2", "pca", "on", "0p85")                            # L2R arm with a control that is no T3 arm; echo max_layer 0
    arm(L2m, "L2R", {K_VOX: 0.85, K_LINE: True}, fit="pca", lines="on", edge=0.85, layer=2, l2_ref="nothing", l2_source="R84", control="nothing")
    # ---- CMP arms (rt)
    cm = lambda ms, tb, q="fq": "r84_rt_cmp_eee_01_w%dms_%s_%s" % (ms, "vacc" if tb == "var_acc" else "none", q)
    c0 = cm(0, "var_acc")
    for ms, tb, q, extra in ((0, "var_acc", "fq", {}), (5, "var_acc", "fq", {}), (5, "none", "fq", {}), (10, "var_acc", "fq", {}), (2, "var_acc", "fq", {"sw_ms": 3}), (1, "var_acc", "fq", {}), (20, "var_acc", "fq", {}), (5, "none", "mq", {}),
                             (0, "none", "fq", {}), (0, "var_acc", "mq", {})):
        ov = {K_TBPN: tb, K_MAXL: 2}
        if ms:
            ov[K_SW] = ms / 1000.0
        row = dict(sw_ms=ms, tbpn=tb, l2_ref="r83_rt_ml_eee_01_l2_%s" % q if ms != 10 else "r83_rt_ml_missing", l2_source="R83", control="r82_rt_ds_eee_01_first_fq", same_batch_control=c0 if ms != 10 else cm(0, "var_acc", "mq") + "_gone")
        row.update(extra)
        arm(cm(ms, tb, q), "CMP", ov, mode="rt", fit="pca", lines="off", edge=0.425, layer=2, **row)
    # ---- OWN arms
    arm("id_sw0", "OWN", {K_SW: 0.0}, mode="rt", kind="id", ref="r82_ref_ok", fam="ID"); arm("id_sw0_bad", "OWN", {K_SW: 0.0}, mode="rt", kind="id", ref="r82_ref_ok", fam="ID")
    SM = {K_VOX: 0.5, K_MAXL: 0}
    arm("sm_sw", "OWN", {K_SW: 0.005}, kind="smoke", dur="60", fam="SMK")
    arm("sm_l0_a", "OWN", dict(SM, **{K_LINE: True, "voxel_map.plane.line_residual.variance": "plane_equivalent", "voxel_map.plane.line_residual.rows": "distance"}), kind="smoke", dur="60", fam="SMK")
    arm("sm_l0_b", "OWN", dict(SM, **{K_FIT: "debiased", "voxel_map.plane.plane_fit_reasons_en": True}), kind="smoke", dur="60", fam="SMK")
    arm("sm_l0_c", "OWN", dict(SM, **DEBON), kind="smoke", dur="60", fam="SMK")
    for nm_, key, val, needle in (("rf_mode", "lio.point_cov.mode", "propagated", "lio/point_cov/mode"), ("rf_win", "lio.point_cov.window_s", 0.01, "lio/point_cov/window_s"), ("rf_tcm", "lio.time_cov.mode", "shared", "lio/time_cov/mode"),
                                  ("rf_tcs", "lio.time_cov.sigma_s", 1e-4, "lio/time_cov/sigma_s"), ("rf_neg", K_SW, -0.01, "lio/point_cov/sample_window_s"), ("rf_big", K_SW, 0.2, "lio/point_cov/sample_window_s")):
        arm(nm_, "OWN", {key: val}, kind="refuse", needles=needle, dur="5", fam="REF")
    arm("x_clean", "OWN", {}); arm("x_dcret", "OWN", {"cbk.lidar.retime_t0_s": 1.0}); arm("x_rm_echo", "OWN", {}); arm("x_rm_job", "OWN", {"lio.point_cov.window_s": 0.01}); arm("x_rm_eng", "OWN", {}); arm("x_zupt", "OWN", {}); arm("x_pv", "OWN", {})
    arm("det_a", "OWN", {}, kind="det_fb", ref="det_orig"); arm("det_orig", "OWN", {})
    arm("id_ok", "OWN", {}, kind="id", ref="r82_ref_ok"); arm("id_bad", "OWN", {}, kind="id", ref="r82_ref_bad")
    arm("msg_dc", "OWN", {}, msglog=1)
    mdir = os.path.join(root, "m"); os.makedirs(mdir)
    with open(os.path.join(mdir, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows([r for r, _ in ARMS])
    json.dump([dict(job_id=r["arm"], overrides=o) for r, o in ARMS], open(os.path.join(mdir, "jobs_x.json"), "w"))
    open(os.path.join(mdir, "l2_reference_values.csv"), "w").write("arm,source\nr83_rt_ml_eee_01_l2_fq,R83\nr83_rt_ml_eee_01_l2_mq,R83\n")
    runs = os.path.join(root, "runs"); os.makedirs(runs)

    def echo(mode, **kw):
        d = {"cbk__lidar__point_filter_num": 3, "imu__ds__mode": "first", "map__max_layer": 2}
        d.update({"cbk__lidar__overlap_policy": "retime_trim", "cbk__lidar__retime_t0_s": 100.0, "cbk__lidar__retime_period_s": 0.1, "cbk__lidar__retime_seq0": 5, "cbk__lidar__retime_ref_offset_s": -0.1} if mode == "rt" else {"cbk__lidar__overlap_policy": "discard"})
        d.update(kw)
        return "".join("  %s: %s\n" % (k.replace("__", "/"), v) for k, v in d.items() if v is not None) + ("[lidar_overlap_policy] policy=retime_trim\n" if mode == "rt" else "")

    def swe(mode, vox, maxl, fit="pca", on=False, **kw):
        d = {"map__voxel_size": vox, "map__max_layer": maxl, "plane__plane_fit_mode": fit, "line_residual__enabled": str(on).lower()}
        if fit == "debiased" and on:
            d.update({"line_residual__variance": "plane_equivalent", "line_residual__rows": "distance", "plane__plane_gap_floor_en": "true", "line_residual__classify_at_fit": "true", "plane__debiased_line_equiv": "mean_cov"})
        d.update(kw)
        return echo(mode, **d)
    D4L = "  voxel_map/plane/line_residual/classify_at_fit = true debiased_line_equiv = mean_cov plane_gap_floor_en = true line_classified=5 line_accepted=5 e1_tiny_kept=0 e0_clamped=0 gaps_floored=3 tilt_voxels_moment=0 tilt_voxels_meancov=4\n"

    def mk(name, log="  voxel_map/plane/x: 1\n", eng="", imu="", files=None, sentinel="DONE", exit_="exit_code=0 signal=none"):
        d = os.path.join(runs, name); os.makedirs(d, exist_ok=True)
        json.dump({"sentinel": sentinel}, open(os.path.join(d, "arm.json"), "w")); open(os.path.join(d, "exit_status.txt"), "w").write(exit_ + "\n")
        open(os.path.join(d, "run.log"), "w").write(log); open(os.path.join(d, "engagement.txt"), "w").write(eng)
        if imu:
            open(os.path.join(d, "engagement_imu.txt"), "w").write(imu)
        for fn, txt in (files or {}).items():
            open(os.path.join(d, fn), "w").write(txt)
    S = lambda g, pre: [s for n, s, _ in g.rows if n.startswith(pre)]
    SD = lambda g, pre: [d for n, s, d in g.rows if n.startswith(pre)]
    nonpass = lambda g, arm_: [(n, s) for n, s, _ in g.rows if (" %s " % arm_) in n and s in ("FAIL", "MISSING") and not n.startswith(("LEN ", "MSG"))]   # the traces are not fabricated for every arm
    SWL = lambda v, n=5, pe=1200, mt="2.5e-07", extra="": "  lio/point_cov sample_window_s=%s nodes=%d point_evaluations=%d clamped_points=0 mean_trace_m2=%s max_trace_m2=1e-05%s\n" % (v, n, pe, mt, extra)
    GR = {"eee_01": GRID["eee_01"]}
    # ---- sweep arms
    mk(T3a, log=swe("dc", 0.35, 0)); mk(L2a, log=swe("dc", 0.35, 2))
    mk(T3b, log=swe("dc", 0.4, 0, "debiased", True, plane__plane_fit_reasons_en="true", plane__plane_fit_trace_frames=10), eng=D4L); mk(L2b, log=swe("dc", 0.4, 2, "debiased", True, plane__plane_fit_reasons_en="true", plane__plane_fit_trace_frames=10), eng=D4L)
    mk(T3c, log=swe("dc", 0.45, 2)); mk(L2c, log=swe("dc", 0.45, 2))
    mk(T3d, log=swe("dc", 0.55, 0)); mk(T3e, log=swe("dc", 0.55, 0)); mk(T3f, log=swe("dc", 0.6, 0)); mk(T3g, log=swe("dc", 0.65, 0))
    mk(T3h, log=swe("dc", 0.7, 0)); mk(L2h, log=swe("dc", 0.7, 2, cbk__lidar__point_filter_num=4))
    mk(T3i, log=swe("dc", 0.3, 0, "debiased", True, line_residual__classify_at_fit=None, plane__debiased_line_equiv=None)); mk(L2i, log=swe("dc", 0.3, 2, "debiased", True), eng=D4L)
    mk(T3j, log=swe("dc", 0.75, 0, "debiased", True, line_residual__classify_at_fit=None, plane__debiased_line_equiv=None), eng=D4L); mk(L2j, log=swe("dc", 0.75, 2, "debiased", True), eng=D4L)
    mk(T3k, log=swe("dc", 0.8, 0, "debiased", True, plane__debiased_line_equiv="off"), eng=D4L); mk(L2k, log=swe("dc", 0.8, 2, "debiased", True), eng=D4L)
    mk(L2m, log=swe("dc", 0.85, 0, "pca", True))
    # ---- CMP arms
    mk(cm(0, "var_acc"), log=echo("rt", point_cov__sample_window_s=0, undistort__time_based_process_noise="var_acc"))
    mk(cm(5, "var_acc"), log=echo("rt", point_cov__sample_window_s=0.005, undistort__time_based_process_noise="var_acc"), imu=SWL("0.005"))
    mk(cm(5, "none"), log=echo("rt", point_cov__sample_window_s=0.005, undistort__time_based_process_noise="none"), imu=SWL("0.005", extra="   <-- INERT: sample_window_s selected and no point was evaluated"))
    mk(cm(10, "var_acc"), log=echo("rt", point_cov__sample_window_s=0.01, undistort__time_based_process_noise="var_acc"), imu=SWL("0.02"))
    mk(cm(2, "var_acc"), log=echo("rt", point_cov__sample_window_s=0.002, undistort__time_based_process_noise="var_acc"))
    mk(cm(1, "var_acc"), log=echo("rt", point_cov__sample_window_s=0.001, undistort__time_based_process_noise="var_acc"), imu=SWL("0.001", n=3))
    mk(cm(20, "var_acc"), log=echo("rt", point_cov__sample_window_s=0.02, undistort__time_based_process_noise="var_acc"), imu=SWL("0.02", mt="0"))
    mk(cm(5, "none", "mq"), log=echo("rt", point_cov__sample_window_s=0.005, undistort__time_based_process_noise="none"), imu=SWL("0.005", extra="   <-- NOT ENGAGED: every added covariance was zero"))
    mk(cm(0, "none"), log=echo("rt", point_cov__sample_window_s=0, undistort__time_based_process_noise="var_acc"), imu=SWL("0.005"))
    mk(cm(0, "var_acc", "mq"), log=echo("rt"))
    g, _ = run_gates(runs, mdir, grid_retime=GR, expect={"T3": (len([1 for r, _ in ARMS if r["group"] == "T3"]),), "L2R": (len([1 for r, _ in ARMS if r["group"] == "L2R"]),), "CMP": (10,)})
    ck(all(s == "PASS" for n, s, _ in g.rows if n.startswith("FAMCOUNT") and "manifest holds" in n), "FAMCOUNT must PASS for the fixture counts: %s" % [(n, s) for n, s, _ in g.rows if n.startswith("FAMCOUNT")])
    g2, _ = run_gates(runs, mdir, grid_retime=GR, expect={"T3": (1296,), "L2R": (846,), "CMP": (216,)})
    ck(all(s == "FAIL" for n, s, _ in g2.rows if n.startswith("FAMCOUNT") and "manifest holds" in n), "FAMCOUNT must FAIL when the manifest holds other counts than 1296 / 846 / 216")
    ck(EXPECT == {"T3": (1296,), "L2R": (846,), "CMP": (216,)}, "the default EXPECT is the R84 table")
    # ---- T3ENG / L2RENG
    ck(not nonpass(g, T3a) and not nonpass(g, L2a), "the good T3 / L2R pair has no FAIL or MISSING row: %s %s" % (nonpass(g, T3a), nonpass(g, L2a)))
    ck(S(g, "T3ENG %s params echo" % T3a) == [] and any(n.startswith("CELL T3ENG T3_dc_eee_01_fq") for n, s, _ in g.rows), "sweep PASS rows are folded into CELL rows (no own row for a PASS)")
    ck(S(g, "T3ENG %s params echo" % T3c) == ["FAIL"] and "max_layer" in SD(g, "T3ENG %s params echo" % T3c)[0], "T3 echo max_layer 2 against the job's 0 must FAIL")
    ck(S(g, "T3ENG %s manifest" % T3d) == ["FAIL"] and "arm name says" in SD(g, "T3ENG %s manifest" % T3d)[0], "job edge differs from the edge in the name and from arm_table: FAIL")
    ck(S(g, "T3ENG %s l2_ref" % T3e) == ["FAIL"] and S(g, "T3ENG %s l2_ref" % T3f) == ["FAIL"], "l2_source outside the set FAIL, empty l2_ref FAIL")
    ck(S(g, "T3ENG %s twin" % T3g) == ["FAIL"], "R84 l2_source without a twin L2R arm: FAIL")
    ck(S(g, "T3ENG %s twin" % T3h) == ["FAIL"] and "point_filter_num" in SD(g, "T3ENG %s twin" % T3h)[0], "the pair differs in more than max_layer: FAIL")
    ck(S(g, "T3ENG %s twin" % T3a) == [] and any(n.startswith("CELL T3ENG T3_dc_eee_01_fq") for n, s, _ in g.rows), "the good pair's twin row is PASS (folded into a CELL row)")
    ck(S(g, "T3ENG %s deb + lines on" % T3b) == [] and not nonpass(g, T3b) and not nonpass(g, L2b), "deb+on arm with classify / equiv in the echo and the D4 line: PASS: %s %s" % (nonpass(g, T3b), nonpass(g, L2b)))
    ck(S(g, "T3ENG %s deb + lines on" % T3i) == ["MISSING"], "deb+on arm where neither echo nor engagement shows classify / equiv: MISSING")
    ck(S(g, "T3ENG %s deb + lines on" % T3j) == [] and S(g, "D4ENG %s" % T3j) == [], "classify / equiv only in the engagement line: PASS")
    ck(S(g, "T3ENG %s deb + lines on" % T3k) == ["FAIL"], "debiased_line_equiv printed off against the job's mean_cov: FAIL")
    ck(S(g, "L2RENG %s params echo" % L2m) == ["FAIL"] and S(g, "L2RENG %s control" % L2m) == ["FAIL"], "L2R echo max_layer 0 against the default 2 FAIL; control that is no T3 arm FAIL")
    ck(S(g, "L2RENG %s params echo" % L2a) == [], "L2R echo max_layer 2 (job has no key: default 2): PASS (folded)")
    # ---- SWENG
    c5 = cm(5, "var_acc")
    ck(not nonpass(g, c5) and not nonpass(g, c0), "good CMP arms (sample window on / off) have no FAIL or MISSING row: %s %s" % (nonpass(g, c5), nonpass(g, c0)))
    ck(S(g, "SWENG %s engagement" % c5) == ["PASS"] and S(g, "SWENG %s params echo" % c5) == ["PASS"] and S(g, "SWENG %s clamped" % c5) == ["INFO"], "SWENG good arm: line PASS, echo PASS, clamped_points INFO")
    ck(S(g, "SWENG %s engagement" % cm(5, "none")) == ["FAIL"] and "INERT" in SD(g, "SWENG %s engagement" % cm(5, "none"))[0], "INERT marker: FAIL")
    ck(S(g, "SWENG %s engagement" % cm(5, "none", "mq")) == ["FAIL"] and "NOT ENGAGED" in SD(g, "SWENG %s engagement" % cm(5, "none", "mq"))[0], "NOT ENGAGED marker: FAIL")
    ck(S(g, "SWENG %s engagement" % cm(10, "var_acc")) == ["FAIL"] and "sample_window_s=0.02" in SD(g, "SWENG %s engagement" % cm(10, "var_acc"))[0], "sample_window_s in the line differs from the job: FAIL")
    ck(S(g, "SWENG %s engagement" % cm(2, "var_acc")) == ["FAIL"], "key on but no engagement line: FAIL")
    ck(S(g, "SWENG %s engagement" % cm(1, "var_acc")) == ["FAIL"] and "nodes" in SD(g, "SWENG %s engagement" % cm(1, "var_acc"))[0], "nodes != 5: FAIL")
    ck(S(g, "SWENG %s engagement" % cm(20, "var_acc")) == ["FAIL"] and "mean_trace_m2" in SD(g, "SWENG %s engagement" % cm(20, "var_acc"))[0], "mean_trace_m2 = 0: FAIL")
    ck(S(g, "NOEXTRA %s" % cm(0, "none")) == ["FAIL"] and "sample_window_s=" in SD(g, "NOEXTRA %s" % cm(0, "none"))[0], "key absent but an engagement line: NOEXTRA FAIL")
    ck(S(g, "TBPNENG %s" % cm(0, "none")) == ["FAIL"], "tbpn job none, echo var_acc: TBPNENG FAIL")
    ck(S(g, "TBPNENG %s" % c5) == ["PASS"], "tbpn echo equals the job: PASS")
    ck(S(g, "CMPCFG %s" % cm(2, "var_acc")) == ["FAIL"] and S(g, "CMPCFG %s" % c5) == ["PASS"], "arm_table sw_ms against the job: FAIL / PASS")
    ck(any("PRESENT" in d and "R83 arm r83_rt_ml_eee_01_l2_fq" in n for n, s, d in g.rows if n.startswith("CMPPAIR %s" % c5)) and SD(g, "CMPPAIR %s" % c5)[0] == "same-batch control PRESENT; R83 arm PRESENT", "CMPPAIR PRESENT / PRESENT: %s" % SD(g, "CMPPAIR %s" % c5))
    ck(SD(g, "CMPPAIR %s" % cm(10, "var_acc")) == ["same-batch control ABSENT; R83 arm ABSENT"] and S(g, "CMPPAIR %s" % cm(10, "var_acc")) == ["INFO"], "CMPPAIR ABSENT / ABSENT is INFO")
    # identity arm with the key explicitly 0
    mk("id_sw0", log=echo("rt", point_cov__sample_window_s=0)); mk("id_sw0_bad", log=echo("rt", point_cov__sample_window_s=0), imu=SWL("0.005"))
    # ---- smoke, refuse, removed keys
    mk("sm_sw", log=echo("dc", point_cov__sample_window_s=0.005), imu=SWL("0.005"))
    L0 = lambda **kw: swe("dc", 0.5, 0, **kw)
    mk("sm_l0_a", log=L0(fit="pca", on=True, line_residual__variance="plane_equivalent", line_residual__rows="distance"))
    mk("sm_l0_b", log=L0(fit="debiased", plane__plane_fit_reasons_en="true"))
    mk("sm_l0_c", log=L0(fit="debiased", on=True), eng=D4L)
    RM = lambda k, extra="was removed in R84 (F-521) and has no replacement": "terminate: configuration key '%s' %s; delete it from the configuration" % (k, extra)
    FL = dict(sentinel="FAILED", exit_="exit_code=-6 signal=SIGABRT")
    mk("rf_mode", log=RM("lio/point_cov/mode"), **FL); mk("rf_win", log="terminate: something unrelated, removed", **FL); mk("rf_tcm", log=RM("lio/time_cov/mode", "is no longer accepted"), **FL)
    mk("rf_tcs", log=RM("lio/time_cov/sigma_s"), **FL); mk("rf_neg", log="terminate: lio/point_cov/sample_window_s = -0.01 outside [0, 0.1]", **FL)
    mk("rf_big", log="terminate: lio/point_cov/sample_window_s = 0.2 outside [0, 0.1]", sentinel="DONE", exit_="exit_code=0 signal=none")
    mk("x_clean"); mk("x_dcret"); mk("x_rm_echo", log="  lio/time_cov/mode: shared\n"); mk("x_rm_job"); mk("x_rm_eng", eng="  lio/point_cov/window_s = 0.02\n"); mk("x_zupt", log="  lio/zupt/mode = off\n")
    mk("x_pv", eng="  voxel_map/plane/plane_var_min = 0.0004 rows=12\n")
    g, _ = run_gates(runs, mdir, grid_retime=GR)
    ck(S(g, "SWENG id_sw0 key explicitly 0") == ["PASS"] and S(g, "SWENG id_sw0_bad key explicitly 0") == ["FAIL"], "identity arm with the key explicitly 0: no engagement line PASS, a line FAIL: %s %s" % (S(g, "SWENG id_sw0 "), S(g, "SWENG id_sw0_bad")))
    ck(S(g, "SMOKE sm_sw") == ["PASS"] and S(g, "SMOKE sm_l0_a") == ["PASS"] and S(g, "SMOKE sm_l0_b") == ["PASS"] and S(g, "SMOKE sm_l0_c") == ["PASS"], "the four smoke arms PASS on good fixtures: %s" % [(n, s, d) for n, s, d in g.rows if n.startswith("SMOKE")])
    ck(S(g, "L0SMOKE all 3") == ["PASS"], "all layer-0 smokes exit 0: L0SMOKE PASS")
    ck(S(g, "REMOVED rf_mode refused") == ["PASS"] and S(g, "REMOVED rf_mode the refusal") == ["PASS"] and S(g, "REMOVED rf_tcs refused") == ["PASS"], "refuse arm with exit != 0, the needle and the word removed: PASS")
    ck(S(g, "REMOVED rf_win refused") == ["FAIL"] and "False" in SD(g, "REMOVED rf_win refused")[0], "refuse arm whose text lacks the needle: FAIL")
    ck(S(g, "REMOVED rf_tcm refused") == ["PASS"] and S(g, "REMOVED rf_tcm the refusal") == ["INFO"], "removed-key refusal without the word removed: INFO")
    ck(S(g, "REMOVED rf_neg refused") == ["PASS"] and S(g, "REMOVED rf_neg the refusal") == [], "range refuse arm: exit and needle only, no `removed` row")
    ck(S(g, "REMOVED rf_big refused") == ["FAIL"], "refuse arm that exited 0: FAIL")
    ck(S(g, "REMOVED x_clean none") == ["PASS"] and S(g, "REMOVED x_rm_echo none") == ["FAIL"] and S(g, "REMOVED x_rm_job none") == ["FAIL"] and S(g, "REMOVED x_rm_eng none") == ["FAIL"], "removed key in a non-refuse arm's echo / job / engagement: FAIL, clean arm PASS")
    ck(S(g, "ZUPTGONE x_zupt") == ["FAIL"] and S(g, "ZUPTGONE x_clean") == ["PASS"], "lio/zupt in an echo: ZUPTGONE FAIL")
    ck(S(g, "NOEXTRA x_pv") == ["FAIL"] and S(g, "NOEXTRA x_clean") == ["PASS"], "NOEXTRA: plane variance engagement without the keys FAIL")
    g_full = g
    ck(S(g, "TBPNENG x_clean") == ["INFO"], "no job key and the echo does not print the key: INFO")
    hf = os.path.join(root, "HOLD_A.txt")
    ck(write_hold(g, hf) is False and not os.path.exists(hf), "SWENG / REMOVED / T3ENG FAILs must NOT write a hold file")
    # ---- L0SMOKE failure
    mk("sm_l0_c", log=L0(fit="debiased", on=True), eng=D4L, sentinel="FAILED", exit_="exit_code=1 signal=none")
    g, _ = run_gates(runs, mdir, grid_retime=GR)
    n_deb_on = len([1 for r, _ in ARMS if r["group"] == "T3" and r["fit"] == "deb" and r["lines"] == "on"])
    lr = [(n, s, d) for n, s, d in g.rows if n.startswith("L0SMOKE sm_l0_c")]
    ck(len(lr) == 1 and lr[0][1] == "FAIL" and "fit=deb lines=on" in lr[0][0] and ("(%d T3 arms)" % n_deb_on) in lr[0][0] and S(g, "SMOKE sm_l0_c") == ["FAIL"], "L0SMOKE failure names the family and the T3 count: %s (T3 deb on = %d)" % (lr, n_deb_on))
    ck(not any(n.startswith("L0SMOKE all") for n, _, _ in g.rows), "no all-PASS L0SMOKE row when one layer-0 smoke failed")
    ck(write_hold(g, hf) is False and not os.path.exists(hf), "a failed layer-0 smoke is reported and writes NO hold file")
    g, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="sm_sw,sm_l0_a")
    ck(S(g, "L0SMOKE sm_l0_c") == ["NOTRUN"] and S(g, "L0SMOKE sm_l0_b") == ["NOTRUN"], "undispatched layer-0 smokes: NOTRUN")
    os.remove(os.path.join(runs, "sm_l0_c", "exit_status.txt"))
    g, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="sm_l0_c")
    ck(S(g, "L0SMOKE sm_l0_c") == ["FAIL"], "a layer-0 smoke without an exit status did not finish with exit 0: FAIL")
    # ---- rt grid and the dc overlap rule
    g, _ = run_gates(runs, mdir, grid_retime={"eee_01": dict(GRID["eee_01"], seq0=6)}, dispatched=cm(5, "var_acc"))
    ck(S(g, "RTGRID %s" % c5) == ["FAIL"], "rt arm whose job grid differs from the bag grid: RTGRID FAIL")
    g, _ = run_gates(runs, mdir, dispatched=cm(5, "var_acc"))
    ck(S(g, "RTGRID %s" % c5) == ["MISSING"], "no grid given: RTGRID MISSING")
    ck(any(n.startswith("CELL OVLCFG T3_dc") and s == "PASS" for n, s, _ in g_full.rows) and S(g_full, "OVLCFG x_dcret overlap mode dc") == ["FAIL"], "dc arm without retime / policy keys: OVLCFG PASS (folded); a dc arm with a retime key FAIL")
    # ---- completeness and the trace rule of the sweep groups
    ph = pd.DataFrame({"arm": [x for x in (T3a, L2a, T3b, L2b, T3c) ], "gt_max_mm": [600.0, 100.0, 700.0, 50.0, 10.0]})
    d_ = os.path.join(runs, T3b); pd.DataFrame({"scan_id": [1, 2], "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d_, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    g, _ = run_gates(runs, mdir, phase=ph, grid_retime=GR, dispatched="T3,L2R")
    ck(S(g, "TRACE %s" % T3a) == ["MISSING"] and S(g, "TRACE %s" % T3b) == [] and not any(n.startswith("TRACE %s" % L2a) for n, _, _ in g.rows), "collapsed sweep arm (gt_max_mm > 500) needs a pos trace; others need none")
    ck(any(n.startswith("SWEEPCOMP %s has a phase_summary row" % T3d) and s == "MISSING" for n, s, _ in g.rows) and not any(n.startswith("LEN %s" % T3a) for n, _, _ in g.rows), "SWEEPCOMP: an arm without a phase row is MISSING; sweeps have no LEN row")
    # ---- DET and IDENT write the hold; nothing else does
    sid = np.arange(1, 41)

    def trace(a, shift=0.0):
        d = os.path.join(runs, a); os.makedirs(d, exist_ok=True)
        pd.DataFrame({"scan_id": sid, "px": sid * 1e-3 + (shift * 1e-3 * (sid >= 20)), "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        json.dump({"excursion_mm": 10.0, "settled_offset_mm": 5.0}, open(os.path.join(d, "arm_summary.json"), "w"))
    mk("det_a", files={"state_trace_blank_runid.sha256": "aa" * 32 + "\n"}); mk("det_orig", files={"state_trace_blank_runid.sha256": "bb" * 32 + "\n"})
    trace("det_a", 0.02); trace("det_orig")
    ref = os.path.join(root, "R82"); os.makedirs(os.path.join(ref, "compact", "r82_ref_ok")); os.makedirs(os.path.join(ref, "compact", "r82_ref_bad"))
    for a_ in ("r82_ref_ok", "r82_ref_bad"):
        pd.DataFrame({"scan_id": sid, "px": sid * 1e-3, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(ref, "compact", a_, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    pd.DataFrame([dict(arm=a_, excursion_mm=10.0, settled_offset_mm=5.0) for a_ in ("r82_ref_ok", "r82_ref_bad")]).to_csv(os.path.join(ref, "phase_summary.csv"), index=False)
    mk("id_ok"); mk("id_bad"); trace("id_ok"); trace("id_bad", 0.05)
    hold = os.path.join(root, "HOLD_FROM_GATES.txt")
    g, idr = run_gates(runs, mdir, ref_returns=[ref], dispatched="det_a,det_orig,id_ok,id_bad")
    ck(S(g, "DET det_a") == ["FAIL"] and S(g, "DETPOS det_a") == ["FAIL"], "DET and DETPOS FAIL on differing digest / trace")
    ck(any(n.startswith("IDENT id_ok") and s == "PASS" for n, s, _ in g.rows) and any(n.startswith("IDENT id_bad") and s == "FAIL" for n, s, _ in g.rows), "IDENT against the R82 return: id_ok PASS, id_bad FAIL")
    ck(write_hold(g, hold) and "DET det_a" in open(hold).read() and "IDENT id_bad" in open(hold).read(), "a DET / DETPOS / IDENT FAIL writes HOLD_FROM_GATES.txt")
    ck(len(idr) == 2 and {r["arm"] for r in idr} == {"id_ok", "id_bad"}, "identity csv rows: %r" % [r["arm"] for r in idr])
    # ---- MSGSCANS: INFO for a dc arm
    d = os.path.join(runs, "msg_dc"); mk("msg_dc"); trace("msg_dc")
    L = pd.DataFrame({"msg_index": range(50), "seq": range(50), "header_stamp": 0.0, "scan_time": 0.0, "gap_header_s": 0.1, "n_raw": 1, "n_kept": 1, "t_first": 0.0, "t_last": 0.0, "off_first_s": 0.0, "off_last_s": 0.0, "span_s": 0.1,
                      "prev_accepted_last_t": 0.0, "overlap_s": [0.0] * 30 + [0.01] * 20, "verdict": [0] * 30 + [1] * 20})
    L.to_csv(os.path.join(d, "lidar_msg_log_all.csv.gz"), index=False, compression="gzip")
    g, _ = run_gates(runs, mdir, dispatched="msg_dc")
    ck(any(n.startswith("MSGSCANS msg_dc dc arm") and s == "INFO" for n, s, _ in g.rows) and not any(n.startswith("MSGSCANS") and s == "FAIL" for n, s, _ in g.rows), "MSGSCANS for a dc arm is INFO")
    # ---- PFT: arms whose job sets plane_fit_trace_frames
    retd = os.path.join(root, "ret"); os.makedirs(retd)
    open(os.path.join(retd, "plane_fit_trace_counts_all.csv"), "w").write("arm,rows\n%s,12\n" % T3b)
    g, _ = run_gates(runs, mdir, ret_dirs=[retd], dispatched=T3b)
    ck(any(n.startswith("PFT all 1 arms") and s == "PASS" for n, s, _ in g.rows), "PFT passes for the arm with trace rows: %s" % [n for n, s, _ in g.rows if "PFT" in n])
    open(os.path.join(retd, "plane_fit_trace_counts_all.csv"), "w").write("arm,rows\nother,3\n")
    g, _ = run_gates(runs, mdir, ret_dirs=[retd], dispatched=T3b)
    ck(any(n.startswith("PFT %s" % T3b) and s == "MISSING" for n, s, _ in g.rows), "PFT: no rows for the arm is MISSING")
    # ---- no leftover R83 family gates in the gate code
    src = open(os.path.abspath(__file__)).read().split("def selftest")[0]
    ck(not re.search(r"DSTENG|MXLENG|VSWENG|PCWENG|PCLENG|MQSENG|MQSFILE|ODOMEXP|HILCOMP|PTENG|DGTENG|ABORTED_RSS|window_cov_profile|peak_rss", src), "no gate of a family that does not exist in R84 remains in the gate code")
    shutil.rmtree(root, ignore_errors=True)
    print("SELFTEST r84_gates", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (nck[0], len(fails)),
          "" if fails else "(T3ENG / L2RENG, SWENG, TBPNENG, REMOVED, ZUPTGONE, SMOKE, L0SMOKE, CMPPAIR, CMPCFG, NOEXTRA, RTGRID / OVLCFG, TRACE / SWEEPCOMP rows PASS on good fixtures and FAIL, MISSING, INFO or NOTRUN on bad ones; "
          "INERT / NOT ENGAGED marker, wrong echo max_layer, removed key in a non-refuse arm, missing needle and a failed layer-0 smoke FAIL without a hold file; DET and IDENT FAIL write HOLD_FROM_GATES.txt; "
          "synthetic fixtures, planner sandbox, not evidence)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--manifest", default="r84_manifest"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grid-retime", default=None); ap.add_argument("--grid-bag", default=None); ap.add_argument("--ref-return", action="append", default=[]); ap.add_argument("--ref-runs", action="append", default=[])
    ap.add_argument("--ref-manifest", action="append", default=[]); ap.add_argument("--phase-summary", default=None); ap.add_argument("--ret-dir", action="append", default=[])
    ap.add_argument("--dispatched", default=None); ap.add_argument("--hold-file", default=None); ap.add_argument("--no-famcount", action="store_true")
    ap.add_argument("--expect", default=None, help="group counts for FAMCOUNT, e.g. \"T3=1296,L2R=846,CMP=216\" (a count may list alternatives with |); default: the SPEC table of this file")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runs:
        sys.exit("--runs is required")
    expect = {k.strip(): tuple(int(x) for x in v.split("|")) for k, v in (p.split("=") for p in a.expect.split(","))} if a.expect else EXPECT
    gb = a.grid_bag or os.path.join(a.manifest, "grid_bag.json")
    grid_retime = json.load(open(a.grid_retime)) if a.grid_retime and os.path.exists(a.grid_retime) else None
    grid_bag = json.load(open(gb)) if os.path.exists(gb) else None
    ph = a.phase_summary
    rets = list(a.ret_dir) + ([os.path.dirname(os.path.abspath(a.out))] if a.out else [])
    if ph is None:
        ph = next((c for c in [os.path.join(d, "phase_summary.csv") for d in rets] + [os.path.join(a.runs, "phase_summary.csv"), os.path.join(a.runs, "..", "phase_summary.csv")] if os.path.exists(c)), None)
    g, id_rows = run_gates(a.runs, a.manifest, grid_retime, grid_bag, a.ref_return, a.ref_runs, ph, a.dispatched, rets, {} if a.no_famcount else expect, a.ref_manifest)
    txt = report(g)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    if a.csv:
        cols = ["arm", "ref_arm", "n_common_scans", "pos_max_abs_diff_mm", "yaw_max_abs_diff_mrad", "dump_equal", "node_id_equal", "digest_equal_if_known"]
        pd.DataFrame(id_rows, columns=cols).to_csv(a.csv, index=False)
    if a.hold_file and write_hold(g, a.hold_file):
        print("HOLD written: %s (a DET or IDENT gate FAILED)" % a.hold_file)
    return 1 if "FAIL" in [l.split()[0] for l in txt.splitlines() if l] else 0


if __name__ == "__main__":
    sys.exit(main())
