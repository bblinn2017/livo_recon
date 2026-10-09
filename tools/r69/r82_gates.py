#!/usr/bin/env python3
"""R82 gates (planning agent's script, derived from r81_gates.py; selftest on SYNTHETIC fixtures only, planner sandbox, not evidence).

    python3 tools/r69/r82_gates.py --runs <runs_dir> --manifest r82_manifest [--grid-retime r80c_grid.json] [--grid-bag r82_manifest/grid_bag.json] \
        [--ref-return R81_RETURN] [--ref-runs <old archive dir>] [--phase-summary phase_summary.csv] [--ret-dir <return dir>] \
        [--dispatched <arm ids file | arm_table.tsv | comma list of arm ids / groups / families / queues>] [--out r82_gates.txt] [--csv r82_identity_vs_old.csv] [--hold-file HOLD_FROM_GATES.txt]
    python3 tools/r69/r82_gates.py --selftest

Same row vocabulary and verdicts as r81_gates.py (PASS, FAIL, MISSING (never a pass), INFO, NOTRUN). The expectation of every arm is read from the MANIFEST ONLY (arm_table.tsv + jobs_*.json); every effective
config key found is compared with the job's override. Grid values come from --grid-retime (r80c_grid.json) and --grid-bag (grid_bag.json), not from the manifest.
Groups: F503, D1, D2, D4, D5, D6, F498, HIL, OWN. New rows in R82:
  TCENG       lio/time_cov engagement line (mode, sigma_s as the job states them, applied > 0; the smoke arm must not carry the INERT flag)
  PVDENG      plane_var_diag_en engagement block: rows > 0 and no INERT flag; plane_var_min / plane_var_scan_s2 lines as in R81 (VMINENG / VS2ENG)
  D4ENG       classify_at_fit / debiased_line_equiv / plane_gap_floor_en engagement block with the counters; a smoke arm must show line_classified > 0 (classify), tilt voxels > 0 (equiv), gaps_floored > 0 (floor on debiased)
  ZCENG       lio/zupt threshold_mode = calib_median, calib_scans as the job says, calibrated=1; ZCFILE zupt_all.csv.gz rows
  PTENG       cbk/lidar/point_time engagement (field, mode, ref) as the job states it
  DGTENG      evo/dense_gt_metrics engagement line and, in the [evo] line, ARE and ATE_first / ARE_first
  TCFILE      the omega smoke arm's time_cov.csv has the columns omega_w_norm, omega_fd_norm, omega_diff_norm (a fact about the frame is NOT judged here)
  SWCELL      F-498 Tier A: the nine edges of each cell (bag, Q), as R81 SWCELL
  HILCOMP     r82_hilti_metrics.csv has a row for every Hilti arm
  D5COMP      r82_switch_pairs.csv has a row for every switch arm
  NOEXTRA     no engagement line of a key the arm did not ask for
A FAIL of a DET or IDENT row writes --hold-file (HOLD_FROM_GATES.txt): the agent places it as HOLD.txt (r81_hold_check.sh). The gate never deletes or cancels anything.
"""
import argparse, csv, glob, gzip, io, json, os, re, shutil, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r82_identity_hold as IH  # noqa: E402

LEN_MIN = 0.97
REF_OFFSET = -0.1
N_SCANS = {"eee_01": 3987, "eee_02": 3210, "eee_03": 1814, "nya_01": 3950, "nya_02": 4287, "nya_03": 4094, "sbs_01": 3542, "sbs_02": 3732, "sbs_03": 3893}   # message / scan counts of the bags (make_r79_manifest.N_SCANS)
EXPECT = {"F503": (72,), "D1": (72,), "D2": (108,), "D4": (74,), "D5": (8,), "D6": (18,), "F498": (162,), "HIL": (48,)}   # SPEC table
EDGES_SWT = [0.20, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
MSG_COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict"]
MSG_COLS_C = MSG_COLS + ["retime_shift_s", "n_trimmed"]
ZD_COLS = ["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"]
SC_COLS = ["frame", "iter", "rot_raw", "pos_raw", "rot_applied", "pos_applied", "scale", "bound", "next_converged"]
EARLY_FRAMES = list(range(1, 21))
F2B_FRAMES = list(range(400, 411))
F2B_COLS = ["arm", "frame", "voxel_point_count", "voxel_distinct_frames"]
EARLY_COLS = ["arm", "frame"]
N479_PER_BAG = (4, 2)                        # switch arms (two directions x two modes), plain controls
SWT_RE = re.compile(r"^r82_rt_sw_(?P<bag>[a-z]+_\d\d)_(?P<line>off)_v(?P<edge>[0-9p]+)_pfn(?P<pfn>\d)_(?P<q>fq|mq)$")
RETIME_KEYS = ("cbk.lidar.retime_t0_s", "cbk.lidar.retime_period_s", "cbk.lidar.retime_seq0", "cbk.lidar.retime_ref_offset_s")
STRICT = re.compile(r"^(voxel_map\.map\.(voxel_size|max_layer)|cbk\.lidar\.(point_filter_num|overlap_policy|retime_.*|point_time_.*)|imu\.process_noise\.switch\..*|imu\.ds\.mode|"
                    r"voxel_map\.plane\.(plane_var_min|plane_var_scan_s2|plane_var_diag_en|plane_gap_floor_en|debiased_line_equiv|plane_fit_mode)|voxel_map\.plane\.line_residual\.(enabled|form|rule|rows|classify_at_fit)|"
                    r"lio\.time_cov\..*|lio\.zupt\.(mode|threshold_mode|calib_.*|enter_scans|sigma_v)|evo\.dense_gt_metrics|calib\.stationary\.num_samples|data_queues\.log_lidar_msgs)$")
THEMES = [("OVLCFG", re.compile(r"^cbk\.lidar\.(overlap_policy|retime_.*)$")), ("MSGKEY", re.compile(r"^data_queues\.log_lidar_msgs$")), ("SWXCFG", re.compile(r"^imu\.process_noise\.switch\..*$")),
          ("VMINCFG", re.compile(r"^voxel_map\.plane\.plane_var_min$")), ("VS2CFG", re.compile(r"^voxel_map\.plane\.plane_var_scan_s2$")), ("PVDCFG", re.compile(r"^voxel_map\.plane\.plane_var_diag_en$")),
          ("D4CFG", re.compile(r"^voxel_map\.plane\.(line_residual\.(classify_at_fit|enabled|rows)|debiased_line_equiv|plane_gap_floor_en|plane_fit_mode)$")), ("TCCFG", re.compile(r"^lio\.time_cov\..*$")),
          ("ZCCFG", re.compile(r"^lio\.zupt\..*$")), ("PTCFG", re.compile(r"^cbk\.lidar\.point_time_.*$")), ("DGTCFG", re.compile(r"^evo\.dense_gt_metrics$")), ("DSCFG", re.compile(r"^imu\.ds\.mode$")),
          ("PFN", re.compile(r"^cbk\.lidar\.point_filter_num$")), ("VOX", re.compile(r"^voxel_map\.map\.(voxel_size|max_layer)$"))]
DEFAULTS = {"voxel_map.plane.plane_var_min": 0.0, "voxel_map.plane.plane_var_scan_s2": 0.0, "voxel_map.plane.plane_var_diag_en": False, "voxel_map.plane.line_residual.classify_at_fit": False,
            "voxel_map.plane.plane_gap_floor_en": False, "lio.time_cov.mode": "off", "lio.zupt.threshold_mode": "fixed", "evo.dense_gt_metrics": False, "cbk.lidar.overlap_policy": "discard",
            "voxel_map.plane.line_residual.enabled": False}   # a key the job does not set is at its compiled default; a build that prints it must print the default
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
    out += ["", "PASS %d  FAIL %d  MISSING %d  INFO %d" % tuple(sum(1 for r in g.rows if r[1] == k) for k in ("PASS", "FAIL", "MISSING", "INFO"))]
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
        table.append(dict(id=r["arm"], group=r.get("group", ""), fam=r.get("fam", ""), mode=r.get("mode", ""), kind=r.get("arm_kind") or r.get("kind", ""), ref=r.get("ref") or None, bag=r.get("bag") or r.get("seq"),
                          batch=r.get("batch", ""), pfn=int(r["pfn"]) if (r.get("pfn") or "").isdigit() else None, dur=None if dur in ("full", "", None) else float(dur), msglog=r.get("msglog") == "1",
                          needles=[x for x in (r.get("needles") or "").split("|") if x], flag=r.get("flag", ""), ov=j.get("overrides"), has_job=r["arm"] in jobs))
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


# ------------------------------------------------------------------------------------------------ per-arm gates
def theme_of(key):
    for nm, rx in THEMES:
        if rx.match(key):
            return nm
    return "CFG"


def cfg_gates(t, ov, log, et):
    """effective-configuration rows of ONE arm from its own log text. -> list of (name, status, detail)."""
    arm, out = t["id"], []
    have = bool(log.strip())
    idx = parse_log(log)
    want = {k: v for k, v in (ov or {}).items() if v is not None and not SKIP_KEYS.match(k) and isinstance(v, (bool, int, float, str))}
    for k, dv in DEFAULTS.items():
        want.setdefault(k, dv)
    if t["mode"] == "dc" and "cbk.lidar.overlap_policy" in (ov or {}):
        out.append(("OVLCFG %s dc arm must not carry cbk/lidar/overlap_policy in its job" % arm, "FAIL", "job sets %s" % ov["cbk.lidar.overlap_policy"]))
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
        out.append(("%s %s effective config: %d key(s) against the job's overrides and the defaults of the new keys" % (th, arm, len(items)), worst(sts), "; ".join(bad) if bad else "; ".join(d for _, s, d in items if s == "PASS")[:300]))
    # overlap policy engagement
    pol = line(log, "[lidar_overlap_policy]")
    if t["mode"] == "rt":
        okp = bool(re.search(r"\[lidar_overlap_policy\]\s+(?:retime_trim\s+)?policy=retime_trim", log)) or bool(pol and "retime_trim" in pol)
        inert = bool(re.search(r"\[lidar_overlap_policy\].*inert when on", log))
        out.append(("OVLENG %s rt arm prints [lidar_overlap_policy] policy=retime_trim and no 'inert when on' error" % arm, "FAIL" if inert or (have and not okp) else ("PASS" if okp else "MISSING"),
                    "line: %s" % (pol or "none")))
    elif t["mode"] == "dc":
        out.append(("OVLENG %s dc arm prints no [lidar_overlap_policy] line (the discard path prints what R80 printed)" % arm, "FAIL" if pol else ("PASS" if have else "MISSING"), "line: %s" % (pol or "none")))
    return out


def rtgrid_gate(g, t, grid_retime, grid_bag):
    """RTGRID: the job's retime keys equal the bag's grid (the R80D make() rule: eight bags from r80c_grid.json exactly as shipped, eee_03 from the bag fit) and the policy is retime_trim with ref offset -0.1."""
    arm, ov = t["id"], t["ov"] or {}
    nm = "RTGRID %s job's retime keys equal the grid of %s (t0_s, period_s, seq0 exactly, ref offset %.1f) and the policy is retime_trim" % (arm, t["bag"], REF_OFFSET)
    gr = grid_of(t["bag"], grid_retime, grid_bag)
    if gr is None:
        g.add(nm, "MISSING", "no grid for %s (--grid-retime / --grid-bag)" % t["bag"]); return
    if t["ov"] is None:
        g.add(nm, "MISSING", "no job for the arm in jobs_*.json"); return
    bad = []
    for k, gv in (("cbk.lidar.retime_t0_s", gr["t0_s"]), ("cbk.lidar.retime_period_s", gr["period_s"]), ("cbk.lidar.retime_seq0", gr["seq0"]), ("cbk.lidar.retime_ref_offset_s", REF_OFFSET)):
        if k not in ov:
            bad.append("%s absent" % k)
        elif float(ov[k]) != float(gv):
            bad.append("%s = %r, grid %r" % (k, ov[k], gv))
    if ov.get("cbk.lidar.overlap_policy") != "retime_trim":
        bad.append("overlap_policy %r" % ov.get("overlap_policy", ov.get("cbk.lidar.overlap_policy")))
    g.add(nm, "FAIL" if bad else "PASS", "; ".join(bad) if bad else "t0 %.7f period %.12f seq0 %s" % (gr["t0_s"], gr["period_s"], gr["seq0"]))


def len_gate(g, t, P):
    arm = t["id"]
    n = None if P is None else len(P)
    if t["bag"] == "hilti" and t["dur"] is None:
        g.add("LEN %s Hilti whole bag: rows in the pos trace (a fact; the planner holds no Hilti scan counts)" % arm, "MISSING" if P is None else "INFO", "rows %s" % n)
        return
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
        g.add("MSGSCANS %s accepted messages >= scans in the pos trace" % arm, "PASS" if na >= len(P) else "FAIL", "accepted %d, scans %d" % (na, len(P)))
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


def engagement_gates(g, t, ov, log, et, d):
    arm = t["id"]; ov = ov or {}; out = []
    have = bool(et.strip()) or bool(log.strip())
    smoke = t["kind"] == "smoke"
    # ---- ZUPT (D6, F-508): calib_median
    zl = line(et, "lio/zupt mode=")
    zmode = ov.get("lio.zupt.mode")
    if zmode:
        z = kv(zl); sc = z.get("scans")
        items = [("mode", z.get("mode"), zmode), ("enter_scans", z.get("enter_scans"), ov.get("lio.zupt.enter_scans")), ("evaluated (scans > 0)", None if sc is None else ("yes" if int(sc) > 0 else "no"), "yes")]
        if ov.get("lio.zupt.threshold_mode") == "calib_median":
            items += [("threshold_mode", z.get("threshold_mode"), "calib_median"), ("calib_scans", z.get("calib_scans"), ov.get("lio.zupt.calib_scans")), ("calibrated", z.get("calibrated"), 1)]
        else:
            items += [("gyro_thresh", z.get("gyro_thresh"), ov.get("lio.zupt.gyro_thresh")), ("acc_thresh", z.get("acc_thresh"), ov.get("lio.zupt.acc_thresh"))]
        if zmode == "apply":
            items.append(("applied > 0 (apply mode, a smoke arm must engage)", None if z.get("applied") is None else ("yes" if int(z["applied"]) > 0 else "no"), "yes" if smoke else None))
        items = [(a, b, c) for a, b, c in items if c is not None]
        bad = ["%s: found %s, wanted %s" % (a, b, c) for a, b, c in items if b is not None and not val_match(b, c, a)]
        miss = ["%s: not found" % a for a, b, c in items if b is None]
        out.append(("ZCENG %s lio/zupt engagement line as the job states it (calib_median: threshold_mode, calib_scans, calibrated = 1)" % arm, "FAIL" if bad else ("MISSING" if miss or not zl else "PASS"), "; ".join(bad + miss) or zl.strip()[:300]))
        if zl and "calibrated=1" in zl:
            out.append(("ZCINFO %s per-bag thresholds the build calibrated (a fact)" % arm, "INFO", " ".join(re.findall(r"(?:median_gyro|median_acc|thr_gyro|thr_acc)=\S+", zl))))
    # ---- time_cov (D1, F-501)
    tl = _eng_line(et, r"lio/time_cov mode=")
    if ov.get("lio.time_cov.mode") == "shared":
        z = kv(tl or ""); bad = []
        if tl is None:
            bad.append("no lio/time_cov engagement line although mode = shared")
        else:
            if z.get("mode") != "shared":
                bad.append("mode %s" % z.get("mode"))
            if z.get("sigma_s") is not None and not val_match(z["sigma_s"], ov.get("lio.time_cov.sigma_s"), "sigma_s"):
                bad.append("sigma_s %s, wanted %s" % (z["sigma_s"], ov.get("lio.time_cov.sigma_s")))
            if z.get("applied") is not None and int(float(z["applied"])) <= 0:
                bad.append("applied = %s (the correction never applied)" % z["applied"])
            if "<-- INERT" in tl:
                bad.append("the build printed INERT")
        out.append(("TCENG %s lio/time_cov engagement line (mode shared, sigma_s as the job, applied > 0, not INERT)" % arm, "FAIL" if bad else ("MISSING" if tl is None and not have else "PASS"), "; ".join(bad) or (tl or "").strip()[:260]))
    # ---- plane variance keys and the diagnostic (D2, F-502)
    vl = next((l for l in et.splitlines() if re.search(r"plane_var_min\s*=.*\brows\s*=", l)), None)
    vmin, vs2 = ov.get("voxel_map.plane.plane_var_min", 0.0), ov.get("voxel_map.plane.plane_var_scan_s2", 0.0)
    von = (vmin or 0) > 0 or (vs2 or 0) > 0
    if von:
        m = re.search(r"clamped\s*=\s*(\d+)", vl or ""); clamped = int(m.group(1)) if m else None
        rowsm = re.search(r"\brows\s*=\s*(\d+)", vl or ""); nr = int(rowsm.group(1)) if rowsm else None
        bad = []
        if vl is None:
            bad.append("no plane_var engagement line although a key is on")
        else:
            if nr is not None and nr <= 0:
                bad.append("rows = %d (the term never evaluated)" % nr)
            if smoke and (vmin or 0) > 0 and (clamped is None or clamped <= 0):
                bad.append("smoke arm: clamped = %s (inert when on)" % clamped)
        tag = "VS2ENG" if (vs2 or 0) > 0 and not (vmin or 0) > 0 else "VMINENG"
        out.append(("%s %s plane variance engagement line (rows > 0%s)" % (tag, arm, "; smoke arm: clamp fired" if smoke else ""), "FAIL" if bad else ("MISSING" if vl is None and not have else "PASS"), "; ".join(bad) or (vl or "").strip()[:240]))
        if vl is not None and clamped is not None and not smoke:
            out.append(("VMININFO %s clamped rows %s (a fact)" % (arm, clamped), "INFO", (vl or "").strip()[:240]))
    dl = _eng_line(et, r"plane_var_diag_en rows=")
    if ov.get("voxel_map.plane.plane_var_diag_en"):
        m = re.search(r"rows=(\d+)", dl or ""); nr = int(m.group(1)) if m else None
        bad = []
        if dl is None:
            bad.append("no plane_var_diag_en engagement block")
        else:
            if nr is not None and nr <= 0:
                bad.append("rows = %d" % nr)
            if "<-- INERT" in dl:
                bad.append("the build printed INERT")
        out.append(("PVDENG %s plane_var_diag_en engagement block (rows > 0, not INERT)" % arm, "FAIL" if bad else ("MISSING" if dl is None and not have else "PASS"), "; ".join(bad) or dl.strip()[:300]))
    # ---- D4: classify_at_fit, debiased_line_equiv, gap floor (counters)
    cl = _eng_line(et, r"line_residual/classify_at_fit\s*=.*line_classified=")
    cls_on = bool(ov.get("voxel_map.plane.line_residual.classify_at_fit"))
    eq = ov.get("voxel_map.plane.debiased_line_equiv")
    fl_deb = bool(ov.get("voxel_map.plane.plane_gap_floor_en")) and ov.get("voxel_map.plane.plane_fit_mode") == "debiased"
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
    # ---- point time (F-504)
    pl = _eng_line(et, r"cbk/lidar/point_time:\s*field=")
    pt_keys = [k for k in ov if k.startswith("cbk.lidar.point_time_")]
    if pt_keys:
        z = kv(pl or ""); bad = []
        if pl is None:
            bad.append("no cbk/lidar/point_time engagement line although a point_time key is set")
        else:
            for k, f in (("cbk.lidar.point_time_field", "field"), ("cbk.lidar.point_time_mode", "mode"), ("cbk.lidar.point_time_ref", "ref")):
                if k in ov and z.get(f) is not None and not val_match(z[f], ov[k], f):
                    bad.append("%s %s, wanted %s" % (f, z[f], ov[k]))
        out.append(("PTENG %s cbk/lidar/point_time engagement line as the job states it" % arm, "FAIL" if bad else ("MISSING" if pl is None and not have else "PASS"), "; ".join(bad) or (pl or "").strip()[:240]))
    # ---- dense ground truth (F-506)
    if ov.get("evo.dense_gt_metrics"):
        el = _eng_line(et, r"evo/dense_gt_metrics:\s*true")
        ev = _eng_line(et, r"\[evo/[^\]]*\].*\bARE=")
        bad = []
        if el is None:
            bad.append("no evo/dense_gt_metrics engagement line")
        if smoke and (ev is None or "ATE_first=" not in ev or "ARE_first=" not in ev):
            bad.append("smoke arm: the [evo] line has no ARE / ATE_first / ARE_first")
        out.append(("DGTENG %s evo/dense_gt_metrics engaged%s" % (arm, " and the [evo] line carries ARE, ATE_first, ARE_first" if smoke else ""), "FAIL" if bad else ("MISSING" if el is None and not have else "PASS"), "; ".join(bad) or (ev or el or "").strip()[:300]))
    # ---- nothing the arm did not ask for
    extra = []
    if zl and not zmode:
        extra.append("ZUPT")
    if tl and ov.get("lio.time_cov.mode") != "shared":
        extra.append("time_cov")
    if vl and not von:
        extra.append("plane variance keys")
    if dl and not ov.get("voxel_map.plane.plane_var_diag_en"):
        extra.append("plane_var_diag")
    if cl and not (cls_on or eq or fl_deb):
        extra.append("D4 keys")
    if pl and not pt_keys:
        extra.append("point_time")
    out.append(("NOEXTRA %s no ZUPT, time_cov, plane-variance, D4 or point-time engagement line that the arm did not ask for" % arm, "FAIL" if extra else ("PASS" if have else "MISSING"), ", ".join(extra)))
    return out


ZC_COLS = ["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"]
TC_COLS = ["omega_w_norm", "omega_fd_norm", "omega_diff_norm"]


def file_gates(g, t, d, ov):
    arm, fam = t["id"], t["fam"]
    if t["group"] == "D6" or arm == "r82_dc_sm_zc_60s":
        Z = csvf(d, "zupt_all.csv.gz")
        nm = "ZCFILE %s zupt_all.csv.gz with columns %s, at least 100 rows, some scans stationary" % (arm, ", ".join(ZC_COLS))
        if Z is None:
            g.add(nm, "MISSING")
        elif [c for c in ZC_COLS if c not in Z.columns]:
            g.add(nm, "FAIL", "missing columns %s" % [c for c in ZC_COLS if c not in Z.columns])
        else:
            st = pd.to_numeric(Z["stationary"], errors="coerce")
            g.add(nm, "PASS" if len(Z) >= 100 and int((st == 1).sum()) > 0 else "FAIL", "rows %d, stationary %d, applied %d" % (len(Z), int((st == 1).sum()), int((pd.to_numeric(Z["applied"], errors="coerce") != 0).sum())))
    if arm == "r82_dc_sm_tc_60s":
        T = csvf(d, "time_cov.csv.gz") if csvf(d, "time_cov.csv.gz") is not None else csvf(d, "time_cov.csv")
        nm = "TCFILE %s time_cov.csv with columns %s and at least 20 rows (the frame check is read by the planner, not judged here)" % (arm, ", ".join(TC_COLS))
        if T is None:
            g.add(nm, "MISSING")
        elif [c for c in TC_COLS if c not in T.columns]:
            g.add(nm, "FAIL", "missing columns %s" % [c for c in TC_COLS if c not in T.columns])
        else:
            g.add(nm, "PASS" if len(T) >= 20 else "FAIL", "rows %d" % len(T))
            w, dd = pd.to_numeric(T["omega_w_norm"], errors="coerce"), pd.to_numeric(T["omega_diff_norm"], errors="coerce")
            g.add("TCFACT %s omega_w_norm median %.5g, omega_diff_norm median %.5g rad/s (a fact)" % (arm, w.median(), dd.median()), "INFO")


def hilti_table_gates(g, table, live, ret_dirs, runs):
    dirs = list(ret_dirs) + [runs, os.path.join(runs, "..")]
    hil = [t for t in table if t["group"] == "HIL" and t["id"] in live]
    if not hil:
        return
    p = find_return_file("r82_hilti_metrics.csv", dirs)
    H = pd.read_csv(p) if p else None
    if H is None or "arm" not in H.columns:
        g.add("HILCOMP r82_hilti_metrics.csv has a row for each of the %d Hilti arms" % len(hil), "MISSING", "file absent or without an arm column")
        return
    have = set(H["arm"])
    miss = [t["id"] for t in hil if t["id"] not in have]
    g.add("HILCOMP r82_hilti_metrics.csv has a row for each of the %d dispatched Hilti arms" % len(hil), "PASS" if not miss else "FAIL", "missing: %s" % ", ".join(miss[:12]))
    for c in ("ate_m", "are_deg", "ate_first_m", "are_first_deg"):
        if c in H.columns:
            bad = int((pd.to_numeric(H[c], errors="coerce").isna()).sum())
            g.add("HILCOMP column %s numeric in every row (a NaN means the metric was not printed)" % c, "PASS" if bad == 0 else "FAIL", "%d non-numeric rows" % bad)
        else:
            g.add("HILCOMP column %s present" % c, "FAIL", "absent")


def ref_gates(g, t, d, present, sent, ex):
    arm = t["id"]
    nm = "REFUSE %s dies at startup, text contains %s" % (arm, " and ".join(repr(n) for n in t["needles"]))
    if not present:
        g.add(nm, "MISSING", "arm.json absent")
    else:
        tail = "".join(rd(d, f) or "" for f in ("run_log_tail.txt", "run.log"))
        has = [n in tail for n in t["needles"]]
        g.add(nm, "PASS" if sent == "FAILED" and all(has) else "FAIL", "sentinel=%s needles_found=%s" % (sent, has))
    nm = "EXIT %s exit_status.txt exists and is non-zero, and the job did not reach DONE (the deliberately refused job of the F-326 check)" % arm
    if ex is None:
        g.add(nm, "MISSING", "exit_status.txt absent")
    elif ex == "bad":
        g.add(nm, "FAIL", "first line of exit_status.txt is not `exit_code=<N> signal=<NAME|none>`")
    elif not present:
        g.add(nm, "MISSING", "exit status %s but arm.json absent: cannot tell whether the job reached DONE" % (ex,))
    else:
        g.add(nm, "PASS" if ex[0] != 0 and sent != "DONE" else "FAIL", "exit status %s signal %s, sentinel %s" % (ex[0], ex[1], sent))


class SwAgg:
    """F498 arms: FAIL / MISSING rows stay one per arm; the PASS rows of a (gate, cell) fold into one row (the report stays inside its size ledger)."""

    def __init__(self, g):
        self.g, self.c = g, {}

    def put(self, kind, cell, name, st, det=""):
        if st != "PASS":
            self.g.add(name, st, det)
        self.c.setdefault((kind, cell), {"PASS": 0, "FAIL": 0, "MISSING": 0})[st] += 1

    def flush(self, n):
        for (kind, cell), c in self.c.items():
            if c["FAIL"] == 0 and c["MISSING"] == 0 and c["PASS"] == n:
                self.g.add("SWCELL %s %s: all %d arms PASS" % (kind, cell, n), "PASS")
            elif c["PASS"] > 0:
                self.g.add("SWCELL %s %s: arms PASS / FAIL / MISSING of the %d of the cell (the FAIL and MISSING arms have their own rows)" % (kind, cell, n), "INFO", "%d / %d / %d" % (c["PASS"], c["FAIL"], c["MISSING"]))


# ------------------------------------------------------------------------------------------------ the gate
def find_return_file(name, dirs):
    for d in dirs:
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def run_gates(runs, manifest, grid_retime=None, grid_bag=None, ref_returns=(), ref_runs=(), phase=None, dispatched=None, ret_dirs=(), expect=EXPECT, ref_manifests=(), tol_mm=IH.TOL_MM):
    g = G()
    table = load_manifest(manifest)
    A = lambda a: os.path.join(runs, a)
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    disp = dispatched_set(dispatched, table) if (dispatched is None or isinstance(dispatched, str)) else set(dispatched)
    live = {t["id"] for t in table if disp is None or t["id"] in disp}
    queues = []
    for t in table:
        if t["batch"] not in queues:
            queues.append(t["batch"])
    for q in queues:
        if not any(t["id"] in live for t in table if t["batch"] == q):
            g.add("NOTRUN queue %s: no arm of it was dispatched; no other gate is affected" % q, "INFO")
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
    id_rows, sw = [], SwAgg(g)
    by_id = {t["id"]: t for t in table}
    P_cache = {}
    popen = lambda a: P_cache.setdefault(a, csvf(A(a), "pos_trace_all.csv.gz"))
    swcells = {}
    for t in table:
        if t["id"] not in live:
            continue
        arm, grp, d = t["id"], t["group"], A(t["id"])
        is_sw = grp == "F498"
        if is_sw:
            m = SWT_RE.match(arm)
            cell = "sw_%s_%s_%s_p%s" % (m.group("bag"), m.group("q"), m.group("line"), m.group("pfn")) if m else "sw_unparsed"
            swcells.setdefault(cell, []).append(arm)
        put = (lambda kind, name, st, det="": sw.put(kind, cell, name, st, det)) if is_sw else (lambda kind, name, st, det="": g.add(name, st, det))
        pres = present(arm)
        sent = (jsn(d, "arm.json") or {}).get("sentinel") if pres else None
        ex = exit_status(d) if os.path.isdir(d) else None
        if t["kind"] == "refuse":
            ref_gates(g, t, d, pres, sent, ex)
            continue
        put("RUN", "RUN %s completed (arm.json sentinel DONE)" % arm, "MISSING" if not pres else ("PASS" if sent == "DONE" else "FAIL"), "" if not pres else "sentinel=%s" % sent)
        nm = "EXIT %s exit_status.txt `exit_code=<N> signal=<NAME|none>` with N = 0" % arm
        if ex is None:
            put("EXIT", nm, "MISSING", "exit_status.txt absent")
        elif ex == "bad":
            put("EXIT", nm, "FAIL", "first line is not `exit_code=<N> signal=<NAME|none>` nor an integer")
        else:
            put("EXIT", nm, "PASS" if ex[0] == 0 else "FAIL", "exit status %s signal %s" % ex)
        if not pres:
            continue
        log = rd(d, "run.log") or ""
        et = (rd(d, "engagement.txt") or "") + "\n" + log
        if t["ov"] is None:
            put("CFG", "CFG %s effective config" % arm, "MISSING", "no job for the arm in jobs_*.json")
        else:
            cg = cfg_gates(t, t["ov"], log, et)
            if is_sw:
                sts = [s for n, s, dt in cg if s != "INFO"]
                bad = ["%s: %s" % (n.split(" ")[0], dt) for n, s, dt in cg if s in ("FAIL", "MISSING")]
                put("CFG", "SWCFG %s effective config: edge, lines, rows, point filter, plane fit, overlap policy, nothing extra" % arm, worst(sts) if sts else "MISSING", "; ".join(bad))
            else:
                for n, s, dt in cg:
                    g.add(n, s, dt)
        if t["mode"] == "rt":
            if is_sw:
                tmp = G(); rtgrid_gate(tmp, t, grid_retime, grid_bag)
                put("RTGRID", "RTGRID %s job's retime keys equal the bag grid" % arm, tmp.rows[0][1], tmp.rows[0][2])
            else:
                rtgrid_gate(g, t, grid_retime, grid_bag)
        if is_sw:
            continue                    # F498: summaries only; no pos_trace is required (a trace only when gt_max_mm > 500), no engagement gates
        for n, s, dt in engagement_gates(g, t, t["ov"], log, et, d):
            g.add(n, s, dt)
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
        file_gates(g, t, d, t["ov"])
    sw.flush(len(EDGES_SWT))
    # ---- sweep completeness: summaries only
    Pn = phase if isinstance(phase, pd.DataFrame) else (pd.read_csv(phase) if phase and os.path.exists(phase) else None)
    have = set(Pn["arm"]) if (Pn is not None and "arm" in Pn.columns) else None
    for cell, arms_ in sorted(swcells.items()):
        if have is None:
            g.add("SWCOMP %s phase_summary.csv rows of the %d arms of the cell" % (cell, len(arms_)), "MISSING", "phase_summary.csv absent or without an arm column"); continue
        miss = [a for a in arms_ if a not in have]
        g.add("SWCOMP %s arms with a phase_summary row (a fact)" % cell, "INFO", "%d of %d" % (len(arms_) - len(miss), len(arms_)))
        for a in miss:
            g.add("SWCOMP %s has a phase_summary row" % a, "MISSING")
        if not miss and len(arms_) == len(EDGES_SWT):
            g.add("SWCOMP %s all %d arms have a phase_summary row" % (cell, len(arms_)), "PASS")
        elif len(arms_) != len(EDGES_SWT):
            g.add("SWCOMP %s dispatched arms of the cell: %d of the %d edges" % (cell, len(arms_), len(EDGES_SWT)), "INFO")
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
        return [("per-scan position, excursion, settled offset and digest equal %s" % t["ref"], "MISSING", "no --ref-return given: the reference returns (R80, R80C) are required for the identity rows")], None
    r = IH.check_arm(t["id"], t["ref"], runs, list(ref_returns), list(ref_runs), [manifest], list(ref_manifests), tol_mm)
    return [(n, s, dt) for n, s, dt in r["rows"] if not (n.startswith("POS yaw") and False)], r


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
    """HILCOMP, D5COMP, PFT: group-level completeness from files of the return (found in ret_dirs)."""
    dirs = list(ret_dirs) + [runs, os.path.join(runs, "..")]
    arms = lambda f: [t for t in table if f(t) and t["id"] in live]
    hilti_table_gates(g, table, live, ret_dirs, runs)
    d5 = arms(lambda t: t["group"] == "D5")
    if d5:
        p = find_return_file("r82_switch_pairs.csv", dirs)
        if p:
            S = pd.read_csv(p)
            have = set(S["arm"]) if "arm" in S.columns else set()
            miss = [t["id"] for t in d5 if t["id"] not in have]
            g.add("D5COMP r82_switch_pairs.csv has a row for every one of the %d switch arms" % len(d5), "PASS" if not miss else "FAIL", "missing: %s" % ", ".join(miss[:12]))
        else:
            g.add("D5COMP r82_switch_pairs.csv has a row for every switch arm", "MISSING", "file not found in the return files")
    pft = arms(lambda t: t["group"] == "D4" or (t["fam"] == "SMK" and "cls" in t["id"]))
    if pft:
        p = find_return_file("plane_fit_trace_counts_all.csv", dirs)
        C = pd.read_csv(p) if p else None
        if C is None or "arm" not in C.columns:
            g.add("PFT plane_fit_trace_counts_all.csv has rows for the %d D4 and D4-smoke arms (written by the collector)" % len(pft), "MISSING", "file absent")
        else:
            bad = 0
            for t in pft:
                n = int(C[C["arm"] == t["id"]]["rows"].sum()) if "rows" in C.columns else 0
                if n <= 0:
                    bad += 1; g.add("PFT %s has rows in plane_fit_trace_counts_all.csv" % t["id"], "MISSING", "no rows")
            if not bad:
                g.add("PFT all %d D4 and D4-smoke arms have rows in plane_fit_trace_counts_all.csv" % len(pft), "PASS")


def write_hold(g, hold_file):
    bad = [(n, s, d) for n, s, d in g.rows if s == "FAIL" and (n.startswith("DET ") or n.startswith("DETPOS ") or n.startswith("IDENT "))]
    if not bad:
        return False
    import time
    body = ["HOLD (from r81_gates.py, %s): a determinism or identity gate FAILED. Do not dispatch the unstarted queues; let running jobs finish; cancel nothing; delete nothing; change nothing in the tree." % time.strftime("%Y-%m-%d %H:%M:%S"),
            "Investigate (config dump diff, run.log, first-divergence scan, tree diff), report to the planner. The planner's written word lifts the hold (rename HOLD.txt to HOLD.released_<date>.txt; never delete it).", ""]
    body += ["%s   [%s]" % (n, d) for n, s, d in bad]
    with open(hold_file, "a" if os.path.exists(hold_file) else "w") as f:
        f.write("\n".join(body) + "\n")
    return True


# ------------------------------------------------------------------------------------------------ selftest fixtures
def selftest():
    """Synthetic: the manifest of make_r82_manifest.py on a stub job list, a few fabricated arm folders. Planner sandbox, not evidence."""
    import subprocess, tempfile
    root = tempfile.mkdtemp(prefix="r82_gates_selftest_")
    mdir = os.path.join(root, "m")
    subprocess.check_call([sys.executable, os.path.join(HERE, "make_r82_manifest.py"), "--selftest"], stdout=subprocess.DEVNULL)
    sys.path.insert(0, HERE)
    import make_r82_manifest as MM
    J = MM.R81.stub_jobs(); gr, gb = MM.R81.stub_grids()
    MM.build(None, J, gr, gb, {b: 300 for b in MM.BAGS}, mdir, "/r", quiet=True)
    fails = []
    ck = lambda c, m: fails.append(m) if not c else None
    table = load_manifest(mdir)
    by = {t["id"]: t for t in table}
    # nothing present: every arm MISSING, no crash, FAMCOUNT PASS
    runs = os.path.join(root, "runs"); os.makedirs(runs)
    g, _ = run_gates(runs, mdir, gr, gb, dispatched="r82_dc_refuse_time_cov_no_sigma,r82_dc_sm_tc_60s,r82_dc_sm_zc_60s,r82_rt_d4_eee_01_dbcx_fq,r82_dc_sm_hl_exp04_60s")
    st = {n: s_ for n, s_, _d in g.rows}
    ck(all(s_ == "PASS" for n, s_ in st.items() if n.startswith("FAMCOUNT") and "manifest holds" in n), "FAMCOUNT must PASS for every group")
    ck(any(s_ == "MISSING" for n, s_ in st.items() if n.startswith("RUN ")), "an absent arm must be MISSING")
    # fabricate the refuse arm, the tc smoke arm and the D4 arm
    def mkarm(arm, sentinel="DONE", exit_="exit_code=0 signal=none", log="", eng=""):
        d = os.path.join(runs, arm); os.makedirs(d, exist_ok=True)
        json.dump({"sentinel": sentinel}, open(os.path.join(d, "arm.json"), "w")); open(os.path.join(d, "exit_status.txt"), "w").write(exit_ + "\n")
        open(os.path.join(d, "run.log"), "w").write(log); open(os.path.join(d, "engagement.txt"), "w").write(eng)
    mkarm("r82_dc_refuse_time_cov_no_sigma", "FAILED", "exit_code=134 signal=SIGABRT", log="terminate: lio/time_cov/sigma_s must be in (0, 0.1] seconds")
    mkarm("r82_dc_sm_tc_60s", log="", eng="  lio/time_cov mode=shared sigma_s=0.002 iterations=5000 applied=4000 mean_s_factor=1.2 max_s_factor=9 mean_rms_g=0.3\n")
    mkarm("r82_rt_d4_eee_01_dbcx_fq", eng="  voxel_map/plane/line_residual/classify_at_fit = true debiased_line_equiv = moment plane_gap_floor_en = true line_classified=10 line_accepted=9 e1_tiny_kept=1 e0_clamped=2 gaps_floored=5 tilt_voxels_moment=7 tilt_voxels_meancov=0\n")
    g, _ = run_gates(runs, mdir, gr, gb, dispatched="r82_dc_refuse_time_cov_no_sigma,r82_dc_sm_tc_60s,r82_rt_d4_eee_01_dbcx_fq")
    rows = {n: s_ for n, s_, _d in g.rows}
    ck(rows.get("REFUSE r82_dc_refuse_time_cov_no_sigma dies at startup, text contains 'lio/time_cov/sigma_s'") == "PASS", "refuse arm must PASS: %s" % [n for n in rows if n.startswith("REFUSE")])
    ck(any(n.startswith("TCENG r82_dc_sm_tc_60s") and s_ == "PASS" for n, s_ in rows.items()), "TCENG must PASS on a good line")
    ck(any(n.startswith("D4ENG r82_rt_d4_eee_01_dbcx_fq") and s_ == "PASS" for n, s_ in rows.items()), "D4ENG must PASS on a good line")
    ck(any(n.startswith("TCFILE") and s_ == "MISSING" for n, s_ in rows.items()), "TCFILE must be MISSING without time_cov.csv")
    # a time_cov line with applied=0 must FAIL; a D4 smoke arm with line_classified=0 must FAIL
    mkarm("r82_dc_sm_tc_60s", eng="  lio/time_cov mode=shared sigma_s=0.002 iterations=5000 applied=0 mean_s_factor=0 max_s_factor=0 mean_rms_g=0   <-- INERT: time_cov selected and no correction was applied\n")
    mkarm("r82_dc_sm_cls_pca_60s", eng="  voxel_map/plane/line_residual/classify_at_fit = true debiased_line_equiv = off plane_gap_floor_en = true line_classified=0 line_accepted=0 e1_tiny_kept=0 e0_clamped=0 gaps_floored=0 tilt_voxels_moment=0 tilt_voxels_meancov=0   <-- INERT: classify_at_fit was set and no fit was classified a line\n")
    g, _ = run_gates(runs, mdir, gr, gb, dispatched="r82_dc_sm_tc_60s,r82_dc_sm_cls_pca_60s")
    rows = {n: s_ for n, s_, _d in g.rows}
    ck(any(n.startswith("TCENG r82_dc_sm_tc_60s") and s_ == "FAIL" for n, s_ in rows.items()), "TCENG must FAIL on applied=0 / INERT")
    ck(any(n.startswith("D4ENG r82_dc_sm_cls_pca_60s") and s_ == "FAIL" for n, s_ in rows.items()), "D4ENG must FAIL on a smoke arm with line_classified = 0")
    # a stray engagement line the arm did not ask for
    mkarm("r82_dc_ds_eee_01_first_fq", eng="  lio/time_cov mode=shared sigma_s=0.002 iterations=1 applied=1\n")
    g, _ = run_gates(runs, mdir, gr, gb, dispatched="r82_dc_ds_eee_01_first_fq")
    ck(any(n.startswith("NOEXTRA r82_dc_ds_eee_01_first_fq") and s_ == "FAIL" for n, s_, _d in g.rows), "NOEXTRA must FAIL on an engagement line the arm did not ask for")
    print("SELFTEST r82_gates", "FAIL " + "; ".join(fails) if fails else "PASS (synthetic fixtures; planner sandbox, not evidence)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--manifest", default="r82_manifest"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grid-retime", default=None); ap.add_argument("--grid-bag", default=None); ap.add_argument("--ref-return", action="append", default=[]); ap.add_argument("--ref-runs", action="append", default=[])
    ap.add_argument("--ref-manifest", action="append", default=[]); ap.add_argument("--phase-summary", default=None); ap.add_argument("--ret-dir", action="append", default=[])
    ap.add_argument("--dispatched", default=None); ap.add_argument("--hold-file", default=None); ap.add_argument("--no-famcount", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runs:
        sys.exit("--runs is required")
    gb = a.grid_bag or os.path.join(a.manifest, "grid_bag.json")
    grid_retime = json.load(open(a.grid_retime)) if a.grid_retime and os.path.exists(a.grid_retime) else None
    grid_bag = json.load(open(gb)) if os.path.exists(gb) else None
    ph = a.phase_summary
    rets = list(a.ret_dir) + ([os.path.dirname(os.path.abspath(a.out))] if a.out else [])
    if ph is None:
        ph = next((c for c in [os.path.join(d, "phase_summary.csv") for d in rets] + [os.path.join(a.runs, "phase_summary.csv"), os.path.join(a.runs, "..", "phase_summary.csv")] if os.path.exists(c)), None)
    g, id_rows = run_gates(a.runs, a.manifest, grid_retime, grid_bag, a.ref_return, a.ref_runs, ph, a.dispatched, rets, {} if a.no_famcount else EXPECT, a.ref_manifest)
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
