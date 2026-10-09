#!/usr/bin/env python3
"""R83 gates (planning agent's script, derived from r82_gates.py; selftest on SYNTHETIC fixtures only, planner sandbox, not evidence).

    python3 tools/r69/r83_gates.py --runs <runs_dir> --manifest r83_manifest [--grid-retime r80c_grid.json] [--grid-bag r83_manifest/grid_bag.json] \
        [--ref-return R82_RETURN] [--ref-runs <old archive dir>] [--phase-summary phase_summary.csv] [--ret-dir <return dir>] [--expect "DST=12,MXL=36,..."] \
        [--dispatched <arm ids file | arm_table.tsv | comma list of arm ids / groups / families / queues>] [--out r83_gates.txt] [--csv r83_identity_vs_old.csv] [--hold-file HOLD_FROM_GATES.txt]
    python3 tools/r69/r83_gates.py --selftest

Same row vocabulary and verdicts as r82_gates.py (PASS, FAIL, MISSING (never a pass), INFO, NOTRUN). The expectation of every arm is read from the MANIFEST ONLY (arm_table.tsv + jobs_*.json): every effective
config key found is compared with the job's override, and every family row below takes its expected value from the job of the arm (never from a per-arm constant in this file). The only constants are the
SPEC family counts of FAMCOUNT (override with --expect) and the patch-defined Hilti point-time values (HILTI_PT) used when a Hilti job sets no point_time key.
Groups: DST, MXL, VSW, D1B, D4P, MQS, OWN (and any other group the table has; FAMCOUNT lists a group the SPEC table does not know). Rows of R83:
  DSTENG    (also PCLENG rows for the D4P arms that carry the dump keys, the 8 eee_01 arms; the job must agree with the arm table column dump_frames; the other D4P arms engage no dump) residual dump present for the arm (residual_dump.csv or residual_dump_all.csv.gz), column voxel_layer present and non-empty (p3), the set of `frame` values equals lio.residual_dump_frames of the job,
            the job carries lio.residual_dump_voxel_info true; plane_fit_reasons.csv present with rows (voxel_map.plane.plane_fit_reasons_en of the job)
  MXLENG    the params echo shows voxel_map.map.max_layer equal to the job value (a MXL arm whose job has no such key is a FAIL)
  VSWENG    the params echo shows voxel_map.map.voxel_size equal to the job value, the job value equals the edge in the arm name (v0p35 = 0.35); an overlap-mode dc arm: no cbk.lidar.retime_* and no overlap_policy key in the job
  PCWENG    lio.point_cov.window_s of the job > 0: point_cov_window.csv present with rows, column window_s equal to the job, profiler.txt of the arm (outputs.profiler.export_path; run.log only as fallback) has the timer imu/window_cov_profile with frames > 0 (PCWVAL: sd_p_med_m not all zero);
            window_s 0 or key absent: neither the file nor the timer
  PCLENG    D4P: peak_rss.txt present for the arm; an ABORTED_RSS marker (file, arm.json or peak_rss.txt) is a FAIL and is REPORTED, not a stop (no hold); convergence_mode and points max of the params echo equal the job
  MQSENG    the params echo shows imu.process_noise.motion.gyro_scale and acc_scale equal to the job; motion_q_scan.csv present (MQSFILE: its gyro_scale / acc_scale columns equal the job values)
  ZUPTGONE  no lio/zupt key in any params echo or engagement text and none in any job; a refuse arm whose job sets a lio.zupt key was refused with the removed-key text naming it
  ODOMEXP   a job with outputs.odom.export true: the params echo prints outputs/odom/export: true and odometry.txt is present in the arm folder
  NOEXTRA   nothing engaged that the job did not ask for: no time_cov, plane-variance, D4, point-time (non-Hilti), point_cov_window file or timer, voxel_layer column in a dump without lio.residual_dump_voxel_info
  PFT       arms whose job sets voxel_map.plane.plane_fit_trace_frames > 0 have rows in plane_fit_trace_counts_all.csv
  HILCOMP   r83_hilti_metrics.csv (else r82_hilti_metrics.csv, the name r82_hilti_reduce.py writes) has a row for every dispatched Hilti arm
  DET, DETPOS, IDENT   as in R82 (IDENT = the identity hold of r83_identity_hold.py against the R82 return)
Fixes against r82_gates.py (see REPORT.md): DGTCFG read the job's evo.dense_gt_metrics through a skip list and fell back to the default false on the Hilti arms (the job says true); PTCFG reported MISSING for point_time keys that the
params echo never prints although the engagement line shows them; MSGSCANS demanded accepted >= scans for dc arms, where the trace holds more scans than accepted messages (INFO there).
A FAIL of a DET, DETPOS or IDENT row writes --hold-file (HOLD_FROM_GATES.txt): the agent places it as HOLD.txt (r83_hold_check.sh). Every other FAIL (incl. PCLENG ABORTED_RSS) is reported only. The gate never deletes or cancels anything.
"""
import argparse, csv, glob, gzip, io, json, os, re, shutil, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r83_identity_hold as IH  # noqa: E402

LEN_MIN = 0.97
REF_OFFSET = -0.1
N_SCANS = {"eee_01": 3987, "eee_02": 3210, "eee_03": 1814, "nya_01": 3950, "nya_02": 4287, "nya_03": 4094, "sbs_01": 3542, "sbs_02": 3732, "sbs_03": 3893}   # message / scan counts of the bags (make_r79_manifest.N_SCANS)
EXPECT = {"DST": (12,), "MXL": (36,), "VSW": (234,), "D1B": (72,), "D4P": (24,), "MQS": (108,)}   # SPEC table (planner, 2026-10-07); OWN is a fact, not checked
HILTI_PT = {"field": "timestamp", "mode": "absolute_s_f64"}   # patch r83_p2 sets these in config/slam_2022_handheld.yaml; used only for a Hilti arm whose job sets no point_time key
MSG_COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict"]
MSG_COLS_C = MSG_COLS + ["retime_shift_s", "n_trimmed"]
VSW_RE = re.compile(r"^r83_dc_sw_(?P<bag>[a-z]+_\d\d)_(?P<line>off)_v(?P<edge>[0-9p]+)_pfn(?P<pfn>\d)_(?P<q>fq|mq)$")
RETIME_KEYS = ("cbk.lidar.retime_t0_s", "cbk.lidar.retime_period_s", "cbk.lidar.retime_seq0", "cbk.lidar.retime_ref_offset_s")
K_WIN, K_MAXL, K_VOX, K_PTS, K_CONV = "lio.point_cov.window_s", "voxel_map.map.max_layer", "voxel_map.map.voxel_size", "voxel_map.points.max", "voxel_map.plane.convergence_mode"
K_GYRO, K_ACC = "imu.process_noise.motion.gyro_scale", "imu.process_noise.motion.acc_scale"
K_DUMP, K_DUMPV, K_REASONS, K_TRACE = "lio.residual_dump_frames", "lio.residual_dump_voxel_info", "voxel_map.plane.plane_fit_reasons_en", "voxel_map.plane.plane_fit_trace_frames"
STRICT = re.compile(r"^(voxel_map\.map\.(voxel_size|max_layer)|voxel_map\.points\.max|cbk\.lidar\.(point_filter_num|overlap_policy|retime_.*)|imu\.process_noise\.switch\..*|imu\.process_noise\.model|"
                    r"imu\.process_noise\.motion\.(gyro|acc)_scale|imu\.ds\.mode|lio\.point_cov\.window_s|lio\.residual_dump_(frames|voxel_info)|"
                    r"voxel_map\.plane\.(plane_var_min|plane_var_scan_s2|plane_var_diag_en|plane_gap_floor_en|debiased_line_equiv|plane_fit_mode|convergence_mode|plane_fit_reasons_en|plane_fit_trace_frames)|"
                    r"voxel_map\.plane\.line_residual\.(enabled|form|rule|rows|classify_at_fit)|lio\.time_cov\..*|evo\.dense_gt_metrics|calib\.stationary\.num_samples|data_queues\.log_lidar_msgs)$")
THEMES = [("OVLCFG", re.compile(r"^cbk\.lidar\.(overlap_policy|retime_.*)$")), ("MSGKEY", re.compile(r"^data_queues\.log_lidar_msgs$")), ("SWXCFG", re.compile(r"^imu\.process_noise\.switch\..*$")),
          ("MQSCFG", re.compile(r"^imu\.process_noise\.(model|motion\..*)$")), ("PCWCFG", re.compile(r"^lio\.point_cov\..*$")), ("DSTCFG", re.compile(r"^(lio\.residual_dump_.*|voxel_map\.plane\.plane_fit_(reasons_en|trace_frames))$")),
          ("PCLCFG", re.compile(r"^voxel_map\.(points\.max|plane\.convergence_mode)$")),
          ("VMINCFG", re.compile(r"^voxel_map\.plane\.plane_var_min$")), ("VS2CFG", re.compile(r"^voxel_map\.plane\.plane_var_scan_s2$")), ("PVDCFG", re.compile(r"^voxel_map\.plane\.plane_var_diag_en$")),
          ("D4CFG", re.compile(r"^voxel_map\.plane\.(line_residual\.(classify_at_fit|enabled|rows)|debiased_line_equiv|plane_gap_floor_en|plane_fit_mode)$")), ("TCCFG", re.compile(r"^lio\.time_cov\..*$")),
          ("PTCFG", re.compile(r"^cbk\.lidar\.point_time_.*$")), ("DGTCFG", re.compile(r"^evo\.dense_gt_metrics$")), ("DSCFG", re.compile(r"^imu\.ds\.mode$")),
          ("PFN", re.compile(r"^cbk\.lidar\.point_filter_num$")), ("VOX", re.compile(r"^voxel_map\.map\.(voxel_size|max_layer)$"))]
DEFAULTS = {"voxel_map.plane.plane_var_min": 0.0, "voxel_map.plane.plane_var_scan_s2": 0.0, "voxel_map.plane.plane_var_diag_en": False, "voxel_map.plane.line_residual.classify_at_fit": False,
            "voxel_map.plane.plane_gap_floor_en": False, "lio.time_cov.mode": "off", "evo.dense_gt_metrics": False, "cbk.lidar.overlap_policy": "discard",
            "voxel_map.plane.line_residual.enabled": False, "lio.point_cov.window_s": 0.0, "voxel_map.plane.convergence_mode": "normal", "voxel_map.points.max": 50, "voxel_map.map.max_layer": 2}   # a key the job does not set is at its compiled default (code defaults of src/map/voxelmap.cpp); a build that prints it must print the default
SKIP_KEYS = re.compile(r"^(outputs\.|eval\.|evo\.(?!dense_gt_metrics$)|vio\.|calib\.p0\.|state\.cov\.|diagnostics\.|estimator\.mode$|spline\.)")   # R83: evo.dense_gt_metrics is NOT skipped (R82 skipped all evo.*, dropped the job's true on the Hilti arms and compared against the default false)


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
                          needles=[x for x in (r.get("needles") or "").split("|") if x], flag=r.get("flag", ""), rss=(r.get("rss_abort_frac") or "").strip(), dumpf=(r.get("dump_frames") or "").strip(), req=(r.get("requires") or ""), ov=j.get("overrides"), has_job=r["arm"] in jobs))
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
        if st == "NF" and theme_of(k) == "PTCFG" and re.search(r"cbk/lidar/point_time:\s*field=", et or ""):
            st, det = "INFO", "%s is not printed by the params echo; the engagement line is judged by PTENG" % k      # R82 reported MISSING for keys the build never echoes
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


# ------------------------------------------------------------------------------------------------ R83 family gates
def find_any(d, names):
    """first existing <d>/<name> or <d>/<name>.gz."""
    for n in names:
        for suf in ("", ".gz"):
            p = os.path.join(d, n + suf)
            if os.path.isfile(p):
                return p
    return None


def read_cols(p, cols=None):
    """-> (header, DataFrame with the wanted columns that exist, or all) ; (None, None) when unreadable or empty."""
    try:
        hdr = list(pd.read_csv(p, nrows=0).columns)
        use = [c for c in cols if c in hdr] if cols else None
        return hdr, pd.read_csv(p, usecols=use)
    except Exception:
        return None, None


TIMER_NAME = "imu/window_cov_profile"
PROF_LINE = re.compile(r"imu/window_cov_profile\s+([0-9.eE+-]+)\s*ms\s*\((\d+)\s*frames?\)")


def window_timer(d, et):
    """-> (found, src, n_frames or None). profiler.txt of the arm first (outputs.profiler.export_path; lines `<timer> <avg ms> ms (<n> frames)`), run.log / engagement text only as the fallback."""
    t = rd(d, "profiler.txt")
    if t is not None:
        for l in t.splitlines():
            if TIMER_NAME in l:
                m = PROF_LINE.search(l)
                return True, "profiler.txt", (int(m.group(2)) if m else None)
    if TIMER_NAME in (et or ""):
        m = PROF_LINE.search(et)
        return True, "run.log", (int(m.group(2)) if m else None)
    return False, "", None


def frames_of(v):
    return {int(x) for x in re.findall(r"\d+", str(v))} if v not in (None, "") else set()


def echo_row(idx, key, want, have_log):
    """-> (status, detail) of one params-echo comparison: PASS / FAIL / MISSING (not printed, or no log)."""
    if not have_log:
        return "MISSING", "no run.log text"
    st, det = judge_key(idx, key, want)
    return ("MISSING" if st == "NF" else st), det


def aborted_markers(d):
    hits = []
    if os.path.isdir(d):
        hits += [n for n in os.listdir(d) if n.upper().startswith("ABORTED_RSS")]
    for f in ("peak_rss.txt", "arm.json", "exit_status.txt", "run_log_tail.txt"):
        if "ABORTED_RSS" in (rd(d, f) or ""):
            hits.append(f + " (text)")
    return hits


def family_gates(t, ov, log, et, d):
    """R83 family rows of ONE arm (not a refuse arm). The expectation is the job's own value; a family arm whose job lacks its key is a FAIL (manifest defect). -> list of (name, status, detail)."""
    arm, grp, ov = t["id"], t["group"], ov or {}
    have = bool(log.strip())
    idx = parse_log(log)
    out = []
    add = lambda n, s, dt="": out.append((n, s, dt))
    # ---- DST: the dump with the voxel layer, the plane-fit reasons
    jf = frames_of(ov.get(K_DUMP))
    pfx = "PCLENG" if grp == "D4P" else "DSTENG"      # the 8 eee_01 D4P arms carry the dump too: their rows are PCLENG rows; the other 16 D4P arms must not engage it
    tf = frames_of(t.get("dumpf"))                    # arm table column dump_frames (the planner's expectation); the job must agree with it
    if grp == "DST" or jf or tf:
        if not jf:
            add(pfx + " %s the job sets %s (a %s arm must)" % (arm, K_DUMP, "DST" if grp == "DST" else "dump-carrying"), "FAIL", "key absent from the job")
        elif tf and tf != jf:
            add(pfx + " %s the job's %s equals the arm table column dump_frames" % (arm, K_DUMP), "FAIL", "job %s, arm table %s" % (sorted(jf), sorted(tf)))
        else:
            p = find_any(d, ["residual_dump.csv", "residual_dump_all.csv"])
            hdr, D = read_cols(p, ["frame", "voxel_layer"]) if p else (None, None)
            nm = pfx + " %s residual dump present and its frames equal the job's %s (%s)" % (arm, K_DUMP, ",".join(str(x) for x in sorted(jf)))
            if p is None:
                add(nm, "MISSING", "no residual_dump.csv / residual_dump_all.csv.gz")
            elif D is None or "frame" not in D.columns or not len(D):
                add(nm, "FAIL", "dump unreadable, empty or without a frame column")
            else:
                got = {int(x) for x in D["frame"].dropna().unique()}
                add(nm, "PASS" if got == jf else "FAIL", "dump frames %s" % ",".join(str(x) for x in sorted(got)) if got == jf else "job frames not in the dump: %s; dump frames not in the job: %s" % (sorted(jf - got), sorted(got - jf)))
            nm = pfx + " %s dump column voxel_layer present and non-empty (job sets %s true)" % (arm, K_DUMPV)
            if not ov.get(K_DUMPV):
                add(nm, "FAIL", "the job does not set %s true" % K_DUMPV)
            elif p is None:
                add(nm, "MISSING", "no dump")
            elif D is None or "voxel_layer" not in D.columns:
                add(nm, "FAIL", "column voxel_layer absent (patch r83_p3 not in the build, or the key not honoured)")
            else:
                nn = int(D["voxel_layer"].notna().sum())
                add(nm, "PASS" if nn > 0 else "FAIL", "%d of %d rows have a layer" % (nn, len(D)))
    if grp == "DST" or ov.get(K_REASONS):
        nm = "DSTENG %s plane_fit_reasons.csv present with rows (the job sets %s)" % (arm, K_REASONS)
        p = find_any(d, ["plane_fit_reasons.csv", "plane_fit_reasons_all.csv"])
        if not ov.get(K_REASONS):
            add(nm, "FAIL", "the job does not set %s true" % K_REASONS)
        elif p is None:
            add(nm, "MISSING", "file absent")
        else:
            _, R = read_cols(p)
            add(nm, "PASS" if R is not None and len(R) > 0 else "FAIL", "rows %s" % (0 if R is None else len(R)))
    # ---- MXL
    if grp == "MXL" or K_MAXL in ov:
        nm = "MXLENG %s params echo voxel_map.map.max_layer equals the job's" % arm
        if K_MAXL not in ov:
            add(nm, "FAIL", "key absent from the job")
        else:
            add(nm, *echo_row(idx, K_MAXL, ov[K_MAXL], have))
    # ---- VSW
    m = VSW_RE.match(arm)
    if grp == "VSW" or m:
        nm = "VSWENG %s params echo voxel_map.map.voxel_size equals the job's, which equals the edge in the arm name" % arm
        if K_VOX not in ov:
            add(nm, "FAIL", "key absent from the job")
        else:
            st, det = echo_row(idx, K_VOX, ov[K_VOX], have)
            if m:
                edge = float(m.group("edge").replace("p", "."))
                if abs(float(ov[K_VOX]) - edge) > 1e-9:
                    st, det = "FAIL", "job voxel_size %r but the arm name says %g" % (ov[K_VOX], edge)
            add(nm, st, det)
        if t["mode"] == "dc":
            bad = [k for k in ov if k in RETIME_KEYS or k == "cbk.lidar.overlap_policy"]
            add("VSWENG %s overlap mode dc: the job has no retime key and no overlap_policy key" % arm, "FAIL" if bad else "PASS", ", ".join(bad))
    # ---- PCW (D1B)
    w = ov.get(K_WIN)
    wf = float(w) if isinstance(w, (int, float)) and not isinstance(w, bool) else None
    won = wf is not None and wf > 0
    cp = find_any(d, ["point_cov_window.csv", "point_cov_window_all.csv"])
    timer, tsrc, tn = window_timer(d, et)
    if grp == "D1B" and K_WIN not in ov:
        add("PCWENG %s the job sets %s (a D1B arm must)" % (arm, K_WIN), "FAIL", "key absent from the job")
    elif won:
        nm = "PCWENG %s point_cov_window.csv present with rows and window_s equal to the job's %g, timer imu/window_cov_profile in profiler.txt (run.log only as fallback) with frames > 0" % (arm, wf)
        if cp is None:
            add(nm, "MISSING", "point_cov_window.csv absent")
        else:
            _, W = read_cols(cp)
            if W is None or not len(W):
                add(nm, "FAIL", "file unreadable or without rows")
            elif "window_s" not in W.columns:
                add(nm, "FAIL", "no window_s column")
            else:
                ws = pd.to_numeric(W["window_s"], errors="coerce")
                bad = int((~np.isclose(ws, wf, rtol=1e-9, atol=1e-12)).sum())
                tbad = (not timer) or (tn is not None and tn <= 0)
                tok = timer and not tbad
                prof_here = os.path.isfile(os.path.join(d, "profiler.txt")) or os.path.isfile(os.path.join(d, "profiler.txt.gz"))
                add(nm, "PASS" if bad == 0 and (tok or not (have or prof_here)) else ("MISSING" if bad == 0 and not (have or prof_here) else "FAIL"),
                    ("rows %d" % len(W)) + ("; %d rows with another window_s" % bad if bad else "") + (("; timer from %s, %s frames" % (tsrc, tn if tn is not None else "n/a")) if timer else "; timer imu/window_cov_profile in neither profiler.txt nor run.log")
                    + ("; timer reports 0 frames" if timer and tn is not None and tn <= 0 else ""))
                if "sd_p_med_m" in W.columns:
                    sp = pd.to_numeric(W["sd_p_med_m"], errors="coerce")
                    add("PCWVAL %s point_cov_window.csv sd_p_med_m is not all zero (a window that is on must produce a covariance)" % arm, "PASS" if float(sp.abs().max() or 0) > 0 else "FAIL", "max %.6g m over %d rows" % (float(sp.abs().max() or 0), len(sp)))
        add("PCWENG %s params echo %s equals the job's" % (arm, K_WIN), *echo_row(idx, K_WIN, w, have))
    elif grp == "D1B" or K_WIN in ov or cp or timer:
        stray = (["point_cov_window.csv present"] if cp else []) + (["imu/window_cov_profile timer in %s" % tsrc] if timer else [])
        add("PCWENG %s window_s is 0 or absent: no point_cov_window.csv and no imu/window_cov_profile timer" % arm, "FAIL" if stray else ("PASS" if have else "MISSING"), ", ".join(stray))
    # ---- PCL (D4P) and the RSS watcher
    ab = aborted_markers(d)
    if grp == "D4P" or K_PTS in ov or K_CONV in ov or ab:
        rp = find_any(d, ["peak_rss.txt"])
        nm = "PCLENG %s peak_rss.txt present and no ABORTED_RSS marker (a marker is a FAIL that is reported, not a stop)" % arm
        if ab:
            add(nm, "FAIL", "ABORTED_RSS: %s" % ", ".join(ab))
        elif rp is None:
            if grp == "D4P" or t.get("rss"):       # the memory queue (arm table: group D4P, column rss_abort_frac) records peak RSS; an OWN smoke arm with the same keys runs unwatched
                add(nm, "MISSING", "peak_rss.txt absent")
        else:
            add(nm, "PASS", " ".join((rd(d, "peak_rss.txt") or "").split())[:120])
        for k in (K_CONV, K_PTS):
            if grp == "D4P" and k not in ov:
                add("PCLENG %s the job sets %s (a D4P arm must)" % (arm, k), "FAIL", "key absent from the job")
            elif k in ov:
                add("PCLENG %s params echo %s equals the job's" % (arm, k), *echo_row(idx, k, ov[k], have))
    # ---- MQS
    if grp == "MQS" or K_GYRO in ov or K_ACC in ov:
        for k in (K_GYRO, K_ACC):
            if k not in ov:
                add("MQSENG %s the job sets %s (an MQS arm must)" % (arm, k), "FAIL", "key absent from the job")
            else:
                add("MQSENG %s params echo %s equals the job's" % (arm, k), *echo_row(idx, k, ov[k], have))
        mp = find_any(d, ["motion_q_scan.csv", "motion_q_scan_all.csv"])
        if not (grp == "MQS" or "motion_q_scan.csv" in t.get("req", "")):
            pass      # the mq arms of the other families set the same keys at their defaults; the manifest asks for the trace of the MQS arms only
        elif mp is None:
            add("MQSENG %s motion_q_scan.csv present" % arm, "MISSING", "file absent")
        else:
            _, M = read_cols(mp, ["acc_scale", "gyro_scale"])
            nrow = 0 if M is None else len(M)
            add("MQSENG %s motion_q_scan.csv present with rows" % arm, "PASS" if nrow else "FAIL", "rows %d" % nrow)
            bad = []
            if nrow:
                for col, k in (("gyro_scale", K_GYRO), ("acc_scale", K_ACC)):
                    if k in ov and col in M.columns and not np.allclose(pd.to_numeric(M[col], errors="coerce"), float(ov[k]), rtol=1e-6):
                        bad.append("%s column differs from the job's %s" % (col, ov[k]))
                    elif k in ov and col not in M.columns:
                        bad.append("no %s column" % col)
            if nrow:
                add("MQSFILE %s motion_q_scan.csv gyro_scale and acc_scale columns equal the job's" % arm, "FAIL" if bad else "PASS", "; ".join(bad))
    # ---- ZUPT gone (every arm)
    zl = [l.strip() for l in (log + "\n" + et).splitlines() if re.search(r"lio[/.]zupt", l)]
    zj = [k for k in ov if k.startswith("lio.zupt.") or k.startswith("lio/zupt/")]
    add("ZUPTGONE %s no lio/zupt key in the params echo or the engagement text and none in the job (removed by r83_p1)" % arm, "FAIL" if (zl or zj) else ("PASS" if have else "MISSING"),
        ("job keys: %s; " % ", ".join(zj) if zj else "") + ("; ".join(zl[:2])[:200] if zl else ""))
    # ---- odometry export
    if ov.get("outputs.odom.export") is True:
        nm = "ODOMEXP %s outputs.odom.export true: the echo prints true and odometry.txt exists in the arm folder" % arm
        ech = [v for kp, v, pr, tl in idx.get("export", []) if kp.endswith("odom/export")]
        op = os.path.join(d, "odometry.txt")
        bad = []
        if have and not any(str(v).lower() == "true" for v in ech):
            bad.append("echo outputs/odom/export: %s" % (ech[0] if ech else "not printed"))
        st = "FAIL" if bad else ("PASS" if os.path.isfile(op) and os.path.getsize(op) > 0 and have else "MISSING")
        add(nm, st, "; ".join(bad) or ("odometry.txt %d bytes" % os.path.getsize(op) if os.path.isfile(op) else "odometry.txt absent from the arm folder"))
    return out


def engagement_gates(g, t, ov, log, et, d):
    arm = t["id"]; ov = ov or {}; out = []
    have = bool(et.strip()) or bool(log.strip())
    smoke = t["kind"] == "smoke"
    tl = _eng_line(et, r"lio/time_cov mode=")
    vl = next((l for l in et.splitlines() if re.search(r"plane_var_min\s*=.*\brows\s*=", l)), None)
    dl = _eng_line(et, r"plane_var_diag_en rows=")
    cl = _eng_line(et, r"line_residual/classify_at_fit\s*=.*line_classified=")
    pl = _eng_line(et, r"cbk/lidar/point_time:\s*field=")
    # ---- D4: classify_at_fit counters (the D4P arms)
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
    # ---- point time (Hilti, patch r83_p2: the values come from config/slam_2022_handheld.yaml unless the job sets them)
    pt_keys = [k for k in ov if k.startswith("cbk.lidar.point_time_")]
    hilti = t["bag"] == "hilti" or str(t["id"]).find("_hl_") >= 0
    if pt_keys or hilti:
        z = kv(pl or ""); bad = []
        if pl is None:
            bad.append("no cbk/lidar/point_time engagement line%s" % (" although a point_time key is set" if pt_keys else " on a Hilti arm (patch r83_p2 sets point time in the Hilti config)"))
        else:
            want = {"field": ov.get("cbk.lidar.point_time_field"), "mode": ov.get("cbk.lidar.point_time_mode"), "ref": ov.get("cbk.lidar.point_time_ref")}
            if not pt_keys:
                want = dict(HILTI_PT, ref=None)
            for f, wv in want.items():
                if wv is not None and z.get(f) is not None and not val_match(z[f], wv, f):
                    bad.append("%s %s, wanted %s" % (f, z[f], wv))
        out.append(("PTENG %s cbk/lidar/point_time engagement line as the job states it%s" % (arm, "" if pt_keys else " (no job key: the patch-defined Hilti values)"), "FAIL" if bad else ("MISSING" if pl is None and not have else "PASS"), "; ".join(bad) or (pl or "").strip()[:240]))
    # ---- dense ground truth (Hilti smoke)
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
    if tl and ov.get("lio.time_cov.mode") != "shared":
        extra.append("time_cov")
    if vl and not ((ov.get("voxel_map.plane.plane_var_min") or 0) > 0 or (ov.get("voxel_map.plane.plane_var_scan_s2") or 0) > 0):
        extra.append("plane variance keys")
    if dl and not ov.get("voxel_map.plane.plane_var_diag_en"):
        extra.append("plane_var_diag")
    if cl and not (cls_on or eq or fl_deb):
        extra.append("D4 keys")
    if pl and not (pt_keys or hilti):
        extra.append("point_time")
    w = ov.get(K_WIN)
    if not (isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0):
        if find_any(d, ["point_cov_window.csv", "point_cov_window_all.csv"]):
            extra.append("point_cov_window.csv")
        if window_timer(d, et)[0]:
            extra.append("imu/window_cov_profile timer")
    dp = find_any(d, ["residual_dump.csv", "residual_dump_all.csv"])
    if dp and not ov.get(K_DUMPV):
        hdr, _ = read_cols(dp, ["frame"])
        if hdr and "voxel_layer" in hdr:
            extra.append("voxel_layer column in a dump (job lacks %s)" % K_DUMPV)
    out.append(("NOEXTRA %s nothing engaged that the arm did not ask for (time_cov, plane variance, D4, point time, point_cov window file or timer, voxel_layer column)" % arm, "FAIL" if extra else ("PASS" if have else "MISSING"), ", ".join(extra)))
    return out


def hilti_table_gates(g, table, live, ret_dirs, runs):
    dirs = list(ret_dirs) + [runs, os.path.join(runs, "..")]
    hil = [t for t in table if (t["group"] == "HIL" or t["bag"] == "hilti") and t["id"] in live]
    if not hil:
        return
    p = find_return_file("r83_hilti_metrics.csv", dirs) or find_return_file("r82_hilti_metrics.csv", dirs)   # r82_hilti_reduce.py writes the r82_ name; a renamed R83 reducer would write the r83_ name
    H = pd.read_csv(p) if p else None
    if H is None or "arm" not in H.columns:
        g.add("HILCOMP r83_hilti_metrics.csv (or r82_hilti_metrics.csv) has a row for each of the %d Hilti arms" % len(hil), "MISSING", "file absent or without an arm column")
        return
    have = set(H["arm"])
    miss = [t["id"] for t in hil if t["id"] not in have]
    g.add("HILCOMP the Hilti metrics file has a row for each of the %d dispatched Hilti arms" % len(hil), "PASS" if not miss else "FAIL", "missing: %s" % ", ".join(miss[:12]))
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


def refuse_zupt_gate(t, d, present):
    """a refuse arm whose job sets a lio.zupt key must have been refused with the removed-key text that names the key (patch r83_p1: `configuration key '<key>' was removed in R83 ...`)."""
    keys = [k for k in (t["ov"] or {}) if k.startswith("lio.zupt.")]
    if not keys:
        return []
    nm = "ZUPTGONE %s refused: the died-at-startup text names the removed key %s and says it was removed" % (t["id"], keys[0].replace(".", "/"))
    if not present:
        return [(nm, "MISSING", "arm.json absent")]
    tail = "".join(rd(d, f) or "" for f in ("run_log_tail.txt", "run.log"))
    hit = keys[0].replace(".", "/") in tail and "removed" in tail
    return [(nm, "PASS" if hit else "FAIL", "key named and 'removed' in the text: %s" % hit)]


class SwAgg:
    """F498 arms: FAIL / MISSING rows stay one per arm; the PASS rows of a (gate, cell) fold into one row (the report stays inside its size ledger)."""

    def __init__(self, g):
        self.g, self.c = g, {}

    def put(self, kind, cell, name, st, det=""):
        if st != "PASS":
            self.g.add(name, st, det)
        self.c.setdefault((kind, cell), {"PASS": 0, "FAIL": 0, "MISSING": 0})[st] += 1

    def flush(self, sizes):
        for (kind, cell), c in self.c.items():
            n = sizes.get(cell, 0)
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
        is_sw = grp == "VSW"
        if is_sw:
            m = VSW_RE.match(arm)
            cell = "vsw_%s_%s_%s_p%s" % (m.group("bag"), m.group("q"), m.group("line"), m.group("pfn")) if m else "vsw_unparsed"
            swcells.setdefault(cell, []).append(arm)
        put = (lambda kind, name, st, det="": sw.put(kind, cell, name, st, det)) if is_sw else (lambda kind, name, st, det="": g.add(name, st, det))
        pres = present(arm)
        sent = (jsn(d, "arm.json") or {}).get("sentinel") if pres else None
        ex = exit_status(d) if os.path.isdir(d) else None
        if t["kind"] == "refuse":
            ref_gates(g, t, d, pres, sent, ex)
            for n_, s_, dt_ in refuse_zupt_gate(t, d, pres):
                g.add(n_, s_, dt_)
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
        for n_, s_, dt_ in (family_gates(t, t["ov"], log, et, d) if t["ov"] is not None else []):
            if is_sw:
                put(n_.split(" ")[0], n_, s_, dt_)
            else:
                g.add(n_, s_, dt_)
        if is_sw:
            continue                    # VSW: summaries only (like the F498 Tier A it mirrors); no pos_trace is required (a trace only when gt_max_mm > 500), no engagement gates
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
    sw.flush({c: len(a_) for c, a_ in swcells.items()})
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
        full = len([t for t in table if t["group"] == "VSW" and (VSW_RE.match(t["id"]) and "vsw_%s_%s_%s_p%s" % (VSW_RE.match(t["id"]).group("bag"), VSW_RE.match(t["id"]).group("q"), VSW_RE.match(t["id"]).group("line"), VSW_RE.match(t["id"]).group("pfn")) == cell)])
        if not miss and len(arms_) == full:
            g.add("SWCOMP %s all %d arms have a phase_summary row" % (cell, len(arms_)), "PASS")
        elif len(arms_) != full:
            g.add("SWCOMP %s dispatched arms of the cell: %d of the %d edges of the manifest" % (cell, len(arms_), full), "INFO")
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
    """HILCOMP, PFT: group-level completeness from files of the return (found in ret_dirs)."""
    dirs = list(ret_dirs) + [runs, os.path.join(runs, "..")]
    arms = lambda f: [t for t in table if f(t) and t["id"] in live]
    hilti_table_gates(g, table, live, ret_dirs, runs)
    pft = [t for t in arms(lambda t: True) if (t["ov"] or {}).get(K_TRACE) not in (None, 0, False, "0")]
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
    body = ["HOLD (from r83_gates.py, %s): a determinism or identity gate FAILED. Do not dispatch the unstarted queues; let running jobs finish; cancel nothing; delete nothing; change nothing in the tree." % time.strftime("%Y-%m-%d %H:%M:%S"),
            "Investigate (config dump diff, run.log, first-divergence scan, tree diff), report to the planner. The planner's written word lifts the hold (rename HOLD.txt to HOLD.released_<date>.txt; never delete it).", ""]
    body += ["%s   [%s]" % (n, d) for n, s, d in bad]
    with open(hold_file, "a" if os.path.exists(hold_file) else "w") as f:
        f.write("\n".join(body) + "\n")
    return True


# ------------------------------------------------------------------------------------------------ selftest fixtures
def selftest():
    """Synthetic: a manifest written here (arm_table.tsv + jobs_x.json; no generator is imported) and fabricated arm folders. Planner sandbox, not evidence about the node."""
    root = tempfile.mkdtemp(prefix="r83_gates_selftest_")
    fails, nck = [], [0]

    def ck(c, m):
        nck[0] += 1
        if not c:
            fails.append(m)
    cols = ["arm", "tier", "group", "fam", "mode", "kind", "arm_kind", "ref", "bag", "seq", "batch", "pfn", "duration_s", "msglog", "needles", "flag", "dump_frames"]
    ARMS = []
    BASE = {"cbk.lidar.point_filter_num": 3, "imu.ds.mode": "first"}

    def arm(name, group, ov, mode="dc", kind="fb", ref="", bag="eee_01", dur="full", msglog=0, needles="", fam="", dumpf=""):
        ARMS.append((dict(arm=name, tier="FB", group=group, fam=fam or group, mode=mode, kind=kind, arm_kind=kind, ref=ref, bag=bag, seq=bag, batch="q_" + bag, pfn=3, duration_s=dur, msglog=msglog, needles=needles, flag="", dump_frames=dumpf), dict(BASE, **ov)))
    FR = "5,6,7,480,481"
    arm("d_ok", "DST", {K_DUMP: FR, K_DUMPV: True, K_REASONS: True, K_TRACE: 10}, dur="40")
    arm("d_frame", "DST", {K_DUMP: FR, K_DUMPV: True, K_REASONS: True}, dur="40")
    arm("d_nolayer", "DST", {K_DUMP: FR, K_DUMPV: True, K_REASONS: True}, dur="40")
    arm("d_noreasons", "DST", {K_DUMP: FR, K_DUMPV: True, K_REASONS: True}, dur="40")
    arm("d_nokey", "DST", {}, dur="40")
    arm("m_ok", "MXL", {K_MAXL: 1})
    arm("m_bad", "MXL", {K_MAXL: 1})
    arm("m_nokey", "MXL", {})
    arm("r83_dc_sw_eee_01_off_v0p35_pfn3_fq", "VSW", {K_VOX: 0.35})
    arm("r83_dc_sw_eee_01_off_v0p40_pfn3_fq", "VSW", {K_VOX: 0.4})
    arm("r83_dc_sw_eee_01_off_v0p45_pfn3_fq", "VSW", {K_VOX: 0.45})
    arm("r83_dc_sw_eee_01_off_v0p50_pfn3_fq", "VSW", {K_VOX: 0.55})            # job disagrees with the edge in the name
    arm("r83_dc_sw_eee_01_off_v0p55_pfn3_fq", "VSW", {K_VOX: 0.55, "cbk.lidar.retime_t0_s": 1.0})   # a dc job with a retime key
    arm("p_on", "D1B", {K_WIN: 0.02})
    arm("p_nofile", "D1B", {K_WIN: 0.02})
    arm("p_notimer", "D1B", {K_WIN: 0.02}); arm("p_runlog", "D1B", {K_WIN: 0.02}); arm("p_zeroframes", "D1B", {K_WIN: 0.02})
    arm("p_zero", "D1B", {K_WIN: 0.0})
    arm("p_zerofile", "D1B", {K_WIN: 0.0})
    arm("p_nokey", "D1B", {})
    arm("p_flat", "D1B", {K_WIN: 0.02})
    arm("c_ok", "D4P", {K_PTS: 200, K_CONV: "always_update", "voxel_map.plane.line_residual.classify_at_fit": True})
    D4X = {K_PTS: 200, K_CONV: "always_update", "voxel_map.plane.line_residual.classify_at_fit": True}
    arm("c_dump", "D4P", dict(D4X, **{K_DUMP: FR, K_DUMPV: True}), dumpf=FR)
    arm("c_dumpbad", "D4P", dict(D4X, **{K_DUMP: FR, K_DUMPV: True}), dumpf=FR)
    arm("c_dumpmiss", "D4P", dict(D4X), dumpf=FR)       # the arm table says the arm dumps, the job has no key
    arm("c_dumpdiff", "D4P", dict(D4X, **{K_DUMP: "5,6", K_DUMPV: True}), dumpf=FR)
    arm("c_abort", "D4P", {K_PTS: 200, K_CONV: "always_update", "voxel_map.plane.line_residual.classify_at_fit": True})
    arm("c_norss", "D4P", {K_PTS: 100, K_CONV: "normal"})
    arm("c_echo", "D4P", {K_PTS: 100, K_CONV: "normal"})
    arm("q_ok", "MQS", {K_GYRO: 3.0, K_ACC: 0.15, "imu.process_noise.model": "isotropic"})
    arm("q_file", "MQS", {K_GYRO: 3.0, K_ACC: 0.15})
    arm("q_nofile", "MQS", {K_GYRO: 1.0, K_ACC: 0.1})
    arm("z_log", "OWN", {})
    arm("z_ref", "OWN", {"lio.zupt.mode": "apply"}, kind="refuse", needles="lio/zupt/mode")
    arm("z_ref_bad", "OWN", {"lio.zupt.mode": "apply"}, kind="refuse", needles="lio/zupt/mode")
    arm("hl_smoke", "OWN", {"evo.dense_gt_metrics": True, "evo.enable": True, "outputs.odom.export": True, "outputs.path": "/x"}, bag="hilti", kind="smoke", dur="60", fam="HL")
    arm("hl_noodom", "OWN", {"evo.dense_gt_metrics": True, "outputs.odom.export": True}, bag="hilti", kind="smoke", dur="60", fam="HL")
    arm("x_layer", "OWN", {})
    arm("x_tc", "OWN", {})
    arm("det_a", "OWN", {}, kind="det_fb", ref="det_orig")
    arm("det_orig", "OWN", {})
    arm("id_ok", "OWN", {}, kind="id", ref="r82_ref_ok")
    arm("id_bad", "OWN", {}, kind="id", ref="r82_ref_bad")
    arm("msg_dc", "OWN", {}, msglog=1)
    mdir = os.path.join(root, "m"); os.makedirs(mdir)
    with open(os.path.join(mdir, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows([r for r, _ in ARMS])
    json.dump([dict(job_id=r["arm"], overrides=o) for r, o in ARMS], open(os.path.join(mdir, "jobs_x.json"), "w"))
    runs = os.path.join(root, "runs"); os.makedirs(runs)
    ECHO = lambda **kw: "".join("  %s: %s\n" % (k.replace("__", "/"), v) for k, v in kw.items())

    def mk(name, log="  voxel_map/plane/x: 1\n", eng="", files=None, sentinel="DONE", exit_="exit_code=0 signal=none"):
        d = os.path.join(runs, name); os.makedirs(d, exist_ok=True)
        json.dump({"sentinel": sentinel}, open(os.path.join(d, "arm.json"), "w")); open(os.path.join(d, "exit_status.txt"), "w").write(exit_ + "\n")
        open(os.path.join(d, "run.log"), "w").write(log); open(os.path.join(d, "engagement.txt"), "w").write(eng)
        for fn, txt in (files or {}).items():
            open(os.path.join(d, fn), "w").write(txt)
    dump = lambda frames, layer=True: "frame,iter,r" + (",voxel_layer" if layer is not None else "") + "\n" + "".join("%d,0,0.01%s\n" % (f, ("," + (str(f % 3) if layer else "")) if layer is not None else "") for f in frames)
    rows_of = lambda g: g.rows
    S = lambda g, pre: [s for n, s, _ in g.rows if n.startswith(pre)]
    T = {t["id"]: t for t in load_manifest(mdir)}
    # ---- nothing present: no crash, FAMCOUNT behaves
    g, _ = run_gates(runs, mdir, dispatched="d_ok", expect={"DST": (5,), "MXL": (3,), "VSW": (5,), "D1B": (9,), "D4P": (8,), "MQS": (3,)})
    ck(all(s == "PASS" for n, s, _ in g.rows if n.startswith("FAMCOUNT") and "manifest holds" in n), "FAMCOUNT must PASS for the fixture counts: %s" % [(n, s) for n, s, _ in g.rows if n.startswith("FAMCOUNT")])
    g, _ = run_gates(runs, mdir, dispatched="d_ok", expect={"DST": (12,), "MXL": (3,), "VSW": (5,), "D1B": (9,), "D4P": (8,), "MQS": (3,)})
    ck(any(n.startswith("FAMCOUNT DST") and s == "FAIL" for n, s, _ in g.rows), "FAMCOUNT must FAIL when the manifest holds 5 DST arms and the SPEC says 12")
    ck(any(s == "MISSING" for n, s, _ in g.rows if n.startswith("RUN ")), "an absent arm must be MISSING")
    # ---- DST
    mk("d_ok", files={"residual_dump.csv": dump([5, 6, 7, 480, 481]), "plane_fit_reasons.csv": "reason,n\nok,3\n"}, log=ECHO(**{"lio__residual_dump_frames": FR.replace(",", ";")}))
    mk("d_frame", files={"residual_dump.csv": dump([5, 6, 7, 480]), "plane_fit_reasons.csv": "reason,n\nok,3\n"})
    mk("d_nolayer", files={"residual_dump.csv": dump([5, 6, 7, 480, 481], layer=False), "plane_fit_reasons.csv": "reason,n\nok,3\n"})
    mk("d_noreasons", files={"residual_dump.csv": dump([5, 6, 7, 480, 481])})
    mk("d_nokey")
    g, _ = run_gates(runs, mdir, dispatched="d_ok,d_frame,d_nolayer,d_noreasons,d_nokey")
    D = lambda a, w: [s for n, s, _ in g.rows if n.startswith("DSTENG %s " % a) and w in n]
    ck(D("d_ok", "frames equal") == ["PASS"] and D("d_ok", "voxel_layer") == ["PASS"] and D("d_ok", "plane_fit_reasons") == ["PASS"], "DST good arm: %s" % [(n, s) for n, s, _ in g.rows if "DSTENG d_ok" in n])
    ck(D("d_frame", "frames equal") == ["FAIL"], "DST dump frames differ from the job must FAIL")
    ck(D("d_nolayer", "voxel_layer") == ["FAIL"], "DST dump without a voxel_layer column must FAIL")
    ck(D("d_noreasons", "plane_fit_reasons") == ["MISSING"], "DST without plane_fit_reasons.csv must be MISSING")
    ck(any(n.startswith("DSTENG d_nokey") and s == "FAIL" for n, s, _ in g.rows), "a DST arm whose job lacks the dump key must FAIL")
    ck(not any(n.startswith("NOEXTRA d_ok") and s == "FAIL" for n, s, _ in g.rows), "voxel_layer in a dump that asked for voxel info is not extra")
    # ---- MXL
    mk("m_ok", log=ECHO(**{"map__max_layer": 1})); mk("m_bad", log=ECHO(**{"map__max_layer": 2})); mk("m_nokey", log=ECHO(**{"map__max_layer": 2}))
    g, _ = run_gates(runs, mdir, dispatched="m_ok,m_bad,m_nokey")
    ck(S(g, "MXLENG m_ok") == ["PASS"] and S(g, "MXLENG m_bad") == ["FAIL"] and S(g, "MXLENG m_nokey") == ["FAIL"], "MXLENG: %s %s %s" % (S(g, "MXLENG m_ok"), S(g, "MXLENG m_bad"), S(g, "MXLENG m_nokey")))
    # ---- VSW (aggregated by cell)
    for e, echo in (("0p35", 0.35), ("0p40", 0.4), ("0p45", 0.45), ("0p50", 0.55), ("0p55", 0.55)):
        mk("r83_dc_sw_eee_01_off_v%s_pfn3_fq" % e, log=ECHO(**{"map__voxel_size": echo, "cbk__lidar__point_filter_num": 3}))
    mk("r83_dc_sw_eee_01_off_v0p40_pfn3_fq", log=ECHO(**{"map__voxel_size": 0.45}))      # echo differs from the job
    g, _ = run_gates(runs, mdir, dispatched="VSW")
    ck(any("VSWENG r83_dc_sw_eee_01_off_v0p40" in n and s == "FAIL" for n, s, _ in g.rows), "VSW echo differs from the job must FAIL (own row)")
    ck(any("VSWENG r83_dc_sw_eee_01_off_v0p50" in n and s == "FAIL" and "arm name says" in d for n, s, d in g.rows), "VSW job edge differs from the name must FAIL")
    ck(any("VSWENG r83_dc_sw_eee_01_off_v0p55" in n and "retime" in n and s == "FAIL" for n, s, _ in g.rows), "a dc VSW job with a retime key must FAIL")
    ck(not any("VSWENG r83_dc_sw_eee_01_off_v0p35" in n and s != "PASS" for n, s, _ in g.rows), "the good VSW arm has no non-PASS row of its own")
    ck(any(n.startswith("SWCELL") for n, s, _ in g.rows), "VSW PASS rows fold into cell rows")
    # ---- PCW
    pcw = lambda n=20: "scan_index,t_abs,t_rel_s,window_s,n_steps,sd_p_med_m,sd_p_p95_m,sd_p_max_m,sd_r_med_rad,sd_r_p95_rad,sd_r_max_rad\n" + "".join("%d,%d,%d,0.02,%d,0.001,0.002,0.003,0.0001,0.0002,0.0003\n" % (i, i, i, 4 + i % 2) for i in range(n))
    TM = "[timers] imu/window_cov_profile calls=20 total_ms=3\n"
    PF = "imu/window_cov_profile 0.031 ms (20 frames)\nimu/other 1.2 ms (20 frames)\n"
    mk("p_on", log=ECHO(**{"point_cov__window_s": 0.02}), files={"point_cov_window.csv": pcw(), "profiler.txt": PF})
    mk("p_runlog", log=ECHO(**{"point_cov__window_s": 0.02}) + TM, files={"point_cov_window.csv": pcw()})
    mk("p_zeroframes", log=ECHO(**{"point_cov__window_s": 0.02}), files={"point_cov_window.csv": pcw(), "profiler.txt": "imu/window_cov_profile 0 ms (0 frames)\n"})
    mk("p_nofile", log=ECHO(**{"point_cov__window_s": 0.02}), files={"profiler.txt": PF})
    mk("p_notimer", log=ECHO(**{"point_cov__window_s": 0.02}), files={"point_cov_window.csv": pcw()})
    mk("p_zero", log=ECHO(**{"point_cov__window_s": 0})); mk("p_zerofile", log=ECHO(**{"point_cov__window_s": 0}), files={"point_cov_window.csv": pcw(), "profiler.txt": PF})
    mk("p_nokey", log=ECHO(**{"point_cov__window_s": 0}))
    mk("p_flat", log=ECHO(**{"point_cov__window_s": 0.02}), files={"point_cov_window.csv": pcw().replace("0.001,0.002,0.003", "0,0,0"), "profiler.txt": PF})
    g, _ = run_gates(runs, mdir, dispatched="D1B")
    P = lambda a, w="window_s is": [s for n, s, _ in g.rows if n.startswith("PCWENG %s " % a) and w in n]
    ck(S(g, "PCWENG p_on point_cov_window.csv present") == ["PASS"] and S(g, "PCWVAL p_on") == ["PASS"], "PCW good arm: %s" % [(n, s) for n, s, _ in g.rows if "p_on" in n and "PCW" in n])
    ck(S(g, "PCWENG p_nofile point_cov_window.csv present") == ["MISSING"], "window on without the file: MISSING")
    ck(S(g, "PCWENG p_notimer point_cov_window.csv present") == ["FAIL"], "window on without the timer: FAIL")
    ck(S(g, "PCWENG p_runlog point_cov_window.csv present") == ["PASS"], "timer only in run.log (fallback): PASS")
    ck(S(g, "PCWENG p_zeroframes point_cov_window.csv present") == ["FAIL"], "profiler.txt timer with 0 frames: FAIL")
    ck(P("p_zero") == ["PASS"], "window 0 without file and timer: PASS %s" % P("p_zero"))
    ck(P("p_zerofile") == ["FAIL"] and any(n.startswith("NOEXTRA p_zerofile") and s == "FAIL" for n, s, _ in g.rows), "window 0 with the file and the timer: FAIL (PCWENG and NOEXTRA)")
    ck(P("p_nokey") == [] and any(n.startswith("PCWENG p_nokey") and s == "FAIL" and "must" in n for n, s, _ in g.rows), "a D1B arm without the window key: FAIL")
    ck(S(g, "PCWVAL p_flat") == ["FAIL"], "a window that is on and whose covariance is all zero: FAIL")
    # ---- PCL and the RSS watcher: no hold from it
    mk("c_ok", log=ECHO(**{"points__max": 200, "plane__convergence_mode": "always_update"}), files={"peak_rss.txt": "peak_rss_kb 1234\n"}, eng="  voxel_map/plane/line_residual/classify_at_fit = true line_classified=5 plane_kept=1\n")
    mk("c_abort", log=ECHO(**{"points__max": 200, "plane__convergence_mode": "always_update"}), files={"peak_rss.txt": "peak_rss_kb 99999\n", "ABORTED_RSS": "killed at 0.5 of RAM\n"}, sentinel="FAILED", exit_="exit_code=137 signal=SIGKILL")
    for a_ in ("c_dump", "c_dumpbad", "c_dumpmiss", "c_dumpdiff"):
        mk(a_, log=ECHO(**{"points__max": 200, "plane__convergence_mode": "always_update"}), files={"peak_rss.txt": "peak_rss_kb 1\n", "residual_dump.csv": dump([5, 6, 7, 480, 481] if a_ != "c_dumpbad" else [5, 6])})
    mk("c_norss", log=ECHO(**{"points__max": 100, "plane__convergence_mode": "normal"}))
    mk("c_echo", log=ECHO(**{"points__max": 50, "plane__convergence_mode": "normal"}), files={"peak_rss.txt": "peak_rss_kb 1\n"})
    g, _ = run_gates(runs, mdir, dispatched="D4P")
    ck(S(g, "PCLENG c_ok peak") == ["PASS"] and S(g, "PCLENG c_ok params echo voxel_map.points.max") == ["PASS"] and S(g, "PCLENG c_ok params echo voxel_map.plane.convergence_mode") == ["PASS"], "PCL good arm: %s" % [(n, s) for n, s, _ in g.rows if "PCLENG c_ok" in n])
    ck(S(g, "PCLENG c_abort peak") == ["FAIL"], "ABORTED_RSS marker is a FAIL")
    ck(S(g, "PCLENG c_norss peak") == ["MISSING"], "no peak_rss.txt is MISSING")
    ck(S(g, "PCLENG c_echo params echo voxel_map.points.max") == ["FAIL"], "echo of points max differs from the job: FAIL")
    PD = lambda a, w: [s_ for n, s_, _ in g.rows if n.startswith("PCLENG %s " % a) and w in n]
    ck(PD("c_dump", "frames equal") == ["PASS"] and PD("c_dump", "voxel_layer") == ["PASS"] and not any(n.startswith("DSTENG c_dump") for n, _, _ in g.rows), "D4P arm with dump keys: PCLENG dump rows PASS, no DSTENG row")
    ck(PD("c_dumpbad", "frames equal") == ["FAIL"], "D4P dump frames differ from the job: FAIL")
    ck(PD("c_dumpmiss", "the job sets") == ["FAIL"], "arm table dump_frames set but the job lacks the key: FAIL")
    ck(PD("c_dumpdiff", "equals the arm table") == ["FAIL"], "job dump frames differ from the arm table column: FAIL")
    ck(not any("residual dump" in n or "voxel_layer" in n for n, _, _ in g.rows if n.startswith(("PCLENG c_ok ", "PCLENG c_echo "))) and not any(n.startswith("DSTENG c_ok") for n, _, _ in g.rows), "a D4P arm without dump keys has no dump row at all")
    hf = os.path.join(root, "HOLD_A.txt")
    ck(write_hold(g, hf) is False and not os.path.exists(hf), "an ABORTED_RSS / PCL FAIL must NOT write a hold file")
    # ---- MQS
    mkq = lambda gs, a_s, n=5: "scan_index,t_abs,t_rel_s,model,beta,acc_scale,gyro_scale,n_samples\n" + "".join("%d,%d,%d,isotropic,0,%s,%s,10\n" % (i, i, i, a_s, gs) for i in range(n))
    mk("q_ok", log=ECHO(**{"motion__gyro_scale": 3, "motion__acc_scale": 0.15, "process_noise__model": "isotropic"}), files={"motion_q_scan.csv": mkq(3, 0.15)})
    mk("q_file", log=ECHO(**{"motion__gyro_scale": 3, "motion__acc_scale": 0.15}), files={"motion_q_scan.csv": mkq(10, 0.15)})
    mk("q_nofile", log=ECHO(**{"motion__gyro_scale": 1, "motion__acc_scale": 0.3}))
    g, _ = run_gates(runs, mdir, dispatched="MQS")
    ck(S(g, "MQSENG q_ok params echo") == ["PASS", "PASS"] and S(g, "MQSFILE q_ok") == ["PASS"], "MQS good arm: %s" % [(n, s) for n, s, _ in g.rows if "q_ok" in n and "MQS" in n])
    ck(S(g, "MQSFILE q_file") == ["FAIL"], "file scale columns differ from the job: FAIL")
    ck("MISSING" in S(g, "MQSENG q_nofile motion_q_scan.csv present") and "FAIL" in S(g, "MQSENG q_nofile params echo"), "MQS: missing file MISSING, echo 0.3 vs job 0.1 FAIL: %s" % [(n, s) for n, s, _ in g.rows if "q_nofile" in n and "MQS" in n])
    # ---- ZUPT gone
    mk("z_log", log="  lio/zupt/mode = off\n")
    mk("z_ref", sentinel="FAILED", exit_="exit_code=134 signal=SIGABRT", log="terminate: configuration key 'lio/zupt/mode' was removed in R83 (F-401, F-508) and has no replacement; delete it from the configuration")
    mk("z_ref_bad", sentinel="FAILED", exit_="exit_code=134 signal=SIGABRT", log="terminate: something else lio/zupt/mode")
    g, _ = run_gates(runs, mdir, dispatched="z_log,z_ref,z_ref_bad")
    ck(S(g, "ZUPTGONE z_log") == ["FAIL"], "a lio/zupt line in a params echo must FAIL")
    ck(S(g, "ZUPTGONE z_ref ") == ["PASS"] and S(g, "ZUPTGONE z_ref_bad") == ["FAIL"], "refuse arm: removed-key text PASS, other text FAIL: %s %s" % (S(g, "ZUPTGONE z_ref "), S(g, "ZUPTGONE z_ref_bad")))
    ck(S(g, "REFUSE z_ref ") == ["PASS"], "the REFUSE row of the zupt refuse arm uses the needles of the table")
    mk("d_nokey", log=ECHO(**{"map__max_layer": 2}))
    g, _ = run_gates(runs, mdir, dispatched="d_nokey")
    ck(S(g, "ZUPTGONE d_nokey") == ["PASS"], "no zupt key anywhere: PASS")
    # ---- Hilti smoke: DGTCFG reads the job (R82 expected false), PTCFG INFO, PTENG with the patch values, ODOMEXP
    hl_log = ECHO(**{"evo__dense_gt_metrics": "true", "outputs__odom__export": "true", "cbk__lidar__point_filter_num": 3})
    mk("hl_smoke", log=hl_log, eng="  cbk/lidar/point_time: field=timestamp mode=absolute_s_f64 ref=header\n  evo/dense_gt_metrics: true\n  [evo/nearest]  n=3  ATE=0.0003m  ARE=104deg  ATE_first=0.0013m ARE_first=0.0168deg\n", files={"odometry.txt": "0 0 0 0\n"})
    mk("hl_noodom", log=hl_log.replace("outputs/odom/export: true", "outputs/odom/export: false"), eng="  cbk/lidar/point_time: field=time mode=offset_ns_u32\n  evo/dense_gt_metrics: true\n")
    g, _ = run_gates(runs, mdir, dispatched="hl_smoke,hl_noodom")
    ck(S(g, "DGTCFG hl_smoke") == ["PASS"], "DGTCFG must read the job's true on a Hilti arm and PASS (R82 FAILed it): %s" % [(n, s, d) for n, s, d in g.rows if "DGTCFG hl_smoke" in n])
    ck(S(g, "PTENG hl_smoke") == ["PASS"] and S(g, "PTENG hl_noodom") == ["FAIL"], "PTENG with the patch-defined Hilti values: %s %s" % (S(g, "PTENG hl_smoke"), S(g, "PTENG hl_noodom")))
    ck("MISSING" not in S(g, "PTCFG hl_smoke"), "PTCFG must not be MISSING when the engagement line shows the point time")
    ck(S(g, "ODOMEXP hl_smoke") == ["PASS"] and S(g, "ODOMEXP hl_noodom") == ["FAIL"], "ODOMEXP: %s %s" % (S(g, "ODOMEXP hl_smoke"), S(g, "ODOMEXP hl_noodom")))
    ck(S(g, "NOEXTRA hl_smoke") == ["PASS"], "a Hilti arm's point-time line is not extra: %s" % [(n, s, d) for n, s, d in g.rows if n.startswith("NOEXTRA hl_smoke")])
    # ---- NOEXTRA
    mk("x_layer", files={"residual_dump.csv": dump([1, 2])})
    mk("x_tc", eng="  lio/time_cov mode=shared sigma_s=0.002 iterations=1 applied=1\n")
    g, _ = run_gates(runs, mdir, dispatched="x_layer,x_tc")
    ck(S(g, "NOEXTRA x_layer") == ["FAIL"] and S(g, "NOEXTRA x_tc") == ["FAIL"], "NOEXTRA: voxel_layer column without the key, time_cov without the key")
    # ---- DET and IDENT write the hold; nothing else does
    sid = np.arange(1, 41)
    def trace(a, shift=0.0):
        d = os.path.join(runs, a); os.makedirs(d, exist_ok=True)
        pd.DataFrame({"scan_id": sid, "px": sid * 1e-3 + (shift * 1e-3 * (sid >= 20)), "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        json.dump({"excursion_mm": 10.0, "settled_offset_mm": 5.0}, open(os.path.join(d, "arm_summary.json"), "w"))
    mk("det_a", files={"state_trace_blank_runid.sha256": "aa" * 32 + "\n"}); mk("det_orig", files={"state_trace_blank_runid.sha256": "bb" * 32 + "\n"})
    trace("det_a", 0.02); trace("det_orig")
    ref = os.path.join(root, "R82"); os.makedirs(os.path.join(ref, "compact", "r82_ref_ok")); os.makedirs(os.path.join(ref, "compact", "r82_ref_bad"))
    pd.DataFrame({"scan_id": sid, "px": sid * 1e-3, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(ref, "compact", "r82_ref_ok", "pos_trace_all.csv.gz"), index=False, compression="gzip")
    pd.DataFrame({"scan_id": sid, "px": sid * 1e-3, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(ref, "compact", "r82_ref_bad", "pos_trace_all.csv.gz"), index=False, compression="gzip")
    pd.DataFrame([dict(arm=a_, excursion_mm=10.0, settled_offset_mm=5.0) for a_ in ("r82_ref_ok", "r82_ref_bad")]).to_csv(os.path.join(ref, "phase_summary.csv"), index=False)
    mk("id_ok"); mk("id_bad"); trace("id_ok"); trace("id_bad", 0.05)
    hold = os.path.join(root, "HOLD_FROM_GATES.txt")
    g, idr = run_gates(runs, mdir, ref_returns=[ref], dispatched="det_a,det_orig,id_ok,id_bad")
    ck(S(g, "DET det_a") == ["FAIL"] and S(g, "DETPOS det_a") == ["FAIL"], "DET and DETPOS FAIL on differing digest / trace")
    ck(any(n.startswith("IDENT id_ok") and s == "PASS" for n, s, _ in g.rows) and any(n.startswith("IDENT id_bad") and s == "FAIL" for n, s, _ in g.rows), "IDENT against the R82 return: id_ok PASS, id_bad FAIL")
    ck(write_hold(g, hold) and "DET det_a" in open(hold).read() and "IDENT id_bad" in open(hold).read(), "a DET / DETPOS / IDENT FAIL writes HOLD_FROM_GATES.txt")
    ck(len(idr) == 2 and {r["arm"] for r in idr} == {"id_ok", "id_bad"}, "identity csv rows: %r" % [r["arm"] for r in idr])
    # ---- MSGSCANS: INFO for a dc arm (R82 FAILed it)
    d = os.path.join(runs, "msg_dc"); mk("msg_dc"); trace("msg_dc")
    L = pd.DataFrame({"msg_index": range(50), "seq": range(50), "header_stamp": 0.0, "scan_time": 0.0, "gap_header_s": 0.1, "n_raw": 1, "n_kept": 1, "t_first": 0.0, "t_last": 0.0, "off_first_s": 0.0, "off_last_s": 0.0, "span_s": 0.1,
                      "prev_accepted_last_t": 0.0, "overlap_s": [0.0] * 30 + [0.01] * 20, "verdict": [0] * 30 + [1] * 20})
    L.to_csv(os.path.join(d, "lidar_msg_log_all.csv.gz"), index=False, compression="gzip")
    g, _ = run_gates(runs, mdir, dispatched="msg_dc")
    ck(any(n.startswith("MSGSCANS msg_dc dc arm") and s == "INFO" for n, s, _ in g.rows) and not any(n.startswith("MSGSCANS") and s == "FAIL" for n, s, _ in g.rows), "MSGSCANS for a dc arm is INFO")
    # ---- PFT: arms whose job sets plane_fit_trace_frames
    retd = os.path.join(root, "ret"); os.makedirs(retd)
    open(os.path.join(retd, "plane_fit_trace_counts_all.csv"), "w").write("arm,rows\nd_ok,12\n")
    g, _ = run_gates(runs, mdir, ret_dirs=[retd], dispatched="d_ok")
    ck(any(n.startswith("PFT all 1 arms") and s == "PASS" for n, s, _ in g.rows), "PFT passes for the DST arm with trace rows: %s" % [n for n, s, _ in g.rows if "PFT" in n])
    open(os.path.join(retd, "plane_fit_trace_counts_all.csv"), "w").write("arm,rows\nother,3\n")
    g, _ = run_gates(runs, mdir, ret_dirs=[retd], dispatched="d_ok")
    ck(any(n.startswith("PFT d_ok") and s == "MISSING" for n, s, _ in g.rows), "PFT: no rows for the arm is MISSING")
    # ---- no tool-internal ZUPT checks left
    ck(not re.search(r"ZCENG|ZCFILE|ZCCFG|TCENG|TCFILE|PVDENG|D5COMP", open(os.path.abspath(__file__)).read().split("def selftest")[0]), "no R82 ZUPT / time_cov / D5 gate names remain in the gate code")
    shutil.rmtree(root, ignore_errors=True)
    print("SELFTEST r83_gates", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (nck[0], len(fails)),
          "" if fails else "(DST / MXL / VSW / D1B / D4P / MQS / ZUPTGONE / ODOMEXP / NOEXTRA rows PASS on good fixtures and FAIL or MISSING on bad ones; DGTCFG reads the job on a Hilti arm; PTCFG not MISSING when the engagement line shows it; "
          "ABORTED_RSS FAIL writes no hold; DET and IDENT FAIL write HOLD_FROM_GATES.txt; MSGSCANS INFO for dc; synthetic fixtures, planner sandbox, not evidence)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--manifest", default="r83_manifest"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grid-retime", default=None); ap.add_argument("--grid-bag", default=None); ap.add_argument("--ref-return", action="append", default=[]); ap.add_argument("--ref-runs", action="append", default=[])
    ap.add_argument("--ref-manifest", action="append", default=[]); ap.add_argument("--phase-summary", default=None); ap.add_argument("--ret-dir", action="append", default=[])
    ap.add_argument("--dispatched", default=None); ap.add_argument("--hold-file", default=None); ap.add_argument("--no-famcount", action="store_true")
    ap.add_argument("--expect", default=None, help="family counts for FAMCOUNT, e.g. \"DST=12,MXL=36,VSW=234\" (a count may list alternatives with |); default: the SPEC table of this file")
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
