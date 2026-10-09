#!/usr/bin/env python3
"""R81 gates (planning agent's script; selftest on SYNTHETIC fixtures only, planner sandbox, not evidence).

    python3 tools/r69/r81_gates.py --runs <runs_dir> --manifest r81_manifest [--grid-retime r80c_grid.json] [--grid-bag r81_manifest/grid_bag.json] \
        [--ref-return R80_RETURN --ref-return R80C_RETURN] [--ref-runs <old archive dir>] [--phase-summary phase_summary.csv] [--ret-dir <return dir>] \
        [--dispatched <arm ids file | arm_table.tsv | comma list of arm ids / groups / families / queues>] [--out r81_gates.txt] [--csv r81_identity_vs_old.csv] [--hold-file HOLD_FROM_GATES.txt]
    python3 tools/r69/r81_gates.py --selftest

runs_dir/<arm>/ = the unbundled arm bundle of each arm (r69_unbundle.py): arm.json, exit_status.txt, run.log, engagement.txt, pos_trace_all.csv.gz, state_trace_blank_runid.sha256, lidar_msg_log_all.csv.gz, ...
The expectation of every arm is read from the MANIFEST ONLY (arm_table.tsv + jobs_*.json overrides), never from this file's own list of arms; the R81 build prints what it did, and every value found is
compared with the job's override (a two-way diff, rule 3.8). Grid values are NOT taken from the manifest: they are read from --grid-retime (the R80C grid, eight bags as shipped) and --grid-bag (the
R80D bag fit: eee_03) so that a wrong manifest cannot agree with itself.
Verdicts: PASS / FAIL / MISSING (absent: never a pass) / INFO (a fact) / NOTRUN (INFO row: the arm's queue was not dispatched). Exit 1 on a FAIL.

Fixes carried from the R80 false-FAILs (r81_tool_fixes.patch items 1 to 6 and r81-draft-notes): EXIT parses `exit_code=<N> signal=<NAME|none>` (a bare integer is still accepted); MSGKEY and every config key is read
case-insensitively from `key = v`, `key: v` and `/ns/key: v` lines, true / 1 and false / 0 are the same, a trailing `[FORCED: ...]` note is ignored, a dump that prints a rounded number matches when the want rounds to the printed digits;
the line-residual setting falls back to the per-key lines when the one-line dump is cut; every DET pair is judged; whole-bag DET repeats use the whole-bag length rule; an EC / short arm uses the manifest's own length rule.

Groups (arm_table column `group` = the SPEC family: SWT, RR, N478, N467, N470, N485B, N469, N468, N485C, N474, N479, OWN; column `fam` = the fine group GP, B3, B1, MG, ZD, NG, PC, DB, DP, EC3, SC, NIS, ZT, PCX, F2B, PCA,
VMIN, VS2, EF, SWX, SWC, ID, DET, SMK, REF; column `mode` = dc | rt; column `arm_kind` = id, det_fb, smoke, refuse, fb; `ref` = identity reference / determinism original; `needles` = `|` separated refusal texts; `msglog` = 1 for the BASE arms).
Rows (each names its arm or its cell, and is decided from that arm's own files only, rule 1.16):
  RUN EXIT                  every arm: arm.json sentinel DONE, exit_status.txt `exit_code=<N> signal=<NAME|none>` equal to 0 (refuse arms: non-zero and not DONE)
  CFG / NISCFG / VMINCFG / VS2CFG / VINFOCFG / SWXCFG / PFN / VOX / OVLCFG / MSGKEY   the effective configuration the build printed, key by key, against the job's overrides (and the defaults of the new keys)
  RTGRID                    rt arms: the retime keys of the job equal the bag's grid (r80c_grid.json for eight bags, the bag fit for eee_03), and the printed values equal them
  OVLENG                    rt arms print `[lidar_overlap_policy] policy=retime_trim` and no `inert when on` error; dc arms print no such line
  LEN                       whole bag >= 97 percent of the scans of the bag; 60 s arms 560 to 700; other short arms 5 to 10 x duration + 20 rows
  IDENT                     identity arms against the old returns (r81_identity_hold.py's own checks, tolerance 0.005 mm on position, excursion, settled offset; digest equality where both exist)
  DET / DETPOS              every determinism pair: state-trace digest equal, pos trace identical
  MSGLOG MSGRULE MSGGRID    BASE arms only (arm_table msglog = 1): the 15 / 17 column log, pushLidar's rule row by row, and for rt arms the grid (scan_time on the grid, retime_shift_s <= 1e-6, rows = messages of the bag, no verdict 1 row under retime_trim)
  ZDCFG ZDFILE QADAPT NGCFG SCCFG SCFILE NOEXTRA   engagement lines (ZUPT, gain, step cap) as the job asked, their files, and no engagement the arm did not ask for
  VMINENG VS2ENG            new-key engagement lines (plane_var_min / plane_var_scan_s2) with the clamp count; a smoke arm must show clamped > 0 (inert when on is an error), a production arm reports it as a fact
  SMOKE                     the four new-key smoke arms and the REFUSE arms (needles, sentinel FAILED, non-zero exit)
  SWCELL SWCOMP             SWT arms: configuration and exit folded per cell (bag, Q, lines, pfn: eight edges each), phase_summary row per arm
  EARLY F2BSUM N479COMP PFT FAMCOUNT   early_grid_summary.csv has frames 1..20 for every N474 arm; f485_2b_summary.csv has frames 400..410 with voxel columns for every N485B arm; N479 has per bag four switch arms and two controls;
                            plane_fit_trace_counts_all.csv has rows for every N469 and DB arm; the manifest's group counts equal the SPEC table (RR: 119, or 107 when the RRX arms were dropped)
A FAIL of a DET or IDENT row writes --hold-file (HOLD_FROM_GATES.txt): the agent places it as HOLD.txt (r81_hold_check.sh). The gate never deletes or cancels anything.
"""
import argparse, csv, glob, gzip, io, json, os, re, shutil, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r81_identity_hold as IH  # noqa: E402

LEN_MIN = 0.97
REF_OFFSET = -0.1
N_SCANS = {"eee_01": 3987, "eee_02": 3210, "eee_03": 1814, "nya_01": 3950, "nya_02": 4287, "nya_03": 4094, "sbs_01": 3542, "sbs_02": 3732, "sbs_03": 3893}   # message / scan counts of the bags (make_r79_manifest.N_SCANS)
EXPECT = {"SWT": (576,), "RR": (119, 107), "N478": (36,), "N467": (36,), "N470": (96,), "N485B": (8,), "N469": (4,), "N468": (32,), "N485C": (32,), "N474": (720,), "N479": (24,)}   # SPEC table; RR 107 = RRX dropped
EDGES_SWT = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MSG_COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict"]
MSG_COLS_C = MSG_COLS + ["retime_shift_s", "n_trimmed"]
ZD_COLS = ["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"]
SC_COLS = ["frame", "iter", "rot_raw", "pos_raw", "rot_applied", "pos_applied", "scale", "bound", "next_converged"]
EARLY_FRAMES = list(range(1, 21))
F2B_FRAMES = list(range(400, 411))
F2B_COLS = ["arm", "frame", "voxel_point_count", "voxel_distinct_frames"]
EARLY_COLS = ["arm", "frame"]
N479_PER_BAG = (4, 2)                        # switch arms (two directions x two modes), plain controls
SWT_RE = re.compile(r"^r81_rt_sw_(?P<bag>[a-z]+_\d\d)_(?P<line>off|hardabs)_v(?P<edge>[0-9p]+)_pfn(?P<pfn>\d)_(?P<q>fq|mq)$")
RETIME_KEYS = ("cbk.lidar.retime_t0_s", "cbk.lidar.retime_period_s", "cbk.lidar.retime_seq0", "cbk.lidar.retime_ref_offset_s")
STRICT = re.compile(r"^(voxel_map\.map\.(voxel_size|max_layer)|cbk\.lidar\.(point_filter_num|overlap_policy|retime_.*)|imu\.process_noise\.gain\..*|imu\.process_noise\.switch\..*|voxel_map\.plane\.(plane_var_min|plane_var_scan_s2)|"
                    r"lio\.residual_dump_voxel_info|calib\.stationary\.num_samples|data_queues\.log_lidar_msgs|voxel_map\.plane\.plane_fit_mode|voxel_map\.plane\.line_residual\.(enabled|form|rule|rows))$")
THEMES = [("OVLCFG", re.compile(r"^cbk\.lidar\.(overlap_policy|retime_.*)$")), ("MSGKEY", re.compile(r"^data_queues\.log_lidar_msgs$")), ("NISCFG", re.compile(r"^imu\.process_noise\.gain\..*$")),
          ("SWXCFG", re.compile(r"^imu\.process_noise\.switch\..*$")), ("VMINCFG", re.compile(r"^voxel_map\.plane\.plane_var_min$")), ("VS2CFG", re.compile(r"^voxel_map\.plane\.plane_var_scan_s2$")),
          ("VINFOCFG", re.compile(r"^lio\.residual_dump_voxel_info$")), ("PFN", re.compile(r"^cbk\.lidar\.point_filter_num$")), ("VOX", re.compile(r"^voxel_map\.map\.(voxel_size|max_layer)$"))]
DEFAULTS = {"voxel_map.plane.plane_var_min": 0.0, "voxel_map.plane.plane_var_scan_s2": 0.0, "lio.residual_dump_voxel_info": False, "cbk.lidar.overlap_policy": "discard",
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


def engagement_gates(g, t, ov, log, et, d):
    arm = t["id"]; ov = ov or {}; out = []
    have = bool(et.strip()) or bool(log.strip())
    # ZUPT detect_only (D7, F-467): thresholds as the job states them
    zl = line(et, "lio/zupt mode=")
    if ov.get("lio.zupt.mode"):
        z = kv(zl); sc = z.get("scans")
        items = [("mode", z.get("mode"), ov["lio.zupt.mode"]), ("gyro_thresh", z.get("gyro_thresh"), ov.get("lio.zupt.gyro_thresh")), ("acc_thresh", z.get("acc_thresh"), ov.get("lio.zupt.acc_thresh")),
                 ("enter_scans", z.get("enter_scans"), ov.get("lio.zupt.enter_scans")), ("evaluated (scans > 0)", None if sc is None else ("yes" if int(sc) > 0 else "no"), "yes")]
        if ov["lio.zupt.mode"] == "detect_only":
            items.append(("applied (detect_only never touches the state)", z.get("applied"), 0))
        items = [(a, b, c) for a, b, c in items if c is not None]
        bad = ["%s: found %s, wanted %s" % (a, b, c) for a, b, c in items if b is not None and not val_match(b, c, a)]
        miss = ["%s: not found" % a for a, b, c in items if b is None]
        out.append(("ZDCFG %s lio/zupt as the engagement line states it (thresholds from the job)" % arm, "FAIL" if bad else ("MISSING" if miss or not zl else "PASS"), "; ".join(bad + miss) or zl.strip()))
    # gain (NG nis_gate: F-476, N478 nis_smooth: F-478)
    gl = line(et, "imu/process_noise/gain mode=")
    gm = ov.get("imu.process_noise.gain.mode")
    if gm in ("nis_gate", "nis_smooth"):
        z = kv(gl); sc = z.get("scans")
        items = [("mode", z.get("mode"), gm), ("evaluated (scans > 0)", None if sc is None else ("yes" if int(sc) > 0 else "no"), "yes")]
        if gm == "nis_gate" and gl:
            items.append(("summary line prints deflating_scans and min_gain_seen", "yes" if ("deflating_scans" in z and "min_gain_seen" in z) else "no", "yes"))
        bad = ["%s: found %s, wanted %s" % (a, b, c) for a, b, c in items if b is not None and not val_match(b, c, a)]
        miss = ["%s: not found" % a for a, b, c in items if b is None]
        tag = "NGCFG" if gm == "nis_gate" else "NISENG"
        st = "FAIL" if bad else ("MISSING" if miss else "PASS")
        if not gl and have and et.strip():
            st = "FAIL"; bad.append("no imu/process_noise/gain engagement line")
        out.append(("%s %s gain mode %s engaged by the gain line" % (tag, arm, gm), st, "; ".join(bad + miss) or gl.strip()[:200]))
        if gl:
            out.append(("NGINFO %s scans with gain != 1: %s (a fact: a gain that never moves is a result, not a defect)" % (arm, z.get("scans_with_gain_ne_1")), "INFO", "max_gain %s min_gain_seen %s" % (z.get("max_gain"), z.get("min_gain_seen"))))
    # step cap (D5)
    cl = line(et, "lio/ekf/step_cap mode=")
    if ov.get("lio.ekf.step_cap.mode"):
        z = kv(cl); it = z.get("iterations")
        items = [("mode", z.get("mode"), ov["lio.ekf.step_cap.mode"]), ("cap_rot", z.get("cap_rot"), ov.get("lio.ekf.step_cap.rotation_rad")), ("cap_pos", z.get("cap_pos"), ov.get("lio.ekf.step_cap.position_m")),
                 ("evaluated (iterations > 0)", None if it is None else ("yes" if int(it) > 0 else "no"), "yes")]
        items = [(a, b, c) for a, b, c in items if c is not None]
        bad = ["%s: found %s, wanted %s" % (a, b, c) for a, b, c in items if b is not None and not val_match(b, c, a)]
        miss = ["%s: not found" % a for a, b, c in items if b is None]
        out.append(("SCCFG %s step cap as the engagement line states it" % arm, "FAIL" if bad else ("MISSING" if miss or not cl else "PASS"), "; ".join(bad + miss) or cl.strip()[:200]))
    # plane variance keys (F-468 / F-485 Part 3): the engagement line of agent A: `voxel_map/plane/plane_var_min = .. plane_var_scan_s2 = .. rows= scan_rows= mean_n_scans= clamped= clamped_fraction=`, only when a key is on
    vl = next((l for l in et.splitlines() if re.search(r"plane_var_min\s*=.*\brows\s*=", l)), None)   # the engagement line, not the `key = value` config echo
    vmin, vs2 = ov.get("voxel_map.plane.plane_var_min", 0.0), ov.get("voxel_map.plane.plane_var_scan_s2", 0.0)
    on = (vmin or 0) > 0 or (vs2 or 0) > 0
    if on:
        m = re.search(r"clamped\s*=\s*(\d+)", vl or ""); clamped = int(m.group(1)) if m else None
        rowsm = re.search(r"\brows\s*=\s*(\d+)", vl or ""); nr = int(rowsm.group(1)) if rowsm else None
        inert = "<-- INERT" in (vl or "")
        smoke = t["kind"] == "smoke"
        bad = []
        if vl is None:
            bad.append("no plane_var engagement line although a key is on")
        else:
            if nr is not None and nr <= 0:
                bad.append("rows = %d (the term never evaluated)" % nr)
            if smoke and ((vmin or 0) > 0) and (clamped is None or clamped <= 0):
                bad.append("smoke arm: clamped = %s (the clamp never fired: inert when on)" % clamped)
            if smoke and (vs2 or 0) > 0 and "mean_n_scans" in (vl or "") and (num((re.search(r"mean_n_scans\s*=\s*([0-9.eE+-]+)", vl) or [0, 0])[1]) or 0) <= 0:
                bad.append("smoke arm: mean_n_scans <= 0 (the scan term never applied)")
        tag = "VS2ENG" if (vs2 or 0) > 0 else "VMINENG"
        out.append(("%s %s plane variance engagement line (rows > 0%s)" % (tag, arm, "; smoke arm: clamp fired" if smoke else ""), "FAIL" if bad else ("MISSING" if vl is None and not have else "PASS"), "; ".join(bad) or (vl or "").strip()[:240]))
        if vl is not None and clamped is not None and not smoke:
            out.append(("VMININFO %s clamped rows %s (a fact%s)" % (arm, clamped, "; INERT flag printed by the build" if inert else ""), "INFO", (vl or "").strip()[:240]))
    # nothing the arm did not ask for
    extra = []
    if zl and not ov.get("lio.zupt.mode"):
        extra.append("ZUPT")
    if gl and gm not in ("nis_gate", "nis_smooth"):
        extra.append("process-noise gain")
    if cl and not ov.get("lio.ekf.step_cap.mode"):
        extra.append("step cap")
    if vl and not on:
        extra.append("plane variance keys")
    out.append(("NOEXTRA %s no ZUPT, gain, step-cap or plane-variance engagement line that the arm did not ask for" % arm, "FAIL" if extra else ("PASS" if have else "MISSING"), ", ".join(extra)))
    return out


def file_gates(g, t, d, ov):
    arm, fam = t["id"], t["fam"]
    if fam in ("ZT", "ZD"):
        Z = csvf(d, "zupt_all.csv.gz")
        nm = "ZDFILE %s zupt_all.csv.gz with columns %s, at least 100 rows, applied always 0" % (arm, ", ".join(ZD_COLS))
        if Z is None:
            g.add(nm, "MISSING")
        elif [c for c in ZD_COLS if c not in Z.columns]:
            g.add(nm, "FAIL", "missing columns %s" % [c for c in ZD_COLS if c not in Z.columns])
        else:
            napp = int((pd.to_numeric(Z["applied"], errors="coerce") != 0).sum())
            g.add(nm, "PASS" if len(Z) >= 100 and napp == 0 else "FAIL", "rows %d, rows with applied != 0: %d" % (len(Z), napp))
            st = pd.to_numeric(Z["stationary"], errors="coerce")
            g.add("ZDFACT %s stationary scans (a fact)" % arm, "INFO", "%d of %d" % (int((st == 1).sum()), len(Z)))
    if fam in ("NIS", "NG"):
        Q = csvf(d, "q_adapt_all.csv.gz")
        nm = "QADAPT %s q_adapt_all.csv.gz with at least 100 rows" % arm
        g.add(nm, "MISSING" if Q is None else ("PASS" if len(Q) >= 100 else "FAIL"), "" if Q is None else "rows %d" % len(Q))
    if fam == "SC":
        S = csvf(d, "step_cap.csv.gz")
        nm = "SCFILE %s step_cap.csv.gz with columns %s and at least 20 rows" % (arm, ", ".join(SC_COLS))
        if S is None:
            g.add(nm, "MISSING")
        elif [c for c in SC_COLS if c not in S.columns]:
            g.add(nm, "FAIL", "missing columns")
        else:
            g.add(nm, "PASS" if len(S) >= 20 else "FAIL", "rows %d" % len(S))
    if t["kind"] == "smoke" and (ov or {}).get("lio.residual_dump_voxel_info"):
        D = csvf(d, "residual_dump_all.csv.gz")
        nm = "SMOKE %s residual_dump has voxel_point_count and voxel_distinct_frames, non-zero (lio/residual_dump_voxel_info)" % arm
        if D is None:
            g.add(nm, "MISSING", "residual_dump_all.csv.gz absent")
        elif "voxel_point_count" not in D.columns or "voxel_distinct_frames" not in D.columns:
            g.add(nm, "FAIL", "columns absent: %s" % [c for c in ("voxel_point_count", "voxel_distinct_frames") if c not in D.columns])
        else:
            nz = int((pd.to_numeric(D["voxel_point_count"], errors="coerce") > 0).sum()), int((pd.to_numeric(D["voxel_distinct_frames"], errors="coerce") > 0).sum())
            g.add(nm, "PASS" if min(nz) > 0 else "FAIL", "rows %d, voxel_point_count > 0 in %d, voxel_distinct_frames > 0 in %d" % (len(D), nz[0], nz[1]))
    if t["kind"] == "smoke" and (ov or {}).get("imu.process_noise.gain.mode") == "nis_smooth":
        Q = csvf(d, "q_adapt_all.csv.gz")
        g.add("SMOKE %s nis_smooth: q_adapt rows present (a fact)" % arm, "INFO", "q_adapt_all.csv.gz %s" % ("absent" if Q is None else "rows %d" % len(Q)))


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
    """SWT arms: FAIL / MISSING rows stay one per arm; the PASS rows of a (gate, cell) fold into one row (the report stays inside its size ledger)."""

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
        is_sw = grp == "SWT"
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
            continue                    # SWT: summaries only; no pos_trace is required (a trace only when gt_max_mm > 500), no engagement gates
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
    """EARLY, F2BSUM, N479COMP, PFT: group-level completeness from files of the return (found in ret_dirs)."""
    dirs = list(ret_dirs) + [runs, os.path.join(runs, "..")]
    arms = lambda f: [t for t in table if f(t) and t["id"] in live]
    # N474 early grid
    n474 = arms(lambda t: t["group"] == "N474")
    if n474:
        p = find_return_file("early_grid_summary.csv", dirs)
        E = pd.read_csv(p) if p else None
        if E is None or any(c not in E.columns for c in EARLY_COLS):
            g.add("EARLY early_grid_summary.csv with columns %s for the %d N474 arms" % (", ".join(EARLY_COLS), len(n474)), "MISSING", "file absent or columns missing")
        else:
            fr = E.groupby("arm")["frame"].apply(lambda s: set(int(x) for x in s))
            bad = 0
            for t in n474:
                got = fr.get(t["id"], set())
                if not set(EARLY_FRAMES) <= got:
                    bad += 1
                    g.add("EARLY %s early_grid_summary.csv has frames 1..20" % t["id"], "MISSING" if not got else "FAIL", "frames present %s" % (sorted(got)[:25] or "none"))
            if not bad:
                g.add("EARLY all %d N474 arms have frames 1..20 in early_grid_summary.csv" % len(n474), "PASS")
    # N485B ground-row dump summary
    f2b = arms(lambda t: t["group"] == "N485B")
    if f2b:
        p = find_return_file("f485_2b_summary.csv", dirs)
        F = pd.read_csv(p) if p else None
        if F is None or any(c not in F.columns for c in F2B_COLS):
            g.add("F2BSUM f485_2b_summary.csv with columns %s for the %d N485B arms" % (", ".join(F2B_COLS), len(f2b)), "MISSING", "file absent or columns missing")
        else:
            bad = 0
            for t in f2b:
                s = F[F["arm"] == t["id"]]
                fr = set(int(x) for x in s["frame"])
                ok = set(F2B_FRAMES) <= fr and (pd.to_numeric(s["voxel_point_count"], errors="coerce") > 0).any() and (pd.to_numeric(s["voxel_distinct_frames"], errors="coerce") > 0).any()
                if not ok:
                    bad += 1
                    g.add("F2BSUM %s f485_2b_summary.csv has frames 400..410 and non-zero voxel_point_count / voxel_distinct_frames" % t["id"], "MISSING" if s.empty else "FAIL", "frames %s" % sorted(fr))
            if not bad:
                g.add("F2BSUM all %d N485B arms have frames 400..410 with voxel columns" % len(f2b), "PASS")
    # N479 completeness: per bag four switch arms and two controls (manifest) and every one of them dispatched and present
    n479 = [t for t in table if t["group"] == "N479"]
    if n479 and any(t["id"] in live for t in n479):
        for bag in sorted({t["bag"] for t in n479}):
            ts = [t for t in n479 if t["bag"] == bag]
            sw_ = [t for t in ts if (t["ov"] or {}).get("imu.process_noise.switch.scan") is not None]
            ct = [t for t in ts if (t["ov"] or {}).get("imu.process_noise.switch.scan") is None]
            dirs_ok = [t for t in ts if os.path.exists(os.path.join(runs, t["id"], "pos_trace_all.csv.gz"))]
            ok = (len(sw_), len(ct)) == N479_PER_BAG and len(dirs_ok) == len(ts)
            g.add("N479COMP %s: %d switch arms and %d plain controls in the manifest (want %d and %d), pos trace present for %d of %d" % (bag, len(sw_), len(ct), N479_PER_BAG[0], N479_PER_BAG[1], len(dirs_ok), len(ts)),
                  "PASS" if ok else ("MISSING" if len(dirs_ok) < len(ts) and (len(sw_), len(ct)) == N479_PER_BAG else "FAIL"), "missing: %s" % ", ".join(t["id"] for t in ts if t not in dirs_ok))
        p = find_return_file("r81_switch_pairs.csv", dirs)
        if p:
            S = pd.read_csv(p)
            have = set(S["arm"]) if "arm" in S.columns else set()
            miss = [t["id"] for t in n479 if (t["ov"] or {}).get("imu.process_noise.switch.scan") is not None and t["id"] not in have and t["id"] in live]
            g.add("N479COMP r81_switch_pairs.csv has a row for every switch arm (names as the manifest has them)", "PASS" if not miss else "FAIL", "missing: %s" % ", ".join(miss[:12]))
        else:
            g.add("N479COMP r81_switch_pairs.csv has a row for every switch arm (names as the manifest has them)", "MISSING", "r81_switch_pairs.csv not found in the return files")
    # F-469 plane-fit trace counts
    pft = arms(lambda t: t["fam"] in ("PCA", "DB"))
    if pft:
        p = find_return_file("plane_fit_trace_counts_all.csv", dirs)
        C = pd.read_csv(p) if p else None
        if C is None or "arm" not in C.columns:
            g.add("PFT plane_fit_trace_counts_all.csv has rows for the %d N469 / DB arms (written by the collector)" % len(pft), "MISSING", "file absent")
        else:
            bad = 0
            for t in pft:
                n = int(C[C["arm"] == t["id"]]["rows"].sum()) if "rows" in C.columns else 0
                if n <= 0:
                    bad += 1; g.add("PFT %s has rows in plane_fit_trace_counts_all.csv" % t["id"], "MISSING", "no rows")
            if not bad:
                g.add("PFT all %d N469 / DB arms have rows in plane_fit_trace_counts_all.csv" % len(pft), "PASS")


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
FIX_N = {"eee_01": 300, "eee_03": 200}
FIX_GRID_RETIME = {"eee_01": {"bag": "eee_01", "n": 307, "seq0": 119, "t0_s": 1609059013.0668645, "period_s": 0.099997231748, "seq_gaps": 0}}
FIX_GRID_BAG = {"eee_03": {"bag": "eee_03", "n": 207, "seq0": 50, "seq_gaps": 0, "retime": {"t0_s": 1609060000.25, "period_s": 0.0999981}}}


def fx_row(arm, group, fam, mode, kind="fb", ref="", bag="eee_01", batch="r81_eee_01", pfn=3, dur="full", msglog=0, needles="", ov=None):
    return dict(arm=arm, group=group, fam=fam, mode=mode, arm_kind=kind, ref=ref, bag=bag, seq=bag, batch=batch, pfn=str(pfn), duration_s=str(dur), msglog=str(msglog), needles=needles, ov=ov or {})


def fx_ov(bag="eee_01", mode="dc", pfn=3, extra=None, edge=0.425):
    o = {"voxel_map.map.voxel_size": edge, "voxel_map.map.max_layer": 2, "cbk.lidar.point_filter_num": pfn, "voxel_map.plane.plane_fit_mode": "pca", "imu.process_noise.model": "fixed", "evo.enable": True}
    if mode == "rt":
        gr = grid_of(bag, FIX_GRID_RETIME, FIX_GRID_BAG)
        o.update({"cbk.lidar.overlap_policy": "retime_trim", "cbk.lidar.retime_t0_s": gr["t0_s"], "cbk.lidar.retime_period_s": gr["period_s"], "cbk.lidar.retime_seq0": gr["seq0"],
                  "cbk.lidar.retime_ref_offset_s": REF_OFFSET, "data_queues.log_lidar_msgs": True})
    o.update(extra or {})
    return o


NIS = {"imu.process_noise.gain.mode": "nis_smooth", "imu.process_noise.gain.nis_slope": 0.1, "imu.process_noise.gain.nis_clip": 0.6931471805599453, "imu.process_noise.gain.decay": 0.95, "imu.process_noise.gain.min_gain": 0.25}
NG = {"imu.process_noise.gain.mode": "nis_gate", "imu.process_noise.gain.nis_hi": 1.3, "imu.process_noise.gain.inflate": 1.25, "imu.process_noise.gain.decay": 0.95, "imu.process_noise.gain.nis_lo": 0.75,
      "imu.process_noise.gain.deflate": 0.8, "imu.process_noise.gain.min_gain": 0.25}
SWI = {"imu.process_noise.switch.scan": 481, "imu.process_noise.switch.model": "isotropic", "imu.process_noise.switch.motion_beta": 0.9}


def fixture_rows():
    R = []
    add = lambda *a, **k: R.append(fx_row(*a, **k))
    add("r81_dc_id_mg_eee_01_fq", "OWN", "ID", "dc", "id", "r80_mg_eee_01_fq", batch="r81_id", msglog=1, ov=fx_ov("eee_01", "dc", extra={"data_queues.log_lidar_msgs": True}))
    add("r81_rt_id_mg_eee_01_fq", "OWN", "ID", "rt", "id", "r80c_rtt_mg_eee_01_fq", batch="r81_id", msglog=1, ov=fx_ov("eee_01", "rt"))
    add("r81_rt_mg_eee_01_fq", "RR", "MG", "rt", msglog=1, ov=fx_ov("eee_01", "rt"))
    add("r81_own_det_x_eee_01", "OWN", "DET", "rt", "det_fb", "r81_rt_mg_eee_01_fq", ov=fx_ov("eee_01", "rt"))
    add("r81_rt_b3_eee_03_mq", "RR", "B3", "rt", bag="eee_03", batch="r81_eee_03", msglog=1, ov=fx_ov("eee_03", "rt"))
    add("r81_dc_nis_eee_01_fq", "N478", "NIS", "dc", ov=fx_ov("eee_01", "dc", extra=NIS))
    add("r81_rt_nis_eee_01_fq", "N478", "NIS", "rt", ov=fx_ov("eee_01", "rt", extra=NIS))
    add("r81_dc_zt_eee_01_a025", "N467", "ZT", "dc", ov=fx_ov("eee_01", "dc", extra={"lio.zupt.mode": "detect_only", "lio.zupt.gyro_thresh": 0.02, "lio.zupt.acc_thresh": 0.25, "lio.zupt.enter_scans": 3}))
    add("r81_rt_zd_eee_01_fq", "RR", "ZD", "rt", ov=fx_ov("eee_01", "rt", extra={"lio.zupt.mode": "detect_only", "lio.zupt.gyro_thresh": 0.02, "lio.zupt.acc_thresh": 0.15, "lio.zupt.enter_scans": 3}))
    add("r81_rt_ng_eee_01_fq", "RR", "NG", "rt", ov=fx_ov("eee_01", "rt", extra=NG))
    add("r81_rt_sc_fq", "RR", "SC", "rt", dur=60, batch="r81_eee_01_s", ov=fx_ov("eee_01", "rt", extra={"lio.ekf.step_cap.mode": "hard", "lio.ekf.step_cap.rotation_rad": 0.01, "lio.ekf.step_cap.position_m": 0.01}))
    add("r81_dc_pcx_eee_03_ps_fq", "N470", "PCX", "dc", bag="eee_03", batch="r81_eee_03", ov=fx_ov("eee_03", "dc", extra={"voxel_map.residual.pose_cov_in_sigma": True}))
    add("r81_dc_pca_fq", "N469", "PCA", "dc", ov=fx_ov("eee_01", "dc", extra={"voxel_map.plane.plane_fit_reasons_en": True, "voxel_map.plane.plane_fit_trace_frames": 10}))
    add("r81_rt_db_fq", "RR", "DB", "rt", ov=fx_ov("eee_01", "rt", extra={"voxel_map.plane.plane_fit_mode": "debiased", "voxel_map.plane.plane_fit_reasons_en": True, "voxel_map.plane.plane_fit_trace_frames": 10}))
    add("r81_dc_vmin_eee_01_vm400_pfn3_fq", "N468", "VMIN", "dc", ov=fx_ov("eee_01", "dc", extra={"voxel_map.plane.plane_var_min": 4.0e-4}))
    add("r81_rt_vs2_eee_01_s200_pfn3_fq", "N485C", "VS2", "rt", ov=fx_ov("eee_01", "rt", extra={"voxel_map.plane.plane_var_min": 4.0e-4, "voxel_map.plane.plane_var_scan_s2": 2.0e-4}))
    add("r81_dc_f2b_pfn3_fq", "N485B", "F2B", "dc", dur=60, batch="r81_eee_01_s", ov=fx_ov("eee_01", "dc", extra={"lio.residual_dump_frames": ",".join(str(i) for i in F2B_FRAMES), "lio.residual_dump_voxel_info": True}))
    for m in ("dc", "rt"):
        add("r81_%s_ef_eee_01_n200_dsfirst_pfn3_fq" % m, "N474", "EF", m, dur=10, batch="r81_eee_01_s", ov=fx_ov("eee_01", m, extra={"calib.stationary.num_samples": 200, "imu.ds.mode": "first"}))
        add("r81_%s_swx_eee_01_fq2mq" % m, "N479", "SWX", m, ov=fx_ov("eee_01", m, extra=SWI))
        add("r81_%s_swx_eee_01_mq2fq" % m, "N479", "SWX", m, ov=fx_ov("eee_01", m, extra=dict(SWI, **{"imu.process_noise.switch.model": "fixed"})))
    for q in ("fq", "mq"):
        add("r81_dc_swc_eee_01_%s" % q, "N479", "SWC", "dc", ov=fx_ov("eee_01", "dc"))
    for e, tg in zip(EDGES_SWT, ("0p25", "0p3", "0p35", "0p4", "0p45", "0p5", "0p55", "0p6")):
        add("r81_rt_sw_eee_01_off_v%s_pfn3_fq" % tg, "SWT", "SW", "rt", ov=fx_ov("eee_01", "rt", edge=e))
    smk = lambda a, extra: add(a, "OWN", "SMK", "dc", "smoke", dur=60, batch="r81_id", ov=fx_ov("eee_01", "dc", extra=extra))
    smk("r81_dc_sm_nis_60s", NIS)
    smk("r81_dc_sm_vmin_60s", {"voxel_map.plane.plane_var_min": 4.0e-4})
    smk("r81_dc_sm_f2b_60s", {"lio.residual_dump_frames": "400,401,402", "lio.residual_dump_voxel_info": True})
    add("r81_dc_refuse_plane_var_min_negative", "OWN", "REF", "dc", "refuse", pfn=3, dur=5, batch="r81_id", needles="voxel_map/plane/plane_var_min", ov={"voxel_map.plane.plane_var_min": -1e-4})
    return R


def fx_write_manifest(mdir, rows):
    os.makedirs(mdir, exist_ok=True)
    cols = ["arm", "group", "fam", "mode", "arm_kind", "ref", "bag", "seq", "batch", "pfn", "duration_s", "msglog", "needles", "kind", "flag"]
    jobs = {}
    with open(os.path.join(mdir, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t"); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "closed" if c == "kind" else "") for c in cols})
            jobs.setdefault(r["batch"], []).append({"job_id": r["arm"], "overrides": r["ov"]})
    for b, lst in jobs.items():
        json.dump(lst, open(os.path.join(mdir, "jobs_%s.json" % b), "w"))


def fx_log(t, ov, o):
    """the lines a build prints for ONE arm (formats of the real code: [params] dump `key: value`, [config/effective] echo `key = value`, a ROS dump line); o = the breakage switches that apply."""
    L = ["/livo_recon/voxel_map/plane/plane_gap_floor_en: True", "[params/voxel_map]", "  map/voxel_size:                  %s" % o.get("vox", ov["voxel_map.map.voxel_size"]),
         "  map/max_layer:                   %d" % ov["voxel_map.map.max_layer"], "  plane/plane_fit_mode:            %s" % o.get("fit", ov["voxel_map.plane.plane_fit_mode"])]
    vm, s2 = o.get("vmin", ov.get("voxel_map.plane.plane_var_min", 0.0)), o.get("s2", ov.get("voxel_map.plane.plane_var_scan_s2", 0.0))
    L += ["  plane/plane_var_min: %s %s" % (vm, "(R81 ENGAGED clamp)" if vm else "(off)"), "  plane/plane_var_scan_s2: %s %s" % (s2, "(R81 ENGAGED scan term)" if s2 else "(off)"),
          "  cbk/lidar/point_filter_num: %d" % o.get("pfn", ov["cbk.lidar.point_filter_num"]), "  cbk/lidar/overlap_policy: %s" % o.get("policy", ov.get("cbk.lidar.overlap_policy", "discard"))]
    if t["mode"] == "rt" or o.get("policy"):
        L += ["  cbk/lidar/retime_t0_s: %.9f" % o.get("t0", ov.get("cbk.lidar.retime_t0_s", 0.0)), "  cbk/lidar/retime_period_s: %.9f" % ov.get("cbk.lidar.retime_period_s", 0.0),
              "  cbk/lidar/retime_seq0: %d" % ov.get("cbk.lidar.retime_seq0", 0), "  cbk/lidar/retime_ref_offset_s: %s" % ov.get("cbk.lidar.retime_ref_offset_s", -0.1)]
    L.append("  data_queues/log_lidar_msgs: %s" % o.get("msgkey", "true" if ov.get("data_queues.log_lidar_msgs") else "false"))
    if not o.get("noline"):
        L.append("  plane/line_residual:             enabled %s / form hard / rule absolute / rel_sigmas 3.0 / shape_ratio 0.1 / enum 0 / variance fit_only / rows per_normal" % ("true" if ov.get("voxel_map.plane.line_residual.enabled") else "false"))
    L.append("[config/effective]  what this run actually does")
    brk = {"voxel_map.map.voxel_size": o.get("vox"), "cbk.lidar.point_filter_num": o.get("pfn"), "data_queues.log_lidar_msgs": o.get("msgkey"), "cbk.lidar.overlap_policy": o.get("policy"),
           "cbk.lidar.retime_t0_s": o.get("t0"), "voxel_map.plane.plane_var_min": o.get("vmin"), "voxel_map.plane.plane_var_scan_s2": o.get("s2")}   # a breakage is printed by every line that prints the key
    for k, v in sorted(ov.items()):
        if v is None or SKIP_KEYS.match(k):
            continue
        k2 = k.replace(".", "/")
        if brk.get(k) is not None:
            v = brk[k]
        if k in o.get("echo", {}):
            v = o["echo"][k]
        L.append("  %s = %s%s" % (k2, ("1" if v else "0") if isinstance(v, bool) else v, "  [FORCED: test]" if k.endswith("decay") else ""))
    if t["mode"] == "rt" and not o.get("nopolicyline"):
        L += ["[lidar_overlap_policy] policy=retime_trim msgs=1 mean_shift_ms=4.1 max_abs_shift_ms=9.0", "[lidar_overlap_policy] retime_trim msgs=1 trimmed_msgs=0 trimmed_points=0"]
        if o.get("inert"):
            L.append("[lidar_overlap_policy] policy=retime_trim is on but no scan time moved in 100 messages (inert when on)")
    if t["mode"] == "dc" and o.get("dcpolicyline"):
        L.append("[lidar_overlap_policy] policy=retime_trim msgs=1 mean_shift_ms=4.1 max_abs_shift_ms=9.0")
    return "\n".join(L) + "\n"


def fx_eng(t, ov, o):
    L = ["[engagement]  did each flag actually do anything?"]
    if ov.get("lio.zupt.mode") and not o.get("nozupt"):
        L.append("  lio/zupt mode=%s gyro_thresh=%s acc_thresh=%s enter_scans=%s sigma_v=0.02 scans=290 stationary=40 applied=%d" % (ov["lio.zupt.mode"], ov["lio.zupt.gyro_thresh"], o.get("zacc", ov["lio.zupt.acc_thresh"]), ov["lio.zupt.enter_scans"], o.get("zapplied", 0)))
    gm = ov.get("imu.process_noise.gain.mode")
    if gm and not o.get("nogain"):
        tail = " inflating_scans=7 deflating_scans=12 min_gain_seen=0.4" if gm == "nis_gate" else ""
        L.append("  imu/process_noise/gain mode=%s apply=both scans=290 scans_with_gain_ne_1=19 max_gain=1.9 mean_nis=1.1%s" % (o.get("gmode", gm), tail))
    if o.get("extra_gain"):
        L.append("  imu/process_noise/gain mode=nis_gate apply=both scans=290 scans_with_gain_ne_1=3")
    if ov.get("lio.ekf.step_cap.mode"):
        L.append("  lio/ekf/step_cap mode=hard cap_rot=%s rad cap_pos=0.01 m iterations=900 bound=4" % o.get("caprot", 0.01))
    vm, s2 = ov.get("voxel_map.plane.plane_var_min", 0.0), ov.get("voxel_map.plane.plane_var_scan_s2", 0.0)
    if (vm or s2) and not o.get("novline"):
        L.append("  voxel_map/plane/plane_var_min = %s plane_var_scan_s2 = %s rows=%d scan_rows=%d mean_n_scans=%.2f clamped=%d clamped_fraction=0.31%s" % (
            vm, s2, o.get("vrows", 5000), 5000 if s2 else 0, 3.2, o.get("clamped", 1500 if vm else 0), "  <-- INERT" if o.get("clamped", 1) == 0 else ""))
    if o.get("extra_vline") and not (vm or s2):
        L.append("  voxel_map/plane/plane_var_min = 0 plane_var_scan_s2 = 0 rows=1 scan_rows=0 mean_n_scans=0 clamped=0 clamped_fraction=0")
    return "\n".join(L) + "\n"


def fx_msglog(t, n_msg, n_scans, gr, o):
    k = np.arange(n_msg)
    seq0 = gr["seq0"] if gr else 0
    sc = (gr["t0_s"] + k * gr["period_s"]) if (gr and t["mode"] == "rt") else 100 + k * 0.1
    L = pd.DataFrame({"msg_index": k, "seq": seq0 + k, "header_stamp": sc + 0.1, "scan_time": sc, "gap_header_s": 0.1, "n_raw": 20000, "n_kept": 6600, "t_first": sc, "t_last": sc + 0.1, "off_first_s": 0.0, "off_last_s": 0.1,
                      "span_s": 0.1, "prev_accepted_last_t": sc - 0.1, "overlap_s": -0.1, "verdict": 0})
    if t["mode"] == "rt":
        L["retime_shift_s"] = -0.01; L["n_trimmed"] = 0
    if o.get("msg_drop_rt"):
        L.loc[3, "verdict"] = 1; L.loc[3, "overlap_s"] = 0.002
    if o.get("msgrule"):
        L.loc[3, "verdict"] = 1; L.loc[3, "overlap_s"] = -0.01
    if o.get("msg_late"):
        L.loc[5, "retime_shift_s"] = 0.002
    if o.get("msg_grid"):
        L.loc[7, "scan_time"] += 0.001
    if o.get("msg_short"):
        L = L.iloc[:-3]
    if o.get("msg_col"):
        L = L.drop(columns=["off_last_s"])
    if o.get("msg_order"):
        L["msg_index"] = L["msg_index"].iloc[::-1].to_numpy()
    return L


def fx_arm(root, t, o, refs):
    """one fake arm dir of the arm table row t; o = breakage switches for THIS arm."""
    arm, ov = t["id"], t["ov"]
    d = os.path.join(root, arm); os.makedirs(d, exist_ok=True)
    w = lambda name, text: open(os.path.join(d, name), "w").write(text)
    if t["kind"] == "refuse":
        w("arm.json", json.dumps({"sentinel": "DONE" if o.get("ref_done") else "FAILED"}))
        w("run_log_tail.txt", "terminate called: " + ("something else" if o.get("ref_needle") else " and ".join(t["needles"])) + "\n")
        if not o.get("ref_noexit"):
            w("exit_status.txt", "exit_code=0 signal=none\n" if o.get("ref_zero") else "exit_code=-6 signal=SIGABRT\n")
        return
    w("arm.json", json.dumps({"sentinel": "FAILED" if o.get("failed") else "DONE"}))
    if not o.get("exit_absent"):
        w("exit_status.txt", "abc\n" if o.get("exit_bad") else ("exit_code=3 signal=none\n" if o.get("exit_nonzero") else "exit_code=0 signal=none\n"))
    w("state_trace_blank_runid.sha256", ("b" if o.get("det") else "a") * 64 + "\n" if t["kind"] == "det_fb" else ("c" if o.get("digest_id") else "a") * 64 + "\n")
    w("run.log", fx_log(t, ov, o)); w("engagement.txt", fx_eng(t, ov, o))
    if t["group"] == "SWT":
        return
    n = FIX_N[t["bag"]]
    if t["dur"] == 60:
        n = 595
    elif t["dur"]:
        n = 90
    if o.get("len_short"):
        n = 40
    P = pd.DataFrame({"scan_id": np.arange(1, n + 1), "px": np.cumsum(np.full(n, 1e-3)), "py": 0.0, "pz": 0.0, "yaw_rad": 0.01})
    if t["kind"] == "det_fb" and o.get("detpos"):
        P.loc[50, "px"] += 1e-9
    if t["kind"] == "id" and o.get("identpos"):
        P.loc[P.scan_id >= 20, "px"] += 0.01
    P.to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    json.dump({"excursion_mm": 82.003503, "settled_offset_mm": 65.332038}, open(os.path.join(d, "arm_summary.json"), "w"))
    if t["msglog"] and not o.get("msg_absent"):
        gr = grid_of(t["bag"], FIX_GRID_RETIME, FIX_GRID_BAG)
        n_msg = gr["n"] if t["mode"] == "rt" else n + 7
        fx_msglog(t, n_msg, n, gr, o).to_csv(os.path.join(d, "lidar_msg_log_all.csv.gz"), index=False, compression="gzip")
    fam = t["fam"]
    if fam in ("ZT", "ZD") and not o.get("zd_nofile"):
        Z = pd.DataFrame({"frame": np.arange(1, 301), "gyro_mean": 0.01, "acc_mean": 0.05, "run": np.arange(300) % 5, "stationary": (np.arange(300) % 3 == 0).astype(int), "applied": 0, "vnorm_before": 0.001})
        if o.get("zd_applied"):
            Z.loc[5, "applied"] = 1
        Z.to_csv(os.path.join(d, "zupt_all.csv.gz"), index=False, compression="gzip")
    if fam in ("NIS", "NG") and not o.get("q_nofile"):
        pd.DataFrame({"frame": np.arange(1, 301), "gain": 1.0}).to_csv(os.path.join(d, "q_adapt_all.csv.gz"), index=False, compression="gzip")
    if fam == "SC":
        pd.DataFrame({c: np.arange(30) for c in SC_COLS}).to_csv(os.path.join(d, "step_cap.csv.gz"), index=False, compression="gzip")
    if t["kind"] == "smoke" and ov.get("lio.residual_dump_voxel_info"):
        D = pd.DataFrame({"frame": 400, "voxel_point_count": 0 if o.get("dump_zero") else 12, "voxel_distinct_frames": 3, "r": 0.1}, index=range(20))
        if o.get("dump_nocol"):
            D = D.drop(columns=["voxel_point_count", "voxel_distinct_frames"])
        D.to_csv(os.path.join(d, "residual_dump_all.csv.gz"), index=False, compression="gzip")


def fx_refs(root, rows):
    """old returns: R80 and R80C, holding exactly what the (good) identity arms wrote."""
    out = []
    for name, refarm in (("R80", "r80_mg_eee_01_fq"), ("R80C", "r80c_rtt_mg_eee_01_fq")):
        d = os.path.join(root, name); os.makedirs(os.path.join(d, "compact", refarm), exist_ok=True)
        n = FIX_N["eee_01"]
        pd.DataFrame({"scan_id": np.arange(1, n + 1), "px": np.cumsum(np.full(n, 1e-3)), "py": 0.0, "pz": 0.0, "yaw_rad": 0.01}).to_csv(os.path.join(d, "compact", refarm, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        pd.DataFrame([dict(arm=refarm, excursion_mm=82.003503, settled_offset_mm=65.332038)]).to_csv(os.path.join(d, "phase_summary.csv"), index=False)
        pd.DataFrame([dict(arm=refarm, state_trace_sha256_run_id_blanked="a" * 64)]).to_csv(os.path.join(d, "completeness.csv"), index=False)
        out.append(d)
    return out


def fx_returns(rd_, rows):
    os.makedirs(rd_, exist_ok=True)
    n474 = [r["arm"] for r in rows if r["group"] == "N474"]
    pd.DataFrame([dict(arm=a, frame=f, pos_err_mm=0.1) for a in n474 for f in EARLY_FRAMES]).to_csv(os.path.join(rd_, "early_grid_summary.csv"), index=False)
    f2b = [r["arm"] for r in rows if r["group"] == "N485B"]
    pd.DataFrame([dict(arm=a, frame=f, voxel_point_count=10, voxel_distinct_frames=2) for a in f2b for f in F2B_FRAMES]).to_csv(os.path.join(rd_, "f485_2b_summary.csv"), index=False)
    pft = [r["arm"] for r in rows if r["fam"] in ("PCA", "DB")]
    pd.DataFrame([dict(arm=a, path="pca", frame=0, reason=0, reason_name="accept", rows=5) for a in pft]).to_csv(os.path.join(rd_, "plane_fit_trace_counts_all.csv"), index=False)
    sw = [r["arm"] for r in rows if r["group"] == "N479" and r["ov"].get("imu.process_noise.switch.scan") is not None]
    pd.DataFrame([dict(arm=a, verdict="x") for a in sw]).to_csv(os.path.join(rd_, "r81_switch_pairs.csv"), index=False)
    pd.DataFrame({"arm": [r["arm"] for r in rows if r["arm_kind"] != "refuse"], "gt_ate_mm": 1.0}).to_csv(os.path.join(rd_, "phase_summary.csv"), index=False)


def selftest():
    import copy
    fails = []
    rows = fixture_rows()
    t_root = tempfile.mkdtemp()
    saved_n = dict(N_SCANS)
    N_SCANS.update(FIX_N)
    try:
        mdir = os.path.join(t_root, "man"); fx_write_manifest(mdir, rows)
        table = load_manifest(mdir)
        T = {t["id"]: t for t in table}
        refs = fx_refs(t_root, rows)
        ret = os.path.join(t_root, "ret"); fx_returns(ret, rows)

        def build(breaks):
            """breaks: {arm: {switch: True}} -> a fresh runs dir (hard links to the good one for every untouched arm)"""
            r = tempfile.mkdtemp(); shutil.rmtree(r)
            shutil.copytree(good, r, copy_function=os.link)
            for arm, o in breaks.items():
                shutil.rmtree(os.path.join(r, arm), ignore_errors=True)
                fx_arm(r, T[arm], o, refs)
            return r

        def gates(runs, man=mdir, ret_dirs=(ret,), refs_=refs, expect=None, phase=None, **kw):
            return run_gates(runs, man, FIX_GRID_RETIME, FIX_GRID_BAG, refs_, [], phase if phase is not None else os.path.join(ret, "phase_summary.csv"), ret_dirs=ret_dirs, expect=expect or {}, **kw)
        good = os.path.join(t_root, "good"); os.makedirs(good)
        for t in table:
            fx_arm(good, t, {}, refs)
        g0, idr = gates(good)
        bad_rows = [(n[:110], s, dd[:160]) for n, s, dd in g0.rows if s in ("FAIL", "MISSING")]
        if bad_rows:
            fails.append("good data: %d FAIL/MISSING, first %s" % (len(bad_rows), bad_rows[:4]))
        nrows = len(g0.rows)
        for need in ("IDENT r81_dc_id_mg_eee_01_fq", "IDENT r81_rt_id_mg_eee_01_fq", "DET r81_own_det_x_eee_01", "DETPOS r81_own_det_x_eee_01", "RTGRID r81_rt_mg_eee_01_fq", "RTGRID r81_rt_b3_eee_03_mq", "MSGGRID r81_rt_b3_eee_03_mq",
                     "NISENG r81_dc_nis_eee_01_fq", "NGCFG r81_rt_ng_eee_01_fq", "VMINENG r81_dc_vmin", "VS2ENG r81_rt_vs2", "SMOKE r81_dc_sm_f2b_60s", "ZDCFG r81_dc_zt_eee_01_a025", "SCCFG r81_rt_sc_fq", "EARLY all 2 N474", "F2BSUM all 1 N485B",
                     "N479COMP eee_01", "PFT all 2", "SWCELL CFG sw_eee_01_fq_off_p3", "SWCOMP sw_eee_01_fq_off_p3 all 8", "EXIT r81_dc_refuse_plane_var_min_negative", "OVLENG r81_rt_mg_eee_01_fq", "OVLENG r81_dc_nis_eee_01_fq"):
            if not any(need in n for n, s, _ in g0.rows):
                fails.append("good data: no row named like %r" % need)
        if not any(n.startswith("MSGLOG") for n, s, _ in g0.rows if "r81_dc_nis" in n) is False and any("MSGLOG r81_dc_nis" in n or "MSGLOG r81_rt_nis" in n or "MSGLOG r81_rt_ng" in n for n, s, _ in g0.rows):
            fails.append("message-log rows must exist for BASE arms only")
        if len(idr) != 2 or any(r["ref_arm"] is None for r in idr):
            fails.append("identity csv rows: %s" % idr)
        # ---- breakages: each trips its own row on its own arm and no row of another arm
        ID1, ID2, MG, DET, B3 = "r81_dc_id_mg_eee_01_fq", "r81_rt_id_mg_eee_01_fq", "r81_rt_mg_eee_01_fq", "r81_own_det_x_eee_01", "r81_rt_b3_eee_03_mq"
        NISD, NISR, ZT, ZD, NGR, SC = "r81_dc_nis_eee_01_fq", "r81_rt_nis_eee_01_fq", "r81_dc_zt_eee_01_a025", "r81_rt_zd_eee_01_fq", "r81_rt_ng_eee_01_fq", "r81_rt_sc_fq"
        VM, VS, F2, PCA, EFD, SWX, SWC = "r81_dc_vmin_eee_01_vm400_pfn3_fq", "r81_rt_vs2_eee_01_s200_pfn3_fq", "r81_dc_f2b_pfn3_fq", "r81_dc_pca_fq", "r81_dc_ef_eee_01_n200_dsfirst_pfn3_fq", "r81_dc_swx_eee_01_fq2mq", "r81_dc_swc_eee_01_fq"
        SW1, SM_N, SM_V, SM_F, RF = "r81_rt_sw_eee_01_off_v0p4_pfn3_fq", "r81_dc_sm_nis_60s", "r81_dc_sm_vmin_60s", "r81_dc_sm_f2b_60s", "r81_dc_refuse_plane_var_min_negative"
        CASES = {   # name: (arm, switches, row prefix that must FAIL/MISSING, expected status or None)
            "failed": (MG, {"failed": 1}, "RUN " + MG, "FAIL"), "exit_nonzero": (MG, {"exit_nonzero": 1}, "EXIT " + MG, "FAIL"), "exit_bad": (MG, {"exit_bad": 1}, "EXIT " + MG, "FAIL"), "exit_absent": (MG, {"exit_absent": 1}, "EXIT " + MG, "MISSING"),
            "vox": (MG, {"vox": 0.35}, "VOX " + MG, "FAIL"), "pfn": (MG, {"pfn": 1}, "PFN " + MG, "FAIL"), "msgkey": (MG, {"msgkey": "false"}, "MSGKEY " + MG, "FAIL"),
            "policy_rt_prints_discard": (MG, {"policy": "discard"}, "OVLCFG " + MG, "FAIL"), "policy_dc_prints_rt": (NISD, {"policy": "retime_trim"}, "OVLCFG " + NISD, "FAIL"),
            "t0_wrong": (MG, {"t0": 1609059013.5}, "OVLCFG " + MG, "FAIL"), "ovl_noline": (MG, {"nopolicyline": 1}, "OVLENG " + MG, "FAIL"), "ovl_inert": (MG, {"inert": 1}, "OVLENG " + MG, "FAIL"), "ovl_dc_line": (NISD, {"dcpolicyline": 1}, "OVLENG " + NISD, "FAIL"),
            "len_short": (MG, {"len_short": 1}, "LEN " + MG, "FAIL"),
            "ident_pos": (ID1, {"identpos": 1}, "IDENT " + ID1, "FAIL"), "ident_rt_pos": (ID2, {"identpos": 1}, "IDENT " + ID2, "FAIL"), "ident_digest": (ID1, {"digest_id": 1}, "IDENT " + ID1, "FAIL"),
            "det_digest": (DET, {"det": 1}, "DET " + DET, "FAIL"), "det_pos": (DET, {"detpos": 1}, "DETPOS " + DET, "FAIL"),
            "msg_absent": (MG, {"msg_absent": 1}, "MSGLOG " + MG, "MISSING"), "msg_rule": (MG, {"msgrule": 1}, "MSGRULE " + MG, "FAIL"), "msg_col": (B3, {"msg_col": 1}, "MSGLOG " + B3, "FAIL"), "msg_order": (B3, {"msg_order": 1}, "MSGLOG " + B3, "FAIL"),
            "msg_grid": (B3, {"msg_grid": 1}, "MSGGRID " + B3, "FAIL"), "msg_late": (MG, {"msg_late": 1}, "MSGGRID " + MG, "FAIL"), "msg_short": (MG, {"msg_short": 1}, "MSGGRID " + MG, "FAIL"), "msg_dropped_rt": (ID2, {"msg_drop_rt": 1}, "MSGGRID " + ID2, "FAIL"),
            "nis_mode": (NISD, {"gmode": "nis_gate"}, "NISENG " + NISD, "FAIL"), "nis_noline": (NISR, {"nogain": 1}, "NISENG " + NISR, "FAIL"), "nis_clip_cfg": (NISD, {"echo": {"imu.process_noise.gain.nis_clip": 0.5}}, "NISCFG " + NISD, "FAIL"),
            "nis_slope_cfg": (NISR, {"echo": {"imu.process_noise.gain.nis_slope": 0.2}}, "NISCFG " + NISR, "FAIL"), "ng_mode": (NGR, {"gmode": "nis_smooth"}, "NGCFG " + NGR, "FAIL"),
            "zd_acc": (ZT, {"zacc": 0.15}, "ZDCFG " + ZT, "FAIL"), "zd_applied": (ZD, {"zapplied": 5}, "ZDCFG " + ZD, "FAIL"), "zd_nofile": (ZD, {"zd_nofile": 1}, "ZDFILE " + ZD, "MISSING"), "zd_file_applied": (ZT, {"zd_applied": 1}, "ZDFILE " + ZT, "FAIL"),
            "q_nofile": (NISD, {"q_nofile": 1}, "QADAPT " + NISD, "MISSING"), "sc_cap": (SC, {"caprot": 0.02}, "SCCFG " + SC, "FAIL"),
            "vmin_cfg": (VM, {"vmin": 5.0e-4}, "VMINCFG " + VM, "FAIL"), "vmin_noline": (VM, {"novline": 1}, "VMINENG " + VM, "FAIL"), "vmin_dumpline_off": (VM, {"vmin": 0.0}, "VMINCFG " + VM, "FAIL"),
            "vs2_cfg": (VS, {"s2": 3.0e-4}, "VS2CFG " + VS, "FAIL"), "vs2_noline": (VS, {"novline": 1}, "VS2ENG " + VS, "FAIL"),
            "smoke_vmin_inert": (SM_V, {"clamped": 0}, "VMINENG " + SM_V, "FAIL"), "smoke_nis_noline": (SM_N, {"nogain": 1}, "NISENG " + SM_N, "FAIL"),
            "smoke_f2b_nocol": (SM_F, {"dump_nocol": 1}, "SMOKE " + SM_F, "FAIL"), "smoke_f2b_zero": (SM_F, {"dump_zero": 1}, "SMOKE " + SM_F, "FAIL"), "smoke_f2b_nofile": (SM_F, {"dump_nocol": 1}, "SMOKE " + SM_F, "FAIL"),
            "extra_gain": (MG, {"extra_gain": 1}, "NOEXTRA " + MG, "FAIL"), "extra_vline": (MG, {"extra_vline": 1}, "NOEXTRA " + MG, "FAIL"),
            "sw_vox": (SW1, {"vox": 0.2}, "SWCFG " + SW1, "FAIL"), "sw_exit": (SW1, {"exit_nonzero": 1}, "EXIT " + SW1, "FAIL"), "sw_run": (SW1, {"failed": 1}, "RUN " + SW1, "FAIL"),
            "ref_zero": (RF, {"ref_zero": 1}, "EXIT " + RF, "FAIL"), "ref_done": (RF, {"ref_done": 1}, "EXIT " + RF, "FAIL"), "ref_needle": (RF, {"ref_needle": 1}, "REFUSE " + RF, "FAIL"), "ref_noexit": (RF, {"ref_noexit": 1}, "EXIT " + RF, "MISSING"),
            "line_dump_fallback_bad": (SW1, {"noline": 1, "vox": 0.3}, "SWCFG " + SW1, "FAIL")}
        # drop a case that duplicates another's trigger
        CASES.pop("smoke_f2b_nofile")
        for name, (arm, o, needle, want) in CASES.items():
            r = build({arm: o})
            gg, _ = gates(r)
            hit = [(n, s) for n, s, _ in gg.rows if n.startswith(needle) and s in ("FAIL", "MISSING")]
            if not hit:
                fails.append("breakage %s did not trip a row named %s" % (name, needle))
            elif want and not all(s == want for n, s in hit):
                fails.append("breakage %s: want %s got %s" % (name, want, hit[:2]))
            other = [(n[:90], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING") and arm not in n and not n.startswith("SWCELL")]
            if other:
                fails.append("breakage %s tripped rows that do not concern %s: %s" % (name, arm, other[:2]))
            shutil.rmtree(r, ignore_errors=True)
        # a false-FAIL regression set from the R80 return: formats that must NOT fail
        r = build({MG: {"echo": {"data_queues.log_lidar_msgs": "True"}, "msgkey": "True"}, SW1: {"noline": 1}})
        gg, _ = gates(r)
        ff = [(n[:100], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING")]
        if ff:
            fails.append("R80 false-FAIL regression (True/1 booleans, cut one-line dump): %s" % ff[:3])
        shutil.rmtree(r, ignore_errors=True)
        # ---- manifest-level breakages: job's retime keys differ from the grid; a missing job; counts against the SPEC
        m2 = os.path.join(t_root, "man2"); shutil.copytree(mdir, m2)
        jp = os.path.join(m2, "jobs_r81_eee_01.json"); J = json.load(open(jp))
        for j in J:
            if j["job_id"] == MG:
                j["overrides"]["cbk.lidar.retime_t0_s"] = 1609059013.5
        json.dump(J, open(jp, "w"))
        gg, _ = gates(good, man=m2)
        hit = [(n, s) for n, s, _ in gg.rows if n.startswith("RTGRID " + MG) and s == "FAIL"]
        if not hit:
            fails.append("RTGRID: a job whose t0 differs from the grid must FAIL")
        cn = {}
        for t in table:
            cn[t["group"]] = cn.get(t["group"], 0) + 1
        gg, _ = gates(good, expect={k: (v,) for k, v in cn.items() if k != "OWN"})
        if any(n.startswith("FAMCOUNT") and s == "FAIL" for n, s, _ in gg.rows):
            fails.append("FAMCOUNT must PASS when the counts equal the SPEC values given")
        gg, _ = gates(good, expect={"N474": (720,)})
        if not any(n.startswith("FAMCOUNT N474") and s == "FAIL" for n, s, _ in gg.rows):
            fails.append("FAMCOUNT must FAIL when the manifest holds 2 N474 arms and the SPEC says 720")
        # ---- early grid / N485B / N479 / PFT completeness
        for name, fn, mut, needle in (("early", "early_grid_summary.csv", lambda df: df[~((df.arm == "r81_rt_ef_eee_01_n200_dsfirst_pfn3_fq") & (df.frame == 20))], "EARLY r81_rt_ef_eee_01_n200_dsfirst_pfn3_fq"),
                                      ("f2b", "f485_2b_summary.csv", lambda df: df[df.frame != 405], "F2BSUM " + F2), ("pft", "plane_fit_trace_counts_all.csv", lambda df: df[df.arm != "r81_dc_pca_fq"], "PFT " + PCA),
                                      ("switch_pairs", "r81_switch_pairs.csv", lambda df: df[df.arm != SWX], "N479COMP r81_switch_pairs.csv")):
            r2 = os.path.join(t_root, "ret_" + name); shutil.copytree(ret, r2)
            df = pd.read_csv(os.path.join(r2, fn)); mut(df).to_csv(os.path.join(r2, fn), index=False)
            gg, _ = gates(good, ret_dirs=(r2,))
            if not [1 for n, s, _ in gg.rows if n.startswith(needle) and s in ("FAIL", "MISSING")]:
                fails.append("completeness breakage %s did not trip a row named %s" % (name, needle))
            os.remove(os.path.join(r2, fn))
            gg, _ = gates(good, ret_dirs=(r2,))
            if not [1 for n, s, _ in gg.rows if n.split(" ")[0] in needle.split(" ")[0] and s == "MISSING"]:
                fails.append("completeness file absent (%s) must be MISSING" % name)
        # N479 control missing from the runs dir
        r = build({}); shutil.rmtree(os.path.join(r, SWC)); gg, _ = gates(r)
        if not [1 for n, s, _ in gg.rows if n.startswith("N479COMP eee_01") and s in ("FAIL", "MISSING")]:
            fails.append("N479 control arm absent must trip N479COMP")
        shutil.rmtree(r, ignore_errors=True)
        # sweep cell with 7 of 8 phase_summary rows
        ph = pd.read_csv(os.path.join(ret, "phase_summary.csv")); ph = ph[ph.arm != SW1]
        gg, _ = gates(good, phase=ph)
        if [(n[-40:]) for n, s, _ in gg.rows if s in ("FAIL", "MISSING")] != [("SWCOMP %s has a phase_summary row" % SW1)[-40:]]:
            fails.append("sweep cell with 7 of 8 phase_summary rows: %s" % [(n[:90], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING")][:3])
        # ---- an undispatched queue is NOTRUN and touches nothing; an empty directory is all MISSING (+ INFO)
        r = build({}); gone = [t["id"] for t in table if t["batch"] == "r81_eee_03"]
        for a in gone:
            shutil.rmtree(os.path.join(r, a))
        gg, _ = gates(r, dispatched=set(T) - set(gone))
        if [(n[:80], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING")] or not any(n.startswith("NOTRUN queue r81_eee_03") for n, s, _ in gg.rows):
            fails.append("undispatched queue: expected NOTRUN INFO only: %s" % [(n[:80], s) for n, s, _ in gg.rows if s in ("FAIL", "MISSING")][:3])
        gg, _ = gates(r)
        if not [1 for n, s, _ in gg.rows if s == "MISSING" and any(a in n for a in gone)] or [1 for n, s, _ in gg.rows if s == "FAIL"]:
            fails.append("dispatched but absent arms must be MISSING (and nothing FAIL)")
        shutil.rmtree(r, ignore_errors=True)
        e = tempfile.mkdtemp(); ge, _ = gates(e, ret_dirs=(), phase=os.path.join(e, "none.csv"))
        if any(s not in ("MISSING", "INFO") for n, s, _ in ge.rows):
            fails.append("empty dir: a row other than MISSING / INFO: %s" % [(n[:80], s) for n, s, _ in ge.rows if s not in ("MISSING", "INFO")][:3])
        # ---- the HOLD file from a DET / IDENT FAIL
        r = build({DET: {"det": 1}}); gg, _ = gates(r); hf = os.path.join(t_root, "HOLD_FROM_GATES.txt")
        wrote = write_hold(gg, hf)
        if not wrote or "DET %s" % DET not in open(hf).read():
            fails.append("a DET FAIL must write the hold file naming the row")
        r = build({MG: {"exit_nonzero": 1}}); gg, _ = gates(r); hf2 = os.path.join(t_root, "HOLD2.txt")
        if write_hold(gg, hf2) or os.path.exists(hf2):
            fails.append("a non-DET / non-IDENT FAIL must not write a hold file")
        shutil.rmtree(r, ignore_errors=True)
        # ---- exit formats
        for txt, want in (("exit_code=0 signal=none\n", (0, "none")), ("exit_code=-6 signal=SIGABRT\n", (-6, "SIGABRT")), ("7\n", (7, "none")), ("garbage\n", "bad")):
            dd = tempfile.mkdtemp(); open(os.path.join(dd, "exit_status.txt"), "w").write(txt)
            if exit_status(dd) != want:
                fails.append("exit_status parse of %r gave %r" % (txt, exit_status(dd)))
            shutil.rmtree(dd, ignore_errors=True)
    finally:
        N_SCANS.clear(); N_SCANS.update(saved_n)
        shutil.rmtree(t_root, ignore_errors=True)
    print("SELFTEST r81_gates", "FAIL " + "; ".join(fails) if fails else
          "PASS (fixture of %d arms covering every family; %d gate rows on good data, none FAIL or MISSING; %d single breakages each trip their own row on their own arm and no other arm's; R80 false-FAIL formats (True/1, cut one-line dump, exit_code=N signal=NAME) do not fail; "
          "RTGRID against a wrong job; FAMCOUNT; EARLY / F2BSUM / PFT / N479COMP incomplete and absent; sweep cell 7 of 8; undispatched queue NOTRUN; empty dir MISSING; DET FAIL writes the hold file, other FAILs do not)" % (len(rows), nrows, len(CASES)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--manifest", default="r81_manifest"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
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
