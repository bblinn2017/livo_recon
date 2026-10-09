#!/usr/bin/env python3
"""R85 gates (planning agent's script, derived from r84_gates.py; selftest on SYNTHETIC fixtures only, planner sandbox, nothing here was run on real data, not evidence).

    python3 tools/r69/r85_gates.py --runs <runs_dir> --manifest r85_manifest [--grid-retime r80c_grid.json] [--grid-bag r85_manifest/grid_bag.json] \
        [--ref-return R84_RETURN] [--ref-runs <old archive dir>] [--phase-summary phase_summary.csv] [--ret-dir <return dir>] [--expect "L0=288,L2R=198,SHT=144"] \
        [--dispatched <arm ids file | arm_table.tsv | comma list of arm ids / groups / families / queues>] [--out r85_gates.txt] [--csv r85_identity_vs_old.csv] [--hold-file HOLD_FROM_GATES.txt]
    python3 tools/r69/r85_gates.py --selftest

The tool states facts only: every row is PASS, FAIL, MISSING (never a pass), INFO or NOTRUN. It interprets nothing. The expectation of every arm is read from the MANIFEST ONLY (arm_table.tsv + jobs_*.json
+ required_outputs.json): every effective config key found in the params echo is compared with the job's override (a key the job does not set is compared with its compiled default), and every family row below takes
its expected value from the job of the arm, never from a per-arm constant in this file. The only constants are the SPEC group counts of EXPECT (override with --expect; OWN is a fact), the five REMOVED keys,
CELLS and SHORT_COLS (copied from tools/r69/make_r85_manifest.py; the SHT static row cross-checks the jobs against CELLS).

Groups: L0 (288 whole-bag arms, voxel edge 0.20 or 0.25 m, voxel_map.map.max_layer 0), L2R (198 layer-2 reference arms of the L0 cells no existing arm covers), SHT (144 arms of 60 s, tier EF, cells A..H =
point_filter_num {3,1} x imu.ds.mode {first,off} x voxel_map.search.neighborhood_size {1,2}), OWN (26: 3 identity, 9 whole-bag determinism repeats, 9 short determinism repeats, 3 smoke, 2 refuse arms).
L0 and L2R are SWEEP groups: summaries only; their PASS rows fold into CELL rows and they need no pos_trace unless gt_max_mm > 500 in the phase summary (then TRACE).
Gate-name scheme (one scheme for all families): <PREFIX> <arm> <what is compared>, where PREFIX is
  L0ENG / L2RENG  sweep arms, manifest part: job voxel_size equals the edge in the arm name and the arm_table columns edge / layer / fit / lines; job max_layer equals layer (0 for L0, 2 for L2R); job plane_fit_mode and
                  line_residual/enabled equal the name and the columns; job point_filter_num is 3, imu.ds.mode is first, the key voxel_map.search.neighborhood_size is absent (nb column 1 = compiled default);
                  L0: l2_ref non-empty, l2_source in {R81, R82A, R83, R85} and the l2_ref name carries the matching round prefix, an R85 source needs the twin L2R arm (control column = the L0 arm) whose job differs in
                  max_layer only; L2R: control is an L0 arm of the manifest. Echo part (arm present): params echo shows max_layer, voxel_size, plane_fit_mode (pca when the job has none) and line_residual/enabled
                  (false when none) equal to the job; deb+lines-on: classify_at_fit and debiased_line_equiv as the job states them (echo or engagement)
  L2MAP           once: l2_reference_map.tsv of the manifest (first column l0_arm) agrees with the l2_ref / l2_source columns of the L0 arms
  SHTENG          every arm with a cell letter (SHT, det_sht, the two smoke arms with a cell): manifest part: job point_filter_num, imu.ds.mode, neighborhood_size equal the CELLS row of the letter and the arm_table
                  nb column; ds_leaf_size null (key removed) exactly when ds is off; SHT arms: name (bag, edge, letter), voxel_size, max_layer 0, duration 60, control = cell A of the same bag and edge, `requires`
                  names state_trace.csv and frame_stats.txt. Echo part: the worst status of the shared themes PFN, DSCFG, NBCFG, VOX, D4CFG of that arm (not re-judged here)
  DETCFG          det_fb / det_sht arm: the job overrides equal those of the ref arm (output paths and run id aside)
  SHTSCAN         every arm with a cell letter (155): compact/<arm>/short_scan_all.csv.gz (looked for in <runs>/<arm>/ then <ret-dir>/compact/<arm>/) exists (MISSING if not); header equals SHORT_COLS exactly; at least 400
                  rows; scan strictly increasing; t_s within [0, 62]; fs_n_residuals, residual_count and fs_n_voxels finite on every row (the detail counts the null rows). A FAIL or MISSING on one of the two SMOKE arms
                  with a cell holds exactly like DET / DETPOS / IDENT; on every other arm it is a fact, not a hold.
  DETS            det_sht arm: the state_trace digest equals the ref arm's AND short_scan_all.csv.gz equals the ref arm's table row for row in all columns (the detail counts the differing rows). Either difference FAILs and holds.
  DET / DETPOS    det_fb arm: digest and full-precision pos trace equal the ref arm's (unchanged).  Digest source: <runs>/<arm>/state_trace_blank_runid.sha256, else the completeness.csv column
                  state_trace_sha256_run_id_blanked found in --ret-dir / the runs directory or its parent.
  IDENT           the three identity arms against their R84 `ref` arms through r85_identity_hold.py and --ref-return (the R84 return): pos trace 0.005 mm, excursion, settled offset, digest; yaw is INFO there (R84 rule
                  carried); no --ref-return, or one that lacks the ref arm, is MISSING, never PASS
  REMOVED         refuse arms (the two R85 refuse arms, needles from the arm_table `needles` column): exit status non-zero and the run.log / stderr text contains every needle; the refusal text should say `removed`
                  (case-insensitive; else INFO). Every other arm: none of the FIVE removed keys (lio/point_cov/mode, window_s, sample_window_s, lio/time_cov/mode, sigma_s) in its job, params echo or engagement text
  ZUPTGONE        no lio/zupt key in any job, params echo or engagement text
  L0SMOKE         the three smoke arms (r85_rt_sm_sht_A_60s, r85_rt_sm_sht_H_60s, r85_dc_sm_l0_pca_off_020_60s): accepted (exit 0 and a params echo present), and each echoed key of the job equals it: max_layer, voxel_size,
                  neighborhood_size, point_filter_num, imu.ds.mode, plane_fit_mode, line_residual/enabled. A FAIL names the arm and the key. Never a hold (the dispatch consequence is the planner's).
  CFG themes      OVLCFG, MSGKEY, PNCFG, TRCCFG, D4CFG, DSCFG, PFN, VOX, NBCFG: effective config of every key against the job's override. NBCFG: voxel_map.search.neighborhood_size, printed by the tree's VoxelMap
                  params dump (src/map/voxelmap.cpp, block [params/voxel_map]) as `  search/neighborhood_size:    <n>` on every run (read in the source, not seen in a log); an arm without the key is judged against
                  the compiled default 1. DSCFG: imu.ds.mode against the echo; a ds-off arm (job value null for imu.ds.ds_leaf_size = key removed) has no leaf in `want`; if the echo prints imu/ds/ds_leaf_size it must carry the
                  `[FORCED: scope 'imu/ds/mode != off' is off]` marker of the tree's ConfigResolver (a live leaf under ds off is a FAIL; not printed is INFO).
  RTGRID / OVLCFG rt arms: the job's retime keys equal the grid of the bag and the policy is retime_trim; dc arms: no retime key, no overlap_policy key
  NOEXTRA, PFT, MSGLOG / MSGRULE / MSGGRID / MSGSCANS, LEN, EXIT, RUN, REFUSE, SWEEPCOMP, FAMCOUNT (unchanged)
  COMPLETE        per dispatched non-refuse arm: arm.json requires_missing (written by the score job from the arm_table `requires` column; SHT arms also require state_trace.csv and frame_stats.txt, which the host deletes
                  later) is empty (PASS), non-empty (FAIL, naming the files), or arm.json lacks the field (MISSING)
  REQUIRED        one row per entry of required_outputs.json: every dispatched arm of the entry has its file (<runs>/<arm>/ or <ret-dir>/compact/<arm>/) with the listed header columns; `generated` entries are looked for in
                  --ret-dir / the runs directory and its parent (columns and min_rows checked); r85_gates.txt and r85_identity_vs_old.csv are produced by this tool (INFO)
Removed in R85 against R84 (code deleted, not disabled): the sample-window gates and theme, the time-based-process-noise gates, the comparison group and its gates.
A FAIL (or, for the SHTSCAN smoke rows, a MISSING) of DET, DETPOS, DETS, IDENT or SHTSCAN-of-a-smoke-arm writes --hold-file (HOLD_FROM_GATES.txt): the agent places it as HOLD.txt (r85_hold_check.sh). Every other
FAIL (incl. L0SMOKE and the other SHTSCAN rows) is reported only. The gate never deletes or cancels anything.
"""
import argparse, csv, glob, gzip, io, json, os, re, shutil, sys, tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r85_identity_hold as IH  # noqa: E402

LEN_MIN = 0.97
REF_OFFSET = -0.1
N_SCANS = {"eee_01": 3987, "eee_02": 3210, "eee_03": 1814, "nya_01": 3950, "nya_02": 4287, "nya_03": 4094, "sbs_01": 3542, "sbs_02": 3732, "sbs_03": 3893}   # message / scan counts of the bags (make_r79_manifest.N_SCANS)
EXPECT = {"L0": (288,), "L2R": (198,), "SHT": (144,)}   # SPEC table (make_r85_manifest.EXPECT); OWN (26) is a fact, not checked
SWEEP = ("L0", "L2R")
L2_SOURCES = ("R81", "R82A", "R83", "R85")
L2_PREFIX = {"R81": "r81_", "R82A": "r82_", "R83": "r83_", "R85": "r85_"}
EDGE_TAGS = ("0p2", "0p25")                                    # make_r85_manifest.EDGES 0.20, 0.25
CELLS = (("A", 3, "first", 1), ("B", 1, "first", 1), ("C", 3, "off", 1), ("D", 1, "off", 1), ("E", 3, "first", 2), ("F", 1, "first", 2), ("G", 3, "off", 2), ("H", 1, "off", 2))   # make_r85_manifest.CELLS
SHORT_COLS = ["scan", "t_s", "residual_count", "fs_n_residuals", "fs_n_voxels", "fs_n_voxels_is_plane", "fs_iters", "fs_dx_pos_mm", "fs_nis", "fs_reduced_chi2", "pos_err_mm"]   # make_r85_manifest.SHORT_COLS
SHORT_MIN_ROWS, SHORT_T_MAX = 400, 62.0
OWN_GATE_FILES = ("r85_gates.txt", "r85_identity_vs_old.csv")
MSG_COLS = ["msg_index", "seq", "header_stamp", "scan_time", "gap_header_s", "n_raw", "n_kept", "t_first", "t_last", "off_first_s", "off_last_s", "span_s", "prev_accepted_last_t", "overlap_s", "verdict"]
MSG_COLS_C = MSG_COLS + ["retime_shift_s", "n_trimmed"]
_E = "|".join(EDGE_TAGS)
SWEEP_RE = re.compile(r"^r85_(?P<ovl>rt|dc)_(?P<kind>l0|l2)_(?P<bag>[a-z]+_\d\d)_(?P<fit>pca|deb)_(?P<lines>off|on)_v(?P<edge>%s)_(?P<q>fq|mq)$" % _E)
SHT_RE = re.compile(r"^r85_rt_sht_(?P<bag>[a-z]+_\d\d)_v(?P<edge>%s)_(?P<cell>[A-H])$" % _E)
RETIME_KEYS = ("cbk.lidar.retime_t0_s", "cbk.lidar.retime_period_s", "cbk.lidar.retime_seq0", "cbk.lidar.retime_ref_offset_s")
K_SW = "lio.point_cov.sample_window_s"                              # REMOVED in R85 (F-520); only the refuse arm carries it
K_MAXL, K_VOX, K_FIT, K_LINE = "voxel_map.map.max_layer", "voxel_map.map.voxel_size", "voxel_map.plane.plane_fit_mode", "voxel_map.plane.line_residual.enabled"
K_CLS, K_EQV, K_TRACE = "voxel_map.plane.line_residual.classify_at_fit", "voxel_map.plane.debiased_line_equiv", "voxel_map.plane.plane_fit_trace_frames"
K_NB, K_PFN, K_DSM, K_DSL = "voxel_map.search.neighborhood_size", "cbk.lidar.point_filter_num", "imu.ds.mode", "imu.ds.ds_leaf_size"
REMOVED_KEYS = ("lio.point_cov.mode", "lio.point_cov.window_s", "lio.time_cov.mode", "lio.time_cov.sigma_s", K_SW)      # refused at startup (R84 F-521 four keys, R85 F-520 sample_window_s)
REMOVED_RE = re.compile(r"lio[/.](point_cov[/.](mode|window_s|sample_window_s)\b|time_cov\b)")
ZUPT_RE = re.compile(r"lio[/.]zupt")
IGN_PAIR = ("outputs.path", "outputs.debug_log_dir", "eval.state_trace_run_id")
STRICT = re.compile(r"^(voxel_map\.map\.(voxel_size|max_layer)|voxel_map\.search\.neighborhood_size|cbk\.lidar\.(point_filter_num|overlap_policy|retime_.*)|imu\.process_noise\.model|imu\.process_noise\.motion\.(gyro|acc)_scale|imu\.ds\.mode|"
                    r"voxel_map\.plane\.(plane_gap_floor_en|debiased_line_equiv|plane_fit_mode|plane_fit_reasons_en|plane_fit_trace_frames)|"
                    r"voxel_map\.plane\.line_residual\.(enabled|form|rule|rows|variance|classify_at_fit)|data_queues\.log_lidar_msgs)$")
THEMES = [("OVLCFG", re.compile(r"^cbk\.lidar\.(overlap_policy|retime_.*)$")), ("MSGKEY", re.compile(r"^data_queues\.log_lidar_msgs$")),
          ("PNCFG", re.compile(r"^imu\.process_noise\.(model|motion\..*)$")),
          ("TRCCFG", re.compile(r"^voxel_map\.plane\.plane_fit_(reasons_en|trace_frames)$")),
          ("D4CFG", re.compile(r"^voxel_map\.plane\.(line_residual\.(classify_at_fit|enabled|rows|variance)|debiased_line_equiv|plane_gap_floor_en|plane_fit_mode)$")),
          ("DSCFG", re.compile(r"^imu\.ds\.mode$")), ("PFN", re.compile(r"^cbk\.lidar\.point_filter_num$")), ("VOX", re.compile(r"^voxel_map\.map\.(voxel_size|max_layer)$")),
          ("NBCFG", re.compile(r"^voxel_map\.search\.neighborhood_size$"))]
DEFAULTS = {"voxel_map.plane.line_residual.classify_at_fit": False, "voxel_map.plane.plane_gap_floor_en": False, "voxel_map.plane.debiased_line_equiv": "off", "cbk.lidar.overlap_policy": "discard",
            "voxel_map.plane.line_residual.enabled": False, "voxel_map.map.max_layer": 2, "voxel_map.plane.plane_fit_mode": "pca", K_NB: 1,
            K_DSM: "first"}   # a key the job does not set is at its compiled default; a build that prints it must print the default (neighborhood_size: paramWarn default 1; imu/ds/mode: default first)
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
        self.smoke_cell = set()          # smoke arms with a cell letter: a FAIL or MISSING SHTSCAN row of one of them holds like DET

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
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_log(text):
    """-> {leaf: [(keypath, value, priority, tail)]}; priority 0 = `key = v` (the [config/effective] echo), 1 = `key: v` of the application's [params] dump, 2 = ROS dump `/ns/key: v`."""
    idx = {}
    for ln in (text or "").splitlines():
        ln = ANSI_RE.sub("", ln)
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
                          fit=g_("fit"), lines=g_("lines"), edge=_f(g_("edge")), layer=_f(g_("layer")), cell=g_("cell"), nb=_f(g_("nb")), requires=g_("requires").split(), l2_ref=g_("l2_ref"), l2_source=g_("l2_source"), control=g_("control"),
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


def has_cell(t):
    """the 155 arms that carry a cell letter: 144 SHT, 9 det_sht, the 2 smoke arms with a cell."""
    return bool(t["cell"]) and t["kind"] in ("sht", "det_sht", "smoke")


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


def sweep_static(t, by_id, ctl):
    """L0ENG / L2RENG rows that need the manifest only: the job against the arm name and the arm_table columns; the l2 reference and the twin pair."""
    arm, grp, ov = t["id"], t["group"], t["ov"]
    P = grp + "ENG"
    if ov is None:
        return [("%s %s manifest consistency" % (P, arm), "MISSING", "no job for the arm in jobs_*.json")]
    out, bad = [], []
    m = SWEEP_RE.match(arm)
    maxl, vox = ov.get(K_MAXL, DEFAULTS[K_MAXL]), ov.get(K_VOX)
    fit_j, on_j = ov.get(K_FIT, DEFAULTS[K_FIT]), bool(ov.get(K_LINE, DEFAULTS[K_LINE]))
    pfn_j, ds_j = ov.get(K_PFN), ov.get(K_DSM, DEFAULTS[K_DSM])
    if not m:
        bad.append("arm name is not of the R85 sweep form")
    else:
        if m.group("kind") != {"L0": "l0", "L2R": "l2"}[grp]:
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
    if ov.get(K_MAXL) != {"L0": 0, "L2R": 2}[grp]:
        bad.append("job max_layer %r but group %s needs %d" % (ov.get(K_MAXL), grp, {"L0": 0, "L2R": 2}[grp]))
    if t["fit"] and fit_j != {"deb": "debiased", "pca": "pca"}.get(t["fit"], t["fit"]):
        bad.append("arm_table fit %s but the job plane_fit_mode is %r" % (t["fit"], fit_j))
    if t["lines"] and on_j != (t["lines"] == "on"):
        bad.append("arm_table lines %s but the job line_residual/enabled is %s" % (t["lines"], on_j))
    if pfn_j != 3:
        bad.append("job point_filter_num %r (sweep arms use 3)" % (pfn_j,))
    if ds_j != "first":
        bad.append("job imu.ds.mode %r (sweep arms use first)" % (ds_j,))
    if K_NB in ov:
        bad.append("job sets %s = %r (sweep arms leave it absent: compiled default 1)" % (K_NB, ov[K_NB]))
    if t["nb"] is not None and t["nb"] != 1:
        bad.append("arm_table nb %g but the job leaves neighborhood_size at its default 1" % t["nb"])
    out.append(("%s %s manifest: the job's voxel_size, max_layer, plane_fit_mode, line_residual/enabled, point_filter_num 3, imu.ds.mode first and the absent neighborhood_size key agree with the arm name and the arm_table columns edge, layer, fit, lines, nb" % (P, arm),
                "FAIL" if bad else "PASS", "; ".join(bad) if bad else "edge %s, max_layer %s, fit %s, lines %s, pfn %s, ds %s" % (vox, maxl, fit_j, on_j, pfn_j, ds_j)))
    if grp == "L0":
        b2 = []
        if not t["l2_ref"]:
            b2.append("l2_ref empty")
        if t["l2_source"] not in L2_SOURCES:
            b2.append("l2_source %r not in %s" % (t["l2_source"], "/".join(L2_SOURCES)))
        elif t["l2_ref"] and not t["l2_ref"].startswith(L2_PREFIX[t["l2_source"]]):
            b2.append("l2_ref %s does not carry the prefix %s of l2_source %s" % (t["l2_ref"], L2_PREFIX[t["l2_source"]], t["l2_source"]))
        if t["control"] != t["l2_ref"]:
            b2.append("control %r differs from l2_ref %r" % (t["control"], t["l2_ref"]))
        out.append(("L0ENG %s l2_ref is non-empty, carries the prefix of its l2_source and l2_source is one of %s" % (arm, ", ".join(L2_SOURCES)), "FAIL" if b2 else "PASS", "; ".join(b2) or "%s (%s)" % (t["l2_ref"], t["l2_source"])))
        if t["l2_source"] == "R85":
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
            out.append(("L0ENG %s twin L2R arm exists (its control column = this arm) and the pair's job overrides differ only in max_layer" % arm, "FAIL" if b3 else "PASS", "; ".join(b3) or t["l2_ref"]))
    else:
        c = by_id.get(t["control"])
        b4 = []
        if c is None or c["group"] != "L0":
            b4.append("control %r is not an L0 arm of the manifest" % t["control"])
        elif c["l2_source"] != "R85" or c["l2_ref"] != arm:
            b4.append("its L0 control has l2_source %s and names %s as l2_ref" % (c["l2_source"], c["l2_ref"]))
        out.append(("L2RENG %s control is the L0 arm of the same cell" % arm, "FAIL" if b4 else "PASS", "; ".join(b4) or t["control"]))
    return out


def sweep_echo(t, log, et):
    """L0ENG / L2RENG rows from the params echo of one present arm; the expectation is the job's value (pca / false / the code default 2 when the job has none)."""
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


def l2map_rows(table, mdir):
    """L2MAP: l2_reference_map.tsv of the manifest (first column l0_arm) against the l2_ref / l2_source columns of the L0 arms. Manifest only."""
    nm = "L2MAP l2_reference_map.tsv (first column l0_arm) agrees with the l2_ref and l2_source columns of the L0 arms"
    p = os.path.join(mdir, "l2_reference_map.tsv")
    if not os.path.exists(p):
        return [(nm, "MISSING", "l2_reference_map.tsv absent from the manifest")]
    try:
        rows = list(csv.DictReader(open(p, newline=""), delimiter="\t"))
    except Exception as e:
        return [(nm, "FAIL", "unreadable: %s" % e)]
    if not rows or "l0_arm" not in rows[0]:
        return [(nm, "FAIL", "no rows or no l0_arm column")]
    mp = {r["l0_arm"]: r for r in rows}
    l0 = [t for t in table if t["group"] == "L0"]
    bad = ["%s: map %s/%s, arm_table %s/%s" % (t["id"], mp[t["id"]].get("l2_ref"), mp[t["id"]].get("l2_source"), t["l2_ref"], t["l2_source"]) for t in l0 if t["id"] in mp and (mp[t["id"]].get("l2_ref") != t["l2_ref"] or mp[t["id"]].get("l2_source") != t["l2_source"])]
    bad += ["%s: not in the map" % t["id"] for t in l0 if t["id"] not in mp]
    bad += ["%s: in the map but not an L0 arm of the manifest" % a for a in mp if a not in {t["id"] for t in l0}]
    return [(nm, "FAIL" if bad else "PASS", "; ".join(bad[:5]) + (" ... %d more" % (len(bad) - 5) if len(bad) > 5 else "") if bad else "%d L0 arms" % len(l0))]


def sht_static(t, by_id):
    """SHTENG manifest part, for every arm with a cell letter."""
    arm, ov = t["id"], t["ov"]
    nm = "SHTENG %s manifest: the job's point_filter_num, imu.ds.mode, neighborhood_size and ds_leaf_size agree with cell %s of the CELLS table and the arm_table nb column%s" % (arm, t["cell"], "; name, edge, voxel_size, max_layer 0, 60 s, control = cell A, requires state_trace.csv and frame_stats.txt" if t["kind"] == "sht" else "")
    if ov is None:
        return [(nm, "MISSING", "no job for the arm in jobs_*.json")]
    cell = next((c for c in CELLS if c[0] == t["cell"]), None)
    if cell is None:
        return [(nm, "FAIL", "cell %r is not one of A..H" % t["cell"])]
    _, pfn, ds, nb = cell
    bad = []
    if ov.get(K_PFN) != pfn:
        bad.append("job point_filter_num %r, cell %s says %d" % (ov.get(K_PFN), cell[0], pfn))
    if ov.get(K_DSM) != ds:
        bad.append("job imu.ds.mode %r, cell %s says %s" % (ov.get(K_DSM), cell[0], ds))
    if ov.get(K_NB) != nb:
        bad.append("job neighborhood_size %r, cell %s says %d" % (ov.get(K_NB), cell[0], nb))
    if (K_DSL in ov and ov[K_DSL] is None) != (ds == "off"):
        bad.append("ds_leaf_size is %s in the job but ds mode is %s (off needs the null override that removes the key)" % ("null" if ov.get(K_DSL, 0) is None and K_DSL in ov else repr(ov.get(K_DSL, "absent")), ds))
    if ds != "off" and not (isinstance(ov.get(K_DSL), (int, float)) and ov.get(K_DSL) > 0):
        bad.append("ds mode %s but the job's ds_leaf_size is %r" % (ds, ov.get(K_DSL, "absent")))
    if t["nb"] is not None and t["nb"] != nb:
        bad.append("arm_table nb %g, cell %s says %d" % (t["nb"], cell[0], nb))
    if t["kind"] == "sht":
        m = SHT_RE.match(arm)
        if not m:
            bad.append("arm name is not of the R85 short form")
        else:
            if m.group("cell") != t["cell"]:
                bad.append("name letter %s but cell column %s" % (m.group("cell"), t["cell"]))
            if m.group("bag") != t["bag"]:
                bad.append("name bag %s but bag column %s" % (m.group("bag"), t["bag"]))
            if ov.get(K_VOX) is None or abs(float(ov[K_VOX]) - float(m.group("edge").replace("p", "."))) > 1e-9:
                bad.append("job voxel_size %r but the name says %s" % (ov.get(K_VOX), m.group("edge").replace("p", ".")))
            if t["edge"] is None or ov.get(K_VOX) is None or abs(float(ov[K_VOX]) - t["edge"]) > 1e-9:
                bad.append("job voxel_size %r but arm_table edge %s" % (ov.get(K_VOX), t["edge"]))
            ctl_id = "r85_rt_sht_%s_v%s_A" % (m.group("bag"), m.group("edge"))
            if t["control"] != ctl_id:
                bad.append("control %r, expected cell A of the same bag and edge: %s" % (t["control"], ctl_id))
            elif by_id.get(ctl_id, {}).get("cell") != "A":
                bad.append("control %s is not a cell A arm of the manifest" % ctl_id)
        if ov.get(K_MAXL) != 0:
            bad.append("job max_layer %r (short arms use 0)" % (ov.get(K_MAXL),))
        if t["dur"] != 60:
            bad.append("duration %r (short arms use 60 s)" % (t["dur"],))
        miss = [f for f in ("state_trace.csv", "frame_stats.txt") if f not in t["requires"]]
        if miss:
            bad.append("requires lacks %s" % ", ".join(miss))
    return [(nm, "FAIL" if bad else "PASS", "; ".join(bad) if bad else "pfn %d, ds %s, nb %d" % (pfn, ds, nb))]


def sht_echo(t, cg):
    """SHTENG echo part: the shared themes PFN, DSCFG, NBCFG, VOX, D4CFG of the arm, rolled up (not re-judged)."""
    th = {"PFN": "point_filter_num", "DSCFG": "imu.ds.mode", "NBCFG": "neighborhood_size", "VOX": "max_layer / voxel_size", "D4CFG": "plane_fit_mode / line_residual"}
    sts, det = [], []
    for k, what in th.items():
        rows = [(s, d) for n, s, d in cg if n.startswith(k + " ")]
        if not rows:
            sts.append("MISSING"); det.append("%s: no row" % k)
        else:
            w = "FAIL" if any(s == "FAIL" for s, _ in rows) else ("MISSING" if any(s == "MISSING" for s, _ in rows) else ("INFO" if all(s == "INFO" for s, _ in rows) else "PASS"))
            sts.append(w); det.append("%s %s" % (k, w))
    worst_ = "FAIL" if "FAIL" in sts else ("MISSING" if "MISSING" in sts else ("INFO" if "INFO" in sts else "PASS"))
    return [("SHTENG %s params echo of point_filter_num, imu.ds.mode, neighborhood_size, max_layer / voxel_size and plane_fit_mode against the job (worst status of the shared themes PFN, DSCFG, NBCFG, VOX, D4CFG)" % t["id"], worst_, "; ".join(det))]


def ds_off_rows(t, ov, log):
    """DSCFG extra row of a ds-off arm: the leaf key was removed by a null override (it is not in the effective-config comparison). The tree's ConfigResolver prints a dead-scope key with the marker
    `[FORCED: scope 'imu/ds/mode != off' is off]`; a printed leaf without that marker is a live leaf under ds off."""
    if (ov or {}).get(K_DSM) != "off":
        return []
    arm, have = t["id"], bool(log.strip())
    nm = "DSCFG %s ds off: the job has no imu.ds.ds_leaf_size (null override) and the echo shows no live leaf" % arm
    if ov.get(K_DSL) is not None:
        return [(nm, "FAIL", "the job sets ds_leaf_size %r under ds off (the node refuses a dead-scope key)" % (ov.get(K_DSL),))]
    if not have:
        return [(nm, "MISSING", "no run.log text")]
    c = lookup(parse_log(log), K_DSL)
    if not c:
        return [(nm, "INFO", "the echo does not print imu/ds/ds_leaf_size (a fact)")]
    live = [v for v, tail in c if "FORCED" not in tail]
    return [(nm, "FAIL" if live else "PASS", ("printed without the FORCED marker: %s" % "/".join(live)) if live else "printed as dead scope: %s" % c[0][0])]


def removed_absent(t, ov, et, have):
    """REMOVED + ZUPTGONE for every arm that is not a refuse arm."""
    arm = t["id"]
    jk = [k for k in (ov or {}) if k in REMOVED_KEYS]
    hits = sorted({m.group(0) for m in REMOVED_RE.finditer(et)})
    out = [("REMOVED %s none of the five removed keys (lio/point_cov/mode, window_s, sample_window_s, lio/time_cov/mode, sigma_s) in the job, the params echo or the engagement text" % arm, "FAIL" if (jk or hits) else ("PASS" if have else "MISSING"),
            ("job keys: %s; " % ", ".join(jk) if jk else "") + ("text: %s" % ", ".join(hits) if hits else ""))]
    zj = [k for k in (ov or {}) if ZUPT_RE.search(k)]
    zl = [l.strip() for l in et.splitlines() if ZUPT_RE.search(l)]
    out.append(("ZUPTGONE %s no lio/zupt key in the job, the params echo or the engagement text" % arm, "FAIL" if (zj or zl) else ("PASS" if have else "MISSING"), ("job keys: %s; " % ", ".join(zj) if zj else "") + "; ".join(zl[:2])[:200]))
    return out


def refuse_removed(t, d, ex, ov):
    """REMOVED rows of a refuse arm: exit non-zero and every needle in the refusal text; the refusal text should say `removed`."""
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


def smoke_keys(ov):
    """the keys L0SMOKE compares with the params echo, all taken from the job (a key the job lacks is at its compiled default)."""
    return [(K_MAXL, ov.get(K_MAXL, DEFAULTS[K_MAXL])), (K_VOX, ov.get(K_VOX)), (K_NB, ov.get(K_NB, DEFAULTS[K_NB])), (K_PFN, ov.get(K_PFN)), (K_DSM, ov.get(K_DSM, DEFAULTS[K_DSM])),
            (K_FIT, ov.get(K_FIT, DEFAULTS[K_FIT])), (K_LINE, bool(ov.get(K_LINE, DEFAULTS[K_LINE])))]


def l0smoke_row(t, ov, ex, log):
    """L0SMOKE of one smoke arm: accepted (exit 0, params echo present) and every smoke key of the job equal in the echo. The detail names the arm and the failing key. Never a hold."""
    arm, have = t["id"], bool(log.strip())
    parts = []
    if ex is None:
        parts.append(("MISSING", "arm %s: exit_status.txt absent" % arm))
    elif ex == "bad":
        parts.append(("FAIL", "arm %s: exit_status.txt unreadable" % arm))
    else:
        parts.append(("PASS" if ex[0] == 0 else "FAIL", "arm %s: exit_code %s" % (arm, ex[0])))
    idx = parse_log(log)
    echo = have and (bool(re.search(r"\[params", log)) or bool(lookup(idx, K_VOX)))
    parts.append(("PASS" if echo else ("MISSING" if not have else "FAIL"), "arm %s: params echo %s" % (arm, "present" if echo else "absent")))
    for k, w in smoke_keys(ov):
        if w is None:
            parts.append(("FAIL", "arm %s key %s: the job has no value" % (arm, k))); continue
        st, det = echo_or_alt(idx, log, k, w, have)
        parts.append((st, "arm %s key %s: %s" % (arm, k, det)))
    bad = [d for s, d in parts if s in ("FAIL", "MISSING")]
    return ("L0SMOKE %s accepted (exit 0, params echo present) and the echoed max_layer, voxel_size, neighborhood_size, point_filter_num, imu.ds.mode, plane_fit_mode, line_residual/enabled equal the job's" % arm,
            worst([s for s, _ in parts]), "; ".join(bad) if bad else "; ".join(d for _, d in parts)[:300])


def l0smoke_notrun(table, live):
    out = []
    sm = [t for t in table if t["kind"] == "smoke"]
    for t in sm:
        if t["id"] not in live:
            out.append(("L0SMOKE %s not dispatched" % t["id"], "NOTRUN", ""))
    return out


def engagement_gates(t, ov, et, have):
    arm = t["id"]; ov = ov or {}; out = []
    smoke = t["kind"] == "smoke"
    vl = next((l for l in et.splitlines() if re.search(r"plane_var_min\s*=.*\brows\s*=", l)), None)
    dl = _eng_line(et, r"plane_var_diag_en rows=")
    cl = _eng_line(et, r"line_residual/classify_at_fit\s*=.*line_classified=")
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
    out.append(("NOEXTRA %s nothing engaged that the arm did not ask for (plane variance, plane_var_diag, D4 classify)" % arm, "FAIL" if extra else ("PASS" if have else "MISSING"), ", ".join(extra)))
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
    """Sweep arms (L0, L2R): FAIL / MISSING rows stay one per arm; PASS and INFO rows of a (gate, cell) fold into one CELL row (the report stays inside its size ledger)."""

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


# ------------------------------------------------------------------------------------------------ short-tier helpers
def arm_dirs(arm, runs, ret_dirs):
    """where a per-arm file may live: the arm folder of the runs directory, then compact/<arm> and <arm> of each return directory."""
    return [os.path.join(runs, arm)] + [os.path.join(r, "compact", arm) for r in ret_dirs] + [os.path.join(r, arm) for r in ret_dirs]


def locate(arm, fname, runs, ret_dirs):
    for dd in arm_dirs(arm, runs, ret_dirs):
        p = find_any(dd, [fname])
        if p:
            return p
    return None


def read_short(arm, runs, ret_dirs):
    """-> (state, header list, DataFrame, path): state 'ok', 'absent' or 'error:<msg>'."""
    p = locate(arm, "short_scan_all.csv.gz", runs, ret_dirs)
    if p is None:
        return "absent", None, None, None
    try:
        txt = gzip.open(p, "rt", errors="replace").read() if p.endswith(".gz") else open(p, errors="replace").read()
        hdr = txt.split("\n", 1)[0].strip().split(",")
        return "ok", hdr, pd.read_csv(io.StringIO(txt)), p
    except Exception as e:
        return "error:%s" % str(e)[:80], None, None, p


def shtscan_rows(t, runs, ret_dirs):
    """SHTSCAN of one arm with a cell letter."""
    arm = t["id"]
    nm = "SHTSCAN %s short_scan_all.csv.gz exists, header = the %d SHORT_COLS, >= %d rows, scan strictly increasing, t_s within [0, %g], fs_n_residuals, residual_count and fs_n_voxels finite on every row" % (arm, len(SHORT_COLS), SHORT_MIN_ROWS, SHORT_T_MAX)
    state, hdr, df, p = read_short(arm, runs, ret_dirs)
    if state == "absent":
        return [(nm, "MISSING", "compact/%s/short_scan_all.csv.gz not found (looked in the arm folder and the return directories)" % arm)]
    if state != "ok":
        return [(nm, "FAIL", "unreadable: %s" % state)]
    if hdr != SHORT_COLS:
        return [(nm, "FAIL", "header %s, wanted %s" % (",".join(hdr), ",".join(SHORT_COLS)))]
    bad, n = [], len(df)
    if n < SHORT_MIN_ROWS:
        bad.append("rows %d < %d" % (n, SHORT_MIN_ROWS))
    sc = pd.to_numeric(df["scan"], errors="coerce").to_numpy(float)
    if not np.isfinite(sc).all() or (n > 1 and not (np.diff(sc) > 0).all()):
        bad.append("scan is not finite and strictly increasing")
    ts = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(float)
    if not np.isfinite(ts).all() or ts.min() < 0 or ts.max() > SHORT_T_MAX:
        bad.append("t_s outside [0, %g] or not finite (min %s, max %s)" % (SHORT_T_MAX, "nan" if not len(ts) else "%.3f" % np.nanmin(ts), "nan" if not len(ts) else "%.3f" % np.nanmax(ts)))
    nul = {c: int((~np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(float))).sum()) for c in ("fs_n_residuals", "residual_count", "fs_n_voxels")}
    for c, k in nul.items():
        if k:
            bad.append("%s null on %d rows" % (c, k))
    return [(nm, "FAIL" if bad else "PASS", "; ".join(bad) if bad else "rows %d, t_s %.2f..%.2f, null rows: %s" % (n, np.nanmin(ts), np.nanmax(ts), ", ".join("%s %d" % kv_ for kv_ in nul.items())))]


def digest_of(arm, runs, ret_dirs):
    """state_trace digest (run_id blanked): the arm folder's state_trace_blank_runid.sha256, else the completeness.csv column state_trace_sha256_run_id_blanked of a return directory / the runs directory / its parent."""
    v = (rd(os.path.join(runs, arm), "state_trace_blank_runid.sha256") or "").strip()
    if v:
        return v.split()[0]
    for dd in list(ret_dirs) + [runs, os.path.join(runs, "..")]:
        p = os.path.join(dd, "completeness.csv")
        if os.path.exists(p):
            try:
                C = pd.read_csv(p)
            except Exception:
                continue
            c = "state_trace_sha256_run_id_blanked"
            if "arm" in C.columns and c in C.columns and (C["arm"] == arm).any():
                x = str(C[C["arm"] == arm].iloc[0][c]).strip()
                if x and x != "nan":
                    return x
    return None


def dets_gates(g, t, runs, by_id, ret_dirs):
    """DETS of one det_sht arm: digest and short_scan_all.csv.gz equal the ref arm's."""
    arm, ref = t["id"], t["ref"]
    if ref is None:
        g.add("DETS %s names no original in the manifest (ref column)" % arm, "FAIL"); return
    if ref not in by_id:
        g.add("DETS %s digest and short_scan table == %s" % (arm, ref), "INFO", "reference arm is not in this manifest"); return
    a, b = digest_of(arm, runs, ret_dirs), digest_of(ref, runs, ret_dirs)
    miss = [n for n, x in ((arm, a), (ref, b)) if x is None]
    g.add("DETS %s state_trace digest == %s" % (arm, ref), "MISSING" if miss else ("PASS" if a == b else "FAIL"), ("digest absent for: " + ", ".join(miss)) if miss else "%s vs %s" % (a[:16], b[:16]))
    nm = "DETS %s short_scan_all.csv.gz equals that of %s row for row in all columns" % (arm, ref)
    sa, ha, A, _ = read_short(arm, runs, ret_dirs)
    sb, hb, B, _ = read_short(ref, runs, ret_dirs)
    if sa != "ok" or sb != "ok":
        g.add(nm, "MISSING" if "absent" in (sa, sb) else "FAIL", "table %s for %s, %s for %s" % (sa, arm, sb, ref)); return
    if ha != hb:
        g.add(nm, "FAIL", "headers differ: %s vs %s" % (",".join(ha), ",".join(hb))); return
    m = min(len(A), len(B))
    X, Y = A.iloc[:m].reset_index(drop=True), B.iloc[:m].reset_index(drop=True)
    same = pd.DataFrame({c: ((X[c] == Y[c]) | (X[c].isna() & Y[c].isna())) for c in A.columns})
    ndiff = int((~same.all(axis=1)).sum()) + abs(len(A) - len(B))
    g.add(nm, "PASS" if ndiff == 0 else "FAIL", "rows %d vs %d, differing rows %d" % (len(A), len(B), ndiff))


def detcfg_row(t, by_id):
    """DETCFG: a determinism repeat has the job of its original (output paths and run id aside). Manifest only."""
    arm, ref = t["id"], t["ref"]
    nm = "DETCFG %s job overrides equal those of %s (output paths and run id aside)" % (arm, ref)
    r = by_id.get(ref)
    if t["ov"] is None or r is None or r["ov"] is None:
        return [(nm, "MISSING", "job of the arm or of its original not in the manifest")]
    ab = object()
    diffs = sorted(k for k in (set(t["ov"]) | set(r["ov"])) - set(IGN_PAIR) if t["ov"].get(k, ab) != r["ov"].get(k, ab))
    return [(nm, "FAIL" if diffs else "PASS", ("differ in %s" % ", ".join(diffs)) if diffs else "")]


def complete_row(t, d):
    """COMPLETE: arm.json requires_missing, written by the score job from the arm_table `requires` column."""
    nm = "COMPLETE %s every file of the arm_table requires column (%d files%s) was present when the score job checked (arm.json requires_missing)" % (
        t["id"], len(t["requires"]), ", incl. state_trace.csv and frame_stats.txt" if "state_trace.csv" in t["requires"] and "frame_stats.txt" in t["requires"] else "")
    meta = jsn(d, "arm.json")
    if meta is None or "requires_missing" not in meta:
        return [(nm, "MISSING", "arm.json absent or without a requires_missing field")]
    rm = meta["requires_missing"]
    rm = rm.split() if isinstance(rm, str) else list(rm or [])
    return [(nm, "FAIL" if rm else "PASS", ("absent: " + " ".join(rm)) if rm else "")]


def hdr_of(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", errors="replace") as f:
        return f.readline().strip().split(",")


def required_gates(g, table, live, mdir, runs, ret_dirs):
    """REQUIRED: one row per entry of the manifest's required_outputs.json (presence, header columns; min_rows for generated files)."""
    p = os.path.join(mdir, "required_outputs.json")
    if not os.path.exists(p):
        g.add("REQUIRED required_outputs.json in the manifest directory", "MISSING", "file absent"); return
    RO = json.load(open(p))
    full = len(live) == len(table)
    gdirs = list(ret_dirs) + [runs, os.path.join(runs, "..")]
    for e in RO:
        f, cols, arms = e["file"], e.get("columns") or [], [a for a in e.get("arms", []) if a in live]
        tag = "REQUIRED %s (%s, %s)" % (f, e.get("followup", ""), e.get("where", ""))
        if f in OWN_GATE_FILES:
            g.add("%s produced by this tool" % tag, "INFO", "written from the rows of this run"); continue
        if e.get("source", "arm") == "generated" and not e.get("arms"):
            q = next((os.path.join(dd, f) for dd in gdirs if os.path.exists(os.path.join(dd, f))), None)
            if q is None:
                g.add("%s generated file found in the return directory / runs directory" % tag, "MISSING", "not found"); continue
            bad = []
            try:
                if f.endswith((".csv", ".csv.gz", ".tsv")):
                    T = pd.read_csv(q, sep="\t" if f.endswith(".tsv") else ",")
                    nrow, have = len(T), list(T.columns)
                else:
                    nrow, have = sum(1 for l in open(q, errors="replace") if l.strip()), None
                miss = [c for c in cols if have is not None and c not in have]
                if miss:
                    bad.append("columns missing: %s" % ", ".join(miss))
                if full and nrow < e.get("min_rows", 0):
                    bad.append("rows %d < %d" % (nrow, e["min_rows"]))
                g.add("%s generated file has its columns%s" % (tag, " and at least %d rows" % e["min_rows"] if full else " (row count not checked: partial dispatch)"), "FAIL" if bad else "PASS", "; ".join(bad) or "rows %d" % nrow)
            except Exception as ex_:
                g.add("%s generated file readable" % tag, "FAIL", str(ex_)[:100])
            continue
        if not arms:
            g.add("%s no dispatched arm of this entry" % tag, "NOTRUN"); continue
        miss, badc = [], []
        for a in arms:
            q = locate(a, f, runs, ret_dirs)
            if q is None:
                miss.append(a)
            elif cols and e.get("where") == "return":
                try:
                    h = hdr_of(q)
                    if any(c not in h for c in cols):
                        badc.append("%s: %s" % (a, ",".join(c for c in cols if c not in h)))
                except Exception:
                    badc.append("%s: unreadable" % a)
        det = ("absent for %d: %s" % (len(miss), ", ".join(miss[:4]) + (" ..." if len(miss) > 4 else ""))) if miss else ""
        det += (("; " if det else "") + "columns missing in %d: %s" % (len(badc), "; ".join(badc[:3]))) if badc else ""
        g.add("%s %d dispatched arm(s) have the file%s" % (tag, len(arms), " with its columns" if cols and e.get("where") == "return" else ""), "FAIL" if badc and not miss else ("MISSING" if miss else "PASS"), det or "all present")


# ------------------------------------------------------------------------------------------------ the gate
def run_gates(runs, manifest, grid_retime=None, grid_bag=None, ref_returns=(), ref_runs=(), phase=None, dispatched=None, ret_dirs=(), expect=EXPECT, ref_manifests=(), tol_mm=IH.TOL_MM):
    g = G()
    table = load_manifest(manifest)
    A = lambda a: os.path.join(runs, a)
    present = lambda a: os.path.isdir(A(a)) and os.path.exists(os.path.join(A(a), "arm.json"))
    disp = dispatched_set(dispatched, table) if (dispatched is None or isinstance(dispatched, str)) else set(dispatched)
    live = {t["id"] for t in table if disp is None or t["id"] in disp}
    by_id = {t["id"]: t for t in table}
    g.smoke_cell = {t["id"] for t in table if t["kind"] == "smoke" and t["cell"]}
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
    if any(t["group"] == "L0" for t in table):
        for n_, s_, dt_ in l2map_rows(table, manifest):
            g.add(n_, s_, dt_)
    Pn = phase if isinstance(phase, pd.DataFrame) else (pd.read_csv(phase) if phase and os.path.exists(phase) else None)
    gtmax = {}
    if Pn is not None and "arm" in Pn.columns and "gt_max_mm" in Pn.columns:
        gtmax = {a: num(v) for a, v in zip(Pn["arm"], Pn["gt_max_mm"])}
    id_rows, sw = [], SwAgg(g)
    P_cache = {}
    popen = lambda a: P_cache.setdefault(a, csvf(A(a), "pos_trace_all.csv.gz"))
    swcells = {}
    smoke_rows = []
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
        if has_cell(t):
            put(sht_static(t, by_id))
        if t["kind"] in ("det_fb", "det_sht"):
            put(detcfg_row(t, by_id))
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
        if has_cell(t):
            put(shtscan_rows(t, runs, ret_dirs))
        if not pres:
            if t["kind"] == "smoke":
                r_ = ("L0SMOKE %s accepted (exit 0, params echo present) and the echoed keys equal the job's" % arm, "MISSING", "arm %s: arm.json absent" % arm)
                put([r_]); smoke_rows.append(r_)
            continue
        put(complete_row(t, d))
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
            put([("SWEEPCFG %s effective config: edge, layer, fit, lines, point filter, downsampling, neighborhood size, overlap policy, nothing extra" % arm, worst(sts) if sts else "MISSING", "; ".join(bad))])
        else:
            put(cg)
        put(ds_off_rows(t, ov, log))
        put(grid_gate(t, grid_retime, grid_bag))
        if is_sw:
            put(sweep_echo(t, log, et))
        if has_cell(t):
            put(sht_echo(t, cg))
        put(removed_absent(t, ov, et, have))
        put(engagement_gates(t, ov, et, have))
        if t["kind"] == "smoke":
            r_ = l0smoke_row(t, ov, ex, log)
            put([r_]); smoke_rows.append(r_)
        if is_sw:
            gm = gtmax.get(arm)
            if gm is not None and gm > 500:
                Pt = popen(arm)
                put([("TRACE %s collapsed (gt_max_mm %.0f > 500): pos_trace_all.csv.gz present with rows" % (arm, gm), "PASS" if Pt is not None and len(Pt) else "MISSING", "")])
            continue                    # L0 / L2R: summaries only; no pos_trace is required unless gt_max_mm > 500
        P = popen(arm)
        len_gate(g, t, P)
        if t["kind"] == "id":
            rows, _ = ident_rows(t, runs, ref_returns, ref_runs, manifest, ref_manifests, tol_mm)
            for n, s, dt in rows:
                g.add("IDENT %s %s" % (arm, n), s, dt)
            id_rows.append(ident_csv_row(t, runs, ref_returns, tol_mm))
        if t["kind"] == "det_fb":
            det_gates(g, t, runs, by_id, popen, ret_dirs)
        if t["kind"] == "det_sht":
            dets_gates(g, t, runs, by_id, ret_dirs)
        if t["msglog"]:
            msglog_gates(g, t, d, P, grid_retime, grid_bag)
    sw.flush()
    for n_, s_, dt_ in l0smoke_notrun(table, live):
        g.add(n_, s_, dt_)
    if smoke_rows and all(s_ == "PASS" for _, s_, _ in smoke_rows) and len(smoke_rows) == sum(1 for t in table if t["kind"] == "smoke"):
        g.add("L0SMOKE all %d smoke arms accepted (exit 0, params echo present) with their echoed keys equal to the jobs'" % len(smoke_rows), "PASS")
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
    required_gates(g, table, live, manifest, runs, ret_dirs)
    return g, id_rows


def det_gates(g, t, runs, by_id, popen, ret_dirs=()):
    arm, ref = t["id"], t["ref"]
    if ref is None:
        g.add("DET %s names no original in the manifest (ref column)" % arm, "FAIL"); return
    if ref not in by_id:
        g.add("DET %s digest == %s" % (arm, ref), "INFO", "reference arm is not in this manifest"); return
    a, b = digest_of(arm, runs, ret_dirs), digest_of(ref, runs, ret_dirs)
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
        return [("per-scan position, excursion, settled offset and digest equal %s" % t["ref"], "MISSING", "no --ref-return given: the R84 reference return is required for the identity rows")], None
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
    """DET / DETPOS / DETS / IDENT FAIL, or FAIL / MISSING of SHTSCAN on a smoke arm with a cell letter."""
    def holds(n, s):
        if s == "FAIL" and n.startswith(("DET ", "DETPOS ", "DETS ", "IDENT ")):
            return True
        return s in ("FAIL", "MISSING") and any(n.startswith("SHTSCAN %s " % a) for a in g.smoke_cell)
    bad = [(n, s, d) for n, s, d in g.rows if holds(n, s)]
    if not bad:
        return False
    import time
    body = ["HOLD (from r85_gates.py, %s): a determinism, identity or smoke-arm short-scan gate FAILED (DET, DETPOS, DETS, IDENT, or SHTSCAN of a smoke arm). Stop dispatching unstarted queues and report; let running jobs finish; cancel nothing; delete nothing; change nothing in the tree." % time.strftime("%Y-%m-%d %H:%M:%S"),
            "Investigate (config dump diff, run.log, first-divergence scan, tree diff) and report to the planner. The planner's written word lifts the hold (rename HOLD.txt to HOLD.released_<date>.txt; never delete it).", ""]
    body += ["%s %s   [%s]" % (s, n, d) for n, s, d in bad]
    with open(hold_file, "a" if os.path.exists(hold_file) else "w") as f:
        f.write("\n".join(body) + "\n")
    return True


# ------------------------------------------------------------------------------------------------ selftest fixtures
def selftest():
    """Synthetic: a manifest written here (arm_table.tsv + jobs_x.json + required_outputs.json; no generator is imported) and fabricated arm folders. Planner sandbox, not evidence about the node."""
    root = tempfile.mkdtemp(prefix="r85_gates_selftest_")
    fails, nck = [], [0]

    def ck(c, m):
        nck[0] += 1
        if not c:
            fails.append(m)
    cols = ["arm", "tier", "group", "fam", "mode", "kind", "arm_kind", "ref", "bag", "seq", "batch", "pfn", "duration_s", "msglog", "needles", "flag", "fit", "lines", "edge", "layer", "cell", "nb", "l2_ref", "l2_source", "control",
            "same_batch_control", "requires"]
    ARMS = []
    BASE = {K_PFN: 3, K_DSM: "first", K_DSL: 0.1}
    GRID = {"eee_01": {"t0_s": 100.0, "period_s": 0.1, "seq0": 5, "n": 3987}}
    RT = {"cbk.lidar.overlap_policy": "retime_trim", "cbk.lidar.retime_t0_s": 100.0, "cbk.lidar.retime_period_s": 0.1, "cbk.lidar.retime_seq0": 5, "cbk.lidar.retime_ref_offset_s": -0.1}
    DEBON = {K_FIT: "debiased", K_LINE: True, "voxel_map.plane.line_residual.variance": "plane_equivalent", "voxel_map.plane.line_residual.rows": "distance", "voxel_map.plane.plane_gap_floor_en": True, K_CLS: True, K_EQV: "mean_cov"}
    REQ = "engagement_imu.txt lidar_msg_log.csv line_rows_per_scan.csv"
    REQ_S = REQ + " plane_fit_reasons.csv state_trace.csv frame_stats.txt"

    def arm(name, group, ov, mode="dc", kind="fb", ref="", bag="eee_01", dur="full", msglog=0, needles="", fam="", base=None, **extra):
        b = dict(BASE if base is None else base, **(RT if mode == "rt" else {}))
        ARMS.append((dict(dict(arm=name, tier="FB", group=group, fam=fam or group, mode=mode, kind=kind, arm_kind=kind, ref=ref, bag=bag, seq=bag, batch="q_" + bag, pfn=3, duration_s=dur, msglog=msglog, needles=needles, flag="", requires=REQ), **extra),
                     dict(b, **ov)))
    sn = lambda kind, bag, fit, lines, tag, q="fq", ovl="dc": "r85_%s_%s_%s_%s_%s_v%s_%s" % (ovl, kind, bag, fit, lines, tag, q)
    TAGV = {"0p2": 0.2, "0p25": 0.25}

    def pair(bag, fit, lines, tag, q="fq", ext0=None, ext2=None, row0=None, row2=None, twin=True):
        """an L0 arm with its R85 twin L2R arm."""
        edge = TAGV[tag]
        ov = dict(DEBON) if (fit == "deb" and lines == "on") else ({K_LINE: True} if lines == "on" else {})
        l0, l2 = sn("l0", bag, fit, lines, tag, q), sn("l2", bag, fit, lines, tag, q)
        r0 = dict(fit=fit, lines=lines, edge=edge, layer=0, nb=1, l2_ref=l2, l2_source="R85", control=l2)
        r0.update(row0 or {})
        arm(l0, "L0", {**ov, K_VOX: edge, K_MAXL: 0, **(ext0 or {})}, bag=bag, **r0)
        if twin:
            r2 = dict(fit=fit, lines=lines, edge=edge, layer=2, nb=1, l2_ref=l0, l2_source="R85", control=l0)
            r2.update(row2 or {})
            arm(l2, "L2R", {**ov, K_VOX: edge, K_MAXL: 2, **(ext2 or {})}, bag=bag, **r2)
        return l0, l2
    # ---- sweep arms (dc; the rt grid is exercised on the SHT arms)
    L0a, L2a = pair("eee_01", "pca", "off", "0p2")
    L0b, L2b = pair("eee_01", "deb", "on", "0p25", ext0={"voxel_map.plane.plane_fit_reasons_en": True, K_TRACE: 10}, ext2={"voxel_map.plane.plane_fit_reasons_en": True, K_TRACE: 10})
    L0c, L2c = pair("eee_01", "pca", "off", "0p25", q="mq")                                # echo of the L0 arm shows max_layer 2
    L0d, _ = pair("eee_02", "pca", "off", "0p2", ext0={K_VOX: 0.25}, twin=False, row0=dict(l2_ref="r83_x", l2_source="R83", control="r83_x"))      # job edge 0.25, name v0p2, column 0.2
    L0e, _ = pair("eee_02", "pca", "on", "0p25", twin=False, row0=dict(l2_ref="r99_x", l2_source="R99", control="r99_x"))   # l2_source outside the set
    L0f, _ = pair("eee_02", "pca", "off", "0p2", q="mq", twin=False, row0=dict(l2_ref="", l2_source="R83", control=""))     # l2_ref empty
    L0g, _ = pair("eee_02", "pca", "off", "0p25", q="mq", twin=False)                                                   # R85 source whose twin does not exist
    L0h, L2h = pair("eee_03", "pca", "on", "0p2", q="mq", ext2={K_PFN: 4})                                              # the pair differs in one more key
    L0i, L2i = pair("eee_03", "deb", "on", "0p2")                                                                       # echo prints neither classify_at_fit nor debiased_line_equiv
    L0j, L2j = pair("eee_03", "deb", "on", "0p25")                                                                      # classify / equiv only in the engagement line
    L0k, L2k = pair("eee_03", "deb", "on", "0p25", q="mq")                                                              # equiv printed off
    L2m = sn("l2", "nya_01", "pca", "on", "0p2")                                                                         # L2R arm with a control that is no L0 arm; echo max_layer 0
    arm(L2m, "L2R", {K_VOX: 0.2, K_LINE: True, K_MAXL: 2}, bag="nya_01", fit="pca", lines="on", edge=0.2, layer=2, nb=1, l2_ref="nothing", l2_source="R85", control="nothing")
    L0n, _ = pair("nya_01", "pca", "off", "0p2", q="mq", twin=False, ext0={K_PFN: 1}, row0=dict(l2_ref="r83_n", l2_source="R83", control="r83_n"))      # point filter 1
    L0o, _ = pair("nya_01", "pca", "off", "0p25", twin=False, ext0={K_NB: 2}, row0=dict(l2_ref="r83_o", l2_source="R83", control="r83_o"))           # neighborhood_size key present
    L0p, _ = pair("nya_01", "pca", "off", "0p25", q="mq", twin=False, ext0={K_DSM: "off", K_DSL: None}, row0=dict(l2_ref="r83_p", l2_source="R83", control="r83_p"))   # ds off
    L0q, _ = pair("nya_02", "pca", "off", "0p2", twin=False, row0=dict(l2_ref="r83_x", l2_source="R81", control="r83_x"))                         # prefix of the reference does not match its source
    L0r, _ = pair("nya_02", "pca", "off", "0p25", twin=False, row0=dict(nb=2, l2_ref="r83_r", l2_source="R83", control="r83_r"))                      # arm_table nb 2, job without the key
    L2s = sn("l2", "nya_02", "pca", "off", "0p2", "mq")                                                                   # L2R arm whose control is an L0 arm that does not name it
    arm(L2s, "L2R", {K_VOX: 0.2, K_MAXL: 2}, bag="nya_02", fit="pca", lines="off", edge=0.2, layer=2, nb=1, l2_ref=L0q, l2_source="R85", control=L0q)
    # ---- SHT arms (rt, eee_01)
    CELLD = {c[0]: c for c in CELLS}

    def sht(letter, tag, **ovx):
        _, pfn, ds, nb = CELLD[letter]
        edge = TAGV[tag]
        ov = dict(DEBON, **{K_VOX: edge, K_MAXL: 0, K_PFN: pfn, K_DSM: ds, K_NB: nb, K_DSL: (None if ds == "off" else 0.1)})
        ov.update(ovx)
        name = "r85_rt_sht_eee_01_v%s_%s" % (tag, letter)
        arm(name, "SHT", ov, mode="rt", kind="sht", dur="60", fam="SHT", base={}, fit="deb", lines="on", edge=edge, layer=0, cell=letter, nb=nb, control="r85_rt_sht_eee_01_v%s_A" % tag, requires=REQ_S)
        return name
    S_A2, S_H2 = sht("A", "0p2"), sht("H", "0p2")
    S_B2 = sht("B", "0p2", **{K_PFN: 3})                       # cell B says point filter 1, the job says 3
    S_C2 = sht("C", "0p2", **{K_DSL: 0.1})                     # ds off but the job keeps a leaf
    S_E2 = sht("E", "0p2")                                     # echo prints neighborhood_size 1, the job says 2
    S_A25 = sht("A", "0p25")
    S_D2 = sht("D", "0p2"); ARMS[-1][0]["nb"] = 2                  # arm_table nb column 2, cell D and the job say 1
    # ---- OWN arms
    arm("r85_dc_id_eee_01_fq", "OWN", {}, kind="id", ref="r84_dc_id_eee_01_fq", fam="ID", msglog=0)
    arm("r85_rt_id_eee_01_fq", "OWN", {}, mode="rt", kind="id", ref="r84_rt_id_eee_01_fq", fam="ID")
    arm("r85_rt_id_eee_01_mq", "OWN", {}, mode="rt", kind="id", ref="r84_rt_id_eee_01_mq", fam="ID")
    SMA, SMH, SM0 = "r85_rt_sm_sht_A_60s", "r85_rt_sm_sht_H_60s", "r85_dc_sm_l0_pca_off_020_60s"
    for nm_, letter, edge in ((SMA, "A", 0.2), (SMH, "H", 0.25)):
        _, pfn, ds, nb = CELLD[letter]
        arm(nm_, "OWN", dict(DEBON, **{K_VOX: edge, K_MAXL: 0, K_PFN: pfn, K_DSM: ds, K_NB: nb, K_DSL: (None if ds == "off" else 0.1)}), mode="rt", kind="smoke", dur="60", fam="SMK", base={}, cell=letter, requires=REQ_S)
    arm(SM0, "OWN", {K_VOX: 0.2, K_MAXL: 0, K_FIT: "pca"}, kind="smoke", dur="60", fam="SMK")
    arm("rf_sw", "OWN", {K_SW: 0.005}, kind="refuse", needles="lio/point_cov/sample_window_s", dur="5", fam="REF")
    arm("rf_mode", "OWN", {"lio.point_cov.mode": "propagated"}, kind="refuse", needles="lio/point_cov/mode", dur="5", fam="REF")
    arm("rf_nomatch", "OWN", {"lio.point_cov.mode": "propagated"}, kind="refuse", needles="lio/point_cov/mode", dur="5", fam="REF")
    arm("rf_zero", "OWN", {K_SW: 0.005}, kind="refuse", needles="lio/point_cov/sample_window_s", dur="5", fam="REF")
    DET_X, DET_O = "r85_own_det_x_eee_01", "r85_rt_l0_eee_01_deb_on_v0p25_fq"
    arm(DET_O, "OWN", {}, mode="rt"); arm(DET_X, "OWN", {}, mode="rt", kind="det_fb", ref=DET_O, fam="DET")
    DET_S = "r85_own_det_s_eee_01"
    ARMS.append((dict(dict(ARMS[[r["arm"] for r, _ in ARMS].index(S_A25)][0]), arm=DET_S, group="OWN", kind="det_sht", arm_kind="det_sht", ref=S_A25, fam="DETS", control="", edge="", layer="", fit="", lines="", nb=""),
                 dict(ARMS[[r["arm"] for r, _ in ARMS].index(S_A25)][1])))
    for x in ("x_clean", "x_rm_job", "x_rm_echo", "x_rm_eng", "x_zupt", "x_pv"):
        arm(x, "OWN", {K_SW: 0.0} if x == "x_rm_job" else {})
    arm("x_dcret", "OWN", {"cbk.lidar.retime_t0_s": 1.0})
    arm("msg_dc", "OWN", {}, msglog=1)
    mdir = os.path.join(root, "m"); os.makedirs(mdir)
    with open(os.path.join(mdir, "arm_table.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows([r for r, _ in ARMS])
    json.dump([dict(job_id=r["arm"], overrides=o) for r, o in ARMS], open(os.path.join(mdir, "jobs_x.json"), "w"))
    with open(os.path.join(mdir, "l2_reference_map.tsv"), "w") as f:
        f.write("l0_arm\tl2_ref\tl2_source\tl2_new\n")
        for r, _ in ARMS:
            if r["group"] == "L0":
                f.write("%s\t%s\t%s\t0\n" % (r["arm"], r["l2_ref"], r["l2_source"]))
    runs = os.path.join(root, "runs"); os.makedirs(runs)

    def echo(mode, **kw):
        d = {"cbk__lidar__point_filter_num": 3, "imu__ds__mode": "first", "imu__ds__ds_leaf_size": 0.1, "search__neighborhood_size": 1, "map__max_layer": 2}
        d.update({"cbk__lidar__overlap_policy": "retime_trim", "cbk__lidar__retime_t0_s": 100.0, "cbk__lidar__retime_period_s": 0.1, "cbk__lidar__retime_seq0": 5, "cbk__lidar__retime_ref_offset_s": -0.1} if mode == "rt" else {"cbk__lidar__overlap_policy": "discard"})
        d.update(kw)
        return "".join("  %s: %s\n" % (k.replace("__", "/"), v) for k, v in d.items() if v is not None) + ("[lidar_overlap_policy] policy=retime_trim\n" if mode == "rt" else "")

    def swe(mode, vox, maxl, fit="pca", on=False, **kw):
        d = {"map__voxel_size": vox, "map__max_layer": maxl, "plane__plane_fit_mode": fit, "line_residual__enabled": str(on).lower()}
        if fit == "debiased" and on:
            d.update({"line_residual__variance": "plane_equivalent", "line_residual__rows": "distance", "plane__plane_gap_floor_en": "true", "line_residual__classify_at_fit": "true", "plane__debiased_line_equiv": "mean_cov"})
        d.update(kw)
        return echo(mode, **d)
    FORCED = "  imu/ds/ds_leaf_size = 0.15  [FORCED: scope 'imu/ds/mode != off' is off]\n"
    D4L = "  voxel_map/plane/line_residual/classify_at_fit = true debiased_line_equiv = mean_cov plane_gap_floor_en = true line_classified=5 line_accepted=5 e1_tiny_kept=0 e0_clamped=0 gaps_floored=3 tilt_voxels_moment=0 tilt_voxels_meancov=4\n"

    def shte(letter, edge, **kw):
        _, pfn, ds, nb = CELLD[letter]
        d = dict(cbk__lidar__point_filter_num=pfn, imu__ds__mode=ds, imu__ds__ds_leaf_size=(None if ds == "off" else 0.1), search__neighborhood_size=nb)
        d.update(kw)
        return swe("rt", edge, 0, "debiased", True, **d) + (FORCED if ds == "off" else "")

    def mk(name, log="  voxel_map/plane/x: 1\n", eng="", imu="", files=None, sentinel="DONE", exit_="exit_code=0 signal=none", requires_missing=()):
        d = os.path.join(runs, name); os.makedirs(d, exist_ok=True)
        json.dump({"sentinel": sentinel, "requires_missing": list(requires_missing)}, open(os.path.join(d, "arm.json"), "w")); open(os.path.join(d, "exit_status.txt"), "w").write(exit_ + "\n")
        open(os.path.join(d, "run.log"), "w").write(log); open(os.path.join(d, "engagement.txt"), "w").write(eng)
        if imu:
            open(os.path.join(d, "engagement_imu.txt"), "w").write(imu)
        for fn, txt in (files or {}).items():
            open(os.path.join(d, fn), "w").write(txt)

    def short_tab(n=500, **mod):
        T = pd.DataFrame({"scan": np.arange(1, n + 1), "t_s": np.linspace(0.0, 59.5, n), "residual_count": 100.0, "fs_n_residuals": 1000.0, "fs_n_voxels": 200.0, "fs_n_voxels_is_plane": 150.0, "fs_iters": 3.0, "fs_dx_pos_mm": 0.1,
                          "fs_nis": 1.0, "fs_reduced_chi2": 1.0, "pos_err_mm": 0.5})
        for k, v in mod.items():
            T[k] = v
        return T

    def put_short(name, T, header=None, sub=""):
        d = os.path.join(runs, name, sub); os.makedirs(d, exist_ok=True)
        if header:
            T = T.copy(); T.columns = header
        T.to_csv(os.path.join(d, "short_scan_all.csv.gz"), index=False, compression="gzip")
    S = lambda g, pre: [s for n, s, _ in g.rows if n.startswith(pre)]
    SD = lambda g, pre: [d for n, s, d in g.rows if n.startswith(pre)]
    nonpass = lambda g, arm_: [(n[:60], s) for n, s, _ in g.rows if (" %s " % arm_) in n and s in ("FAIL", "MISSING") and not n.startswith(("LEN ", "MSG"))]   # the traces are not fabricated for every arm
    GR = {"eee_01": GRID["eee_01"]}
    # ---- sweep arm folders
    mk(L0a, log=swe("dc", 0.2, 0)); mk(L2a, log=swe("dc", 0.2, 2))
    mk(L0b, log=swe("dc", 0.25, 0, "debiased", True, plane__plane_fit_reasons_en="true", plane__plane_fit_trace_frames=10), eng=D4L); mk(L2b, log=swe("dc", 0.25, 2, "debiased", True, plane__plane_fit_reasons_en="true", plane__plane_fit_trace_frames=10), eng=D4L)
    mk(L0c, log=swe("dc", 0.25, 2)); mk(L2c, log=swe("dc", 0.25, 2))
    for a_ in (L0d, L0e, L0f, L0g, L0n, L0o, L0p, L0q, L0r):
        mk(a_, log=swe("dc", 0.2, 0))
    mk(L0h, log=swe("dc", 0.2, 0, "pca", True)); mk(L2h, log=swe("dc", 0.2, 2, "pca", True, cbk__lidar__point_filter_num=4))
    mk(L0i, log=swe("dc", 0.2, 0, "debiased", True, line_residual__classify_at_fit=None, plane__debiased_line_equiv=None)); mk(L2i, log=swe("dc", 0.2, 2, "debiased", True), eng=D4L)
    mk(L0j, log=swe("dc", 0.25, 0, "debiased", True, line_residual__classify_at_fit=None, plane__debiased_line_equiv=None), eng=D4L); mk(L2j, log=swe("dc", 0.25, 2, "debiased", True), eng=D4L)
    mk(L0k, log=swe("dc", 0.25, 0, "debiased", True, plane__debiased_line_equiv="off"), eng=D4L); mk(L2k, log=swe("dc", 0.25, 2, "debiased", True), eng=D4L)
    mk(L2m, log=swe("dc", 0.2, 0, "pca", True)); mk(L2s, log=swe("dc", 0.2, 2))
    # ---- SHT arm folders and tables
    mk(S_A2, log=shte("A", 0.2), eng=D4L); mk(S_H2, log=shte("H", 0.2), eng=D4L)
    mk(S_B2, log=shte("B", 0.2), eng=D4L); mk(S_C2, log=shte("C", 0.2), eng=D4L); mk(S_E2, log=shte("E", 0.2, search__neighborhood_size=1), eng=D4L)
    mk(S_A25, log=shte("A", 0.25), eng=D4L, files={"state_trace_blank_runid.sha256": "aa" * 32 + "\n"})
    mk(DET_S, log=shte("A", 0.25), eng=D4L, files={"state_trace_blank_runid.sha256": "aa" * 32 + "\n"})
    for a_ in (S_A2, S_H2, S_C2, S_E2, S_A25, DET_S):
        put_short(a_, short_tab())
    # S_B2 has no table at all
    mk(SMA, log=shte("A", 0.2), eng=D4L); mk(SMH, log=shte("H", 0.25), eng=D4L)
    put_short(SMA, short_tab()); put_short(SMH, short_tab())
    mk(SM0, log=swe("dc", 0.2, 0))
    # ---- refuse arms, removed keys
    RM = lambda k, extra="was removed in R85 (F-520) and has no replacement": "terminate: configuration key '%s' %s; delete it from the configuration" % (k, extra)
    FL = dict(sentinel="FAILED", exit_="exit_code=-6 signal=SIGABRT")
    mk("rf_sw", log=RM("lio/point_cov/sample_window_s"), **FL); mk("rf_mode", log=RM("lio/point_cov/mode", "is no longer accepted"), **FL)
    mk("rf_nomatch", log="terminate: something unrelated, removed", **FL); mk("rf_zero", log=RM("lio/point_cov/sample_window_s"), sentinel="DONE", exit_="exit_code=0 signal=none")
    mk("x_clean"); mk("x_rm_job"); mk("x_dcret"); mk("x_rm_echo", log="  lio/point_cov/sample_window_s: 0.005\n"); mk("x_rm_eng", eng="  lio/point_cov/sample_window_s = 0.02\n"); mk("x_zupt", log="  lio/zupt/mode = off\n")
    mk("x_pv", eng="  voxel_map/plane/plane_var_min = 0.0004 rows=12\n")
    # ---- identity / determinism inputs
    sid = np.arange(1, 41)

    def trace(a, shift=0.0):
        d = os.path.join(runs, a); os.makedirs(d, exist_ok=True)
        pd.DataFrame({"scan_id": sid, "px": sid * 1e-3 + (shift * 1e-3 * (sid >= 20)), "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d, "pos_trace_all.csv.gz"), index=False, compression="gzip")
        json.dump({"excursion_mm": 10.0, "settled_offset_mm": 5.0}, open(os.path.join(d, "arm_summary.json"), "w"))
    ID_DC, ID_RT, ID_MQ = "r85_dc_id_eee_01_fq", "r85_rt_id_eee_01_fq", "r85_rt_id_eee_01_mq"
    mk(ID_DC, log=echo("dc")); mk(ID_RT, log=echo("rt")); mk(ID_MQ, log=echo("rt"))
    trace(ID_DC); trace(ID_RT, 0.05); trace(ID_MQ)                       # the rt identity arm differs from its reference by 0.05 mm at scan 20
    r84 = os.path.join(root, "R84"); refs = ("r84_dc_id_eee_01_fq", "r84_rt_id_eee_01_fq", "r84_rt_id_eee_01_mq")
    for a_ in refs:
        os.makedirs(os.path.join(r84, "compact", a_))
        pd.DataFrame({"scan_id": sid, "px": sid * 1e-3, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(r84, "compact", a_, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    pd.DataFrame([dict(arm=a_, excursion_mm=10.0, settled_offset_mm=5.0) for a_ in refs]).to_csv(os.path.join(r84, "phase_summary.csv"), index=False)
    mk(DET_X, log=echo("rt"), files={"state_trace_blank_runid.sha256": "aa" * 32 + "\n"}); mk(DET_O, log=echo("rt"), files={"state_trace_blank_runid.sha256": "bb" * 32 + "\n"})
    trace(DET_X, 0.02); trace(DET_O)
    mk("msg_dc"); trace("msg_dc")
    ph = pd.DataFrame({"arm": [L0e, L0a, L2a, L0b, L2b, L0c], "gt_max_mm": [600.0, 5.0, 100.0, 700.0, 50.0, 10.0]})
    d_ = os.path.join(runs, L0b); pd.DataFrame({"scan_id": [1, 2], "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(d_, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    ro = [dict(followup="F-526", title="short", arms=[S_A2, S_B2, SMA], file="short_scan_all.csv.gz", columns=SHORT_COLS, min_rows=400, where="return", source="arm"),
          dict(followup="F-452", title="gen", arms=[], file="r85_l0_rows.csv", columns=["arm", "d_ate_mm"], min_rows=3, where="return", source="generated"),
          dict(followup="F-452", title="gen2", arms=[], file="r85_l0_paired.csv", columns=[], min_rows=1, where="return", source="generated"),
          dict(followup="F-452", title="own", arms=[], file="r85_gates.txt", columns=[], min_rows=1, where="return", source="generated"),
          dict(followup="F-326", title="exit", arms=[S_A2, "x_clean"], file="exit_status.txt", columns=[], min_rows=1, where="archive", source="arm")]
    json.dump(ro, open(os.path.join(mdir, "required_outputs.json"), "w"))
    retd = os.path.join(root, "ret"); os.makedirs(retd)
    pd.DataFrame({"arm": ["a", "b"], "d_ate_mm": [1.0, 2.0]}).to_csv(os.path.join(retd, "r85_l0_rows.csv"), index=False)
    # ================================================================ run 1: everything dispatched
    nsht = len([1 for r, _ in ARMS if r["group"] == "SHT"])
    cnt = lambda grp: len([1 for r, _ in ARMS if r["group"] == grp])
    g, idr = run_gates(runs, mdir, grid_retime=GR, ref_returns=[r84], phase=ph, ret_dirs=[retd], expect={"L0": (cnt("L0"),), "L2R": (cnt("L2R"),), "SHT": (nsht,)})
    ck(all(s == "PASS" for n, s, _ in g.rows if n.startswith("FAMCOUNT") and "manifest holds" in n), "FAMCOUNT must PASS for the fixture counts")
    g2, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=L0a)
    ck(all(s == "FAIL" for n, s, _ in g2.rows if n.startswith("FAMCOUNT") and "manifest holds" in n), "FAMCOUNT must FAIL when the manifest holds other counts than 288 / 198 / 144")
    ck(EXPECT == {"L0": (288,), "L2R": (198,), "SHT": (144,)}, "the default EXPECT is the R85 table")
    # ---- L0ENG / L2RENG
    ck(not nonpass(g, L0a) and not nonpass(g, L2a), "the good L0 / L2R pair has no FAIL or MISSING row: %s %s" % (nonpass(g, L0a), nonpass(g, L2a)))
    ck(S(g, "L0ENG %s params echo" % L0a) == [] and any(n.startswith("CELL L0ENG L0_dc_eee_01_fq") for n, s, _ in g.rows), "sweep PASS rows are folded into CELL rows (no own row for a PASS)")
    ck(S(g, "L0ENG %s params echo" % L0c) == ["FAIL"] and "max_layer" in SD(g, "L0ENG %s params echo" % L0c)[0], "L0 echo max_layer 2 against the job's 0 must FAIL")
    ck(S(g, "L0ENG %s manifest" % L0d) == ["FAIL"] and "arm name says" in SD(g, "L0ENG %s manifest" % L0d)[0] and "arm_table edge" in SD(g, "L0ENG %s manifest" % L0d)[0], "job edge differs from the edge in the name and from arm_table: FAIL")
    ck(S(g, "L0ENG %s l2_ref" % L0e) == ["FAIL"] and S(g, "L0ENG %s l2_ref" % L0f) == ["FAIL"] and S(g, "L0ENG %s l2_ref" % L0q) == ["FAIL"] and "prefix" in SD(g, "L0ENG %s l2_ref" % L0q)[0], "l2_source outside the set FAIL, empty l2_ref FAIL, reference prefix against source FAIL")
    ck(S(g, "L0ENG %s twin" % L0g) == ["FAIL"], "R85 l2_source without a twin L2R arm: FAIL")
    ck(S(g, "L0ENG %s twin" % L0h) == ["FAIL"] and "point_filter_num" in SD(g, "L0ENG %s twin" % L0h)[0], "the pair differs in more than max_layer: FAIL")
    ck(S(g, "L0ENG %s twin" % L0a) == [] and S(g, "L2RENG %s control" % L2a) == [], "the good pair's twin and control rows are PASS (folded into CELL rows)")
    ck(S(g, "L0ENG %s manifest" % L0n) == ["FAIL"] and "point_filter_num" in SD(g, "L0ENG %s manifest" % L0n)[0], "L0 job with point_filter_num 1: manifest FAIL")
    ck(S(g, "L0ENG %s manifest" % L0o) == ["FAIL"] and "neighborhood_size" in SD(g, "L0ENG %s manifest" % L0o)[0], "L0 job that sets neighborhood_size: manifest FAIL")
    ck(S(g, "L0ENG %s manifest" % L0r) == ["FAIL"] and "arm_table nb" in SD(g, "L0ENG %s manifest" % L0r)[0], "L0 arm_table nb column 2 against the absent job key: manifest FAIL")
    ck(S(g, "L0ENG %s manifest" % L0p) == ["FAIL"] and "imu.ds.mode" in SD(g, "L0ENG %s manifest" % L0p)[0], "L0 job with ds off: manifest FAIL")
    ck(S(g, "L0ENG %s deb + lines on" % L0b) == [] and not nonpass(g, L0b) and not nonpass(g, L2b), "deb+on arm with classify / equiv in the echo and the D4 line: PASS: %s %s" % (nonpass(g, L0b), nonpass(g, L2b)))
    ck(S(g, "L0ENG %s deb + lines on" % L0i) == ["MISSING"], "deb+on arm where neither echo nor engagement shows classify / equiv: MISSING")
    ck(S(g, "L0ENG %s deb + lines on" % L0j) == [] and S(g, "D4ENG %s" % L0j) == [], "classify / equiv only in the engagement line: PASS")
    ck(S(g, "L0ENG %s deb + lines on" % L0k) == ["FAIL"], "debiased_line_equiv printed off against the job's mean_cov: FAIL")
    ck(S(g, "L2RENG %s params echo" % L2m) == ["FAIL"] and S(g, "L2RENG %s control" % L2m) == ["FAIL"], "L2R echo max_layer 0 against the job's 2 FAIL; control that is no L0 arm FAIL")
    ck(S(g, "L2RENG %s control" % L2s) == ["FAIL"] and "l2_source" in SD(g, "L2RENG %s control" % L2s)[0], "L2R arm whose L0 control does not name it as its R85 reference: FAIL")
    ck(S(g, "L2MAP") == ["PASS"], "l2_reference_map.tsv agrees with the arm_table: %s" % [(n, s, d) for n, s, d in g.rows if n.startswith("L2MAP")])
    # ---- SHTENG / NBCFG / DSCFG
    ck(not nonpass(g, S_A2) and not nonpass(g, S_H2) and not nonpass(g, S_A25), "good SHT arms (cell A, cell H with ds off) have no FAIL or MISSING row: %s %s" % (nonpass(g, S_A2), nonpass(g, S_H2)))
    ck(S(g, "SHTENG %s manifest" % S_A2) == ["PASS"] and S(g, "SHTENG %s manifest" % S_H2) == ["PASS"] and S(g, "SHTENG %s params echo" % S_A2) == ["PASS"] and S(g, "SHTENG %s params echo" % S_H2) == ["PASS"], "SHTENG manifest and echo rows PASS on cells A and H")
    ck(S(g, "SHTENG %s manifest" % S_B2) == ["FAIL"] and "point_filter_num" in SD(g, "SHTENG %s manifest" % S_B2)[0], "SHT job disagreeing with its cell letter (point filter): SHTENG manifest FAIL")
    ck(S(g, "SHTENG %s manifest" % S_D2) == ["FAIL"] and "arm_table nb" in SD(g, "SHTENG %s manifest" % S_D2)[0], "SHT arm_table nb column against the job and the cell: SHTENG manifest FAIL")
    ck(S(g, "SHTENG %s manifest" % S_C2) == ["FAIL"] and "ds_leaf_size" in SD(g, "SHTENG %s manifest" % S_C2)[0], "ds off job with a leaf: SHTENG manifest FAIL")
    ck(S(g, "NBCFG %s" % S_E2) == ["FAIL"] and S(g, "SHTENG %s params echo" % S_E2) == ["FAIL"], "echo neighborhood_size 1 against the job's 2: NBCFG FAIL, rolled up into SHTENG")
    ck(S(g, "NBCFG %s" % S_A2) == ["PASS"] and S(g, "NBCFG %s" % S_H2) == ["PASS"], "NBCFG PASS where the echo prints the job's value (1 and 2)")
    ck(S(g, "DSCFG %s ds off" % S_H2) == ["PASS"] and S(g, "DSCFG %s effective" % S_H2) == ["PASS"], "ds off arm (null leaf): the mode is judged against the echo, the leaf is printed as dead scope: PASS")
    mk(S_H2, log=shte("H", 0.2).replace("  [FORCED: scope 'imu/ds/mode != off' is off]", ""), eng=D4L)
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=S_H2)
    ck(S(gx, "DSCFG %s ds off" % S_H2) == ["FAIL"], "ds off arm whose echo prints a live leaf (no FORCED marker): FAIL")
    mk(S_H2, log=shte("H", 0.2).replace(FORCED, ""), eng=D4L)
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=S_H2)
    ck(S(gx, "DSCFG %s ds off" % S_H2) == ["INFO"] and S(gx, "DSCFG %s effective" % S_H2) == ["PASS"], "ds off arm whose echo prints no leaf: INFO, never PASS")
    mk(S_H2, log=shte("H", 0.2), eng=D4L)
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=S_A2)
    ck(S(gx, "NBCFG %s" % S_A2) == ["PASS"], "NBCFG row exists for a dispatched SHT arm")
    mk(S_A2, log=shte("A", 0.2).replace("  search/neighborhood_size: 1\n", ""), eng=D4L)
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=S_A2)
    ck(S(gx, "NBCFG %s" % S_A2) == ["MISSING"], "explicit neighborhood_size not printed by the echo: MISSING, never PASS")
    mk(S_A2, log=shte("A", 0.2), eng=D4L)
    # ---- SHTSCAN on the fixtures
    ck(S(g, "SHTSCAN %s " % S_A2) == ["PASS"] and S(g, "SHTSCAN %s " % SMA) == ["PASS"] and S(g, "SHTSCAN %s " % SMH) == ["PASS"] and S(g, "SHTSCAN %s " % S_B2) == ["MISSING"], "SHTSCAN: tables present PASS, no table MISSING: %s" % [(n[:40], s) for n, s, _ in g.rows if n.startswith("SHTSCAN")])
    ck(sorted(n.split()[1] for n, _, _ in g.rows if n.startswith("SHTSCAN ")) == sorted(r["arm"] for r, _ in ARMS if r.get("cell")), "SHTSCAN has one row for each arm that carries a cell letter")
    tt = {"t": os.path.join(root, "tt")}
    TM = {"id": "zz", "kind": "sht", "cell": "A"}
    rd_ = lambda T, **kw: shtscan_rows(TM, tt["t"], [])[0]
    variants = (("good", short_tab(), None, "PASS", ""), ("header", short_tab(), list(reversed(SHORT_COLS)), "FAIL", "header"), ("short", short_tab(100), None, "FAIL", "rows 100"),
                ("nullres", short_tab(fs_n_residuals=np.nan), None, "FAIL", "fs_n_residuals null on 500"), ("nullcount", short_tab().assign(residual_count=[np.nan] * 3 + [1.0] * 497), None, "FAIL", "residual_count null on 3"),
                ("nullvox", short_tab(fs_n_voxels=np.nan), None, "FAIL", "fs_n_voxels null on 500"), ("scan", short_tab().assign(scan=[1, 2, 2] + list(range(4, 501))), None, "FAIL", "strictly increasing"),
                ("tbig", short_tab().assign(t_s=np.linspace(0.0, 70.0, 500)), None, "FAIL", "t_s"), ("tneg", short_tab().assign(t_s=np.linspace(-1.0, 59.0, 500)), None, "FAIL", "t_s"))
    for nm_, T, hdr, want, frag in variants:
        shutil.rmtree(os.path.join(tt["t"], "zz"), ignore_errors=True)
        d_ = os.path.join(tt["t"], "zz"); os.makedirs(d_, exist_ok=True)
        T2 = T.copy()
        if hdr:
            T2.columns = hdr
        T2.to_csv(os.path.join(d_, "short_scan_all.csv.gz"), index=False, compression="gzip")
        r_ = shtscan_rows(TM, tt["t"], [])[0]
        ck(r_[1] == want and frag in r_[2], "SHTSCAN variant %s must be %s (%s): got %s %s" % (nm_, want, frag, r_[1], r_[2]))
    shutil.rmtree(os.path.join(tt["t"], "zz"))
    ck(shtscan_rows(TM, tt["t"], [])[0][1] == "MISSING", "SHTSCAN without a file: MISSING")
    # ---- L0SMOKE
    ck(S(g, "L0SMOKE %s" % SMA) == ["PASS"] and S(g, "L0SMOKE %s" % SMH) == ["PASS"] and S(g, "L0SMOKE %s" % SM0) == ["PASS"] and S(g, "L0SMOKE all 3") == ["PASS"], "the three smoke arms PASS on good fixtures: %s" % [(n[:50], s, d) for n, s, d in g.rows if n.startswith("L0SMOKE")])
    ck(not nonpass(g, SMA) and not nonpass(g, SMH) and not nonpass(g, SM0), "good smoke arms have no FAIL or MISSING row: %s %s %s" % (nonpass(g, SMA), nonpass(g, SMH), nonpass(g, SM0)))
    hf = os.path.join(root, "HOLD_A.txt")
    # ---- refuse, removed keys, zupt, noextra
    ck(S(g, "REMOVED rf_sw refused") == ["PASS"] and S(g, "REMOVED rf_sw the refusal") == ["PASS"] and S(g, "REMOVED rf_mode refused") == ["PASS"] and S(g, "REMOVED rf_mode the refusal") == ["INFO"], "the two refuse arms: exit != 0 and the needle PASS; `removed` PASS or INFO")
    ck(S(g, "REMOVED rf_nomatch refused") == ["FAIL"] and "False" in SD(g, "REMOVED rf_nomatch refused")[0], "refuse arm whose text lacks the needle: FAIL")
    ck(S(g, "REMOVED rf_zero refused") == ["FAIL"], "refuse arm that exited 0: FAIL")
    ck(S(g, "REFUSE rf_sw") == ["PASS"] and S(g, "REFUSE rf_zero") == ["FAIL"], "REFUSE row: sentinel FAILED and the needle PASS; sentinel DONE FAIL")
    ck(S(g, "REMOVED x_clean none") == ["PASS"] and S(g, "REMOVED x_rm_echo none") == ["FAIL"] and S(g, "REMOVED x_rm_job none") == ["FAIL"] and S(g, "REMOVED x_rm_eng none") == ["FAIL"], "lio.point_cov.sample_window_s in a non-refuse arm's job (even 0) / echo / engagement text: REMOVED FAIL; clean arm PASS")
    ck("sample_window_s" in SD(g, "REMOVED x_rm_job none")[0] and "sample_window_s" in SD(g, "REMOVED x_rm_echo none")[0], "the REMOVED detail names the sample_window_s key")
    ck(S(g, "ZUPTGONE x_zupt") == ["FAIL"] and S(g, "ZUPTGONE x_clean") == ["PASS"], "lio/zupt in an echo: ZUPTGONE FAIL")
    ck(S(g, "NOEXTRA x_pv") == ["FAIL"] and S(g, "NOEXTRA x_clean") == ["PASS"], "NOEXTRA: plane variance engagement without the keys FAIL")
    ck(not any(n.startswith(("SWENG", "TBPNENG", "CMPCFG", "CMPPAIR", "SWCFG", "TBPNCFG")) for n, _, _ in g.rows), "no row of a removed R84 gate")
    # ---- rt grid and the dc overlap rule
    gx, _ = run_gates(runs, mdir, grid_retime={"eee_01": dict(GRID["eee_01"], seq0=6)}, dispatched=S_A2)
    ck(S(gx, "RTGRID %s" % S_A2) == ["FAIL"], "rt arm whose job grid differs from the bag grid: RTGRID FAIL")
    gx, _ = run_gates(runs, mdir, dispatched=S_A2)
    ck(S(gx, "RTGRID %s" % S_A2) == ["MISSING"], "no grid given: RTGRID MISSING")
    ck(any(n.startswith("CELL OVLCFG L0_dc") and s == "PASS" for n, s, _ in g.rows) and S(g, "OVLCFG x_dcret overlap mode dc") == ["FAIL"], "dc arm without retime / policy keys: OVLCFG PASS (folded); a dc arm with a retime key FAIL")
    # ---- completeness, trace rule of the sweep groups
    ck(S(g, "TRACE %s" % L0e) == ["MISSING"] and S(g, "TRACE %s" % L0b) == [] and not any(n.startswith("TRACE %s" % L2a) for n, _, _ in g.rows), "collapsed sweep arm (gt_max_mm > 500) needs a pos trace; others need none")
    ck(any(n.startswith("SWEEPCOMP %s has a phase_summary row" % L0d) and s == "MISSING" for n, s, _ in g.rows) and not any(n.startswith("LEN %s" % L0a) for n, _, _ in g.rows), "SWEEPCOMP: an arm without a phase row is MISSING; sweeps have no LEN row")
    ck(S(g, "COMPLETE %s" % S_A2) == ["PASS"], "COMPLETE PASS where arm.json lists no missing required file")
    mk(S_A2, log=shte("A", 0.2), eng=D4L, requires_missing=["state_trace.csv", "frame_stats.txt"])
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=S_A2)
    ck(S(gx, "COMPLETE %s" % S_A2) == ["FAIL"] and "state_trace.csv" in SD(gx, "COMPLETE %s" % S_A2)[0], "COMPLETE FAIL naming the missing required files (SHT arms require state_trace.csv and frame_stats.txt)")
    mk(S_A2, log=shte("A", 0.2), eng=D4L)
    ck(all(("state_trace.csv" in t["requires"] and "frame_stats.txt" in t["requires"]) for t in load_manifest(mdir) if t["cell"]), "the fixture's cell arms require state_trace.csv and frame_stats.txt (as the generator's)")
    # ---- REQUIRED
    rq = {n.split("(")[0].strip(): (s, d) for n, s, d in g.rows if n.startswith("REQUIRED ")}
    ck(rq.get("REQUIRED short_scan_all.csv.gz")[0] == "MISSING" and S_B2 in rq["REQUIRED short_scan_all.csv.gz"][1], "REQUIRED short_scan_all.csv.gz: MISSING naming the arm without a table: %s" % (rq.get("REQUIRED short_scan_all.csv.gz"),))
    ck(any(n.startswith("REQUIRED r85_l0_rows.csv") and s == "FAIL" and "rows 2 < 3" in d for n, s, d in g.rows), "REQUIRED generated file with too few rows: FAIL")
    ck(any(n.startswith("REQUIRED r85_l0_paired.csv") and s == "MISSING" for n, s, _ in g.rows) and any(n.startswith("REQUIRED r85_gates.txt") and s == "INFO" for n, s, _ in g.rows), "REQUIRED: absent generated file MISSING; this tool's own file INFO")
    ck(any(n.startswith("REQUIRED exit_status.txt") and s == "PASS" for n, s, _ in g.rows), "REQUIRED archive entry found in the arm folders: PASS")
    # ================================================================ hold file: what writes it and what does not
    ck(write_hold(g, hf) is True, "run 1 holds (the rt identity arm differs, the whole-bag determinism pair differs)")
    ht = open(hf).read()
    hb = "\n".join(l for l in ht.splitlines() if l.startswith(("FAIL ", "MISSING ")))
    ck("IDENT %s" % ID_RT in hb and "DET %s" % DET_X in hb and "Stop dispatching unstarted queues and report" in ht, "the hold file names the IDENT and DET gates and arms and the instruction")
    ck("SHTSCAN" not in hb and "DETS" not in hb and "L0ENG" not in hb and "L0SMOKE" not in hb and "REMOVED" not in hb, "SHTSCAN of the non-smoke arm, the sweep, smoke and removed-key FAIL rows do NOT write the hold: " + hb[:600])
    ck(any(n.startswith("IDENT %s POS" % ID_DC) and s == "PASS" for n, s, _ in g.rows) and not any(n.startswith("IDENT %s" % ID_DC) and s in ("FAIL", "MISSING") for n, s, _ in g.rows), "identity arm equal to its R84 reference: PASS rows, no FAIL or MISSING")
    ck(any(n.startswith("IDENT %s" % ID_RT) and s == "FAIL" for n, s, _ in g.rows) and {r["arm"] for r in idr} == {ID_DC, ID_RT, ID_MQ}, "IDENT: the arm with a 0.05 mm shift FAILs; the identity csv holds the three arms")
    # a reference return that lacks the ref arm, and no reference return at all, are MISSING never PASS
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=ID_DC)
    ck(all(s == "MISSING" for n, s, _ in gx.rows if n.startswith("IDENT ")), "IDENT without --ref-return: MISSING")
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=ID_DC, ref_returns=[os.path.join(root, "nowhere")])
    ck(any(n.startswith("IDENT ") and s == "MISSING" for n, s, _ in gx.rows) and not any(n.startswith("IDENT ") and n.split()[2:3] == ["POS"] and s == "PASS" for n, s, _ in gx.rows), "IDENT against a return that lacks the ref arm: MISSING, not PASS")
    # DETS
    ck(S(g, "DETS %s state_trace digest" % DET_S) == ["PASS"] and S(g, "DETS %s short_scan" % DET_S) == ["PASS"] and "differing rows 0" in SD(g, "DETS %s short_scan" % DET_S)[0], "DETS PASS: equal digest and equal short_scan table")
    ck(S(g, "DETCFG %s" % DET_S) == ["PASS"] and S(g, "DETCFG %s" % DET_X) == ["PASS"], "DETCFG PASS: the repeat has the original's job")
    # ---- fresh hold file: only SHTSCAN of a non-smoke arm, sweep failures, L0SMOKE and REMOVED fail -> no hold
    runs_keep = runs
    for a_ in (ID_RT, DET_X):
        shutil.rmtree(os.path.join(runs, a_))
    mk(ID_RT, log=echo("rt")); trace(ID_RT)
    mk(DET_X, log=echo("rt"), files={"state_trace_blank_runid.sha256": "bb" * 32 + "\n"}); trace(DET_X)
    g3, _ = run_gates(runs, mdir, grid_retime=GR, ref_returns=[r84], phase=ph, ret_dirs=[retd])
    hf3 = os.path.join(root, "HOLD_B.txt")
    ck(write_hold(g3, hf3) is False and not os.path.exists(hf3), "with DET / IDENT fixed, the remaining FAILs (SHTSCAN missing on a non-smoke arm, sweep, REMOVED, SHTENG, L0SMOKE-free) write NO hold file: %s" % [(n[:50], s) for n, s, _ in g3.rows if s == "FAIL" and n.startswith(("DET", "IDENT", "SHTSCAN"))])
    ck(S(g3, "SHTSCAN %s " % S_B2) == ["MISSING"], "SHTSCAN MISSING on a non-smoke arm is a fact (no hold)")
    # ---- DETS FAIL: the repeat's table differs in 3 rows, and its digest differs -> hold
    T = short_tab(); T.loc[10:12, "fs_n_residuals"] += 1.0
    put_short(DET_S, T)
    g4, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s" % (DET_S, S_A25))
    ck(S(g4, "DETS %s short_scan" % DET_S) == ["FAIL"] and "differing rows 3" in SD(g4, "DETS %s short_scan" % DET_S)[0], "DETS table differing in 3 rows: FAIL with the count: %s" % SD(g4, "DETS %s short_scan" % DET_S))
    hf4 = os.path.join(root, "HOLD_C.txt")
    ck(write_hold(g4, hf4) and "DETS %s" % DET_S in open(hf4).read(), "a DETS FAIL writes the hold file")
    put_short(DET_S, short_tab())
    open(os.path.join(runs, DET_S, "state_trace_blank_runid.sha256"), "w").write("cc" * 32 + "\n")
    g5, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s" % (DET_S, S_A25))
    ck(S(g5, "DETS %s state_trace digest" % DET_S) == ["FAIL"] and S(g5, "DETS %s short_scan" % DET_S) == ["PASS"], "DETS digest differing, table equal: digest row FAIL, table row PASS")
    os.remove(os.path.join(runs, DET_S, "state_trace_blank_runid.sha256"))
    pd.DataFrame([dict(arm=S_A25, state_trace_sha256_run_id_blanked="aa" * 32), dict(arm=DET_S, state_trace_sha256_run_id_blanked="aa" * 32)]).to_csv(os.path.join(retd, "completeness.csv"), index=False)
    g5b, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s" % (DET_S, S_A25), ret_dirs=[retd])
    ck(S(g5b, "DETS %s state_trace digest" % DET_S) == ["PASS"], "digest read from completeness.csv column state_trace_sha256_run_id_blanked when the arm folder has no .sha256 file")
    g5c, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s" % (DET_S, S_A25))
    ck(S(g5c, "DETS %s state_trace digest" % DET_S) == ["MISSING"], "no digest anywhere: MISSING, never PASS")
    open(os.path.join(runs, DET_S, "state_trace_blank_runid.sha256"), "w").write("aa" * 32 + "\n")
    # ---- SHTSCAN FAIL / MISSING of a smoke arm holds
    for nm_, how in (("missing", "rm"), ("nullres", "null"), ("header", "hdr")):
        T = short_tab(fs_n_residuals=np.nan) if how == "null" else short_tab()
        if how == "rm":
            os.remove(os.path.join(runs, SMA, "short_scan_all.csv.gz"))
        else:
            put_short(SMA, T, list(reversed(SHORT_COLS)) if how == "hdr" else None)
        g6, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s,%s" % (SMA, SMH, SM0))
        hfx = os.path.join(root, "HOLD_S_%s.txt" % nm_)
        want = "MISSING" if how == "rm" else "FAIL"
        ck(S(g6, "SHTSCAN %s " % SMA) == [want] and write_hold(g6, hfx) and ("SHTSCAN %s" % SMA) in open(hfx).read() and any(l.startswith("%s SHTSCAN %s " % (want, SMA)) for l in open(hfx).read().splitlines()), "smoke arm SHTSCAN %s holds and the hold names the gate and arm" % nm_)
    put_short(SMA, short_tab())
    put_short(SMH, short_tab(fs_n_residuals=np.nan))
    g7, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s,%s" % (SMA, SMH, SM0))
    hf7 = os.path.join(root, "HOLD_H.txt")
    ck(write_hold(g7, hf7) and ("SHTSCAN %s" % SMH) in open(hf7).read() and ("SHTSCAN %s" % SMA) not in open(hf7).read(), "the second smoke arm (cell H) holds on its own")
    put_short(SMH, short_tab())
    # ---- L0SMOKE failures name the arm and the key; no hold
    mk(SMH, log=shte("H", 0.25, imu__ds__mode="first"), eng=D4L)
    g8, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s,%s" % (SMA, SMH, SM0))
    lr = [(n, s, d) for n, s, d in g8.rows if n.startswith("L0SMOKE %s" % SMH)]
    ck(len(lr) == 1 and lr[0][1] == "FAIL" and SMH in lr[0][2] and "imu.ds.mode" in lr[0][2], "L0SMOKE failure names the arm and the key: %s" % lr)
    ck(not any(n.startswith("L0SMOKE all") for n, _, _ in g8.rows) and write_hold(g8, os.path.join(root, "HOLD_I.txt")) is False, "a failed smoke key is reported and writes NO hold file; no all-PASS L0SMOKE row")
    mk(SM0, log=swe("dc", 0.2, 0), exit_="exit_code=1 signal=none", sentinel="FAILED")
    g9, _ = run_gates(runs, mdir, grid_retime=GR, dispatched="%s,%s,%s" % (SMA, SMH, SM0))
    lr = [(n, s, d) for n, s, d in g9.rows if n.startswith("L0SMOKE %s" % SM0)]
    ck(len(lr) == 1 and lr[0][1] == "FAIL" and "exit_code 1" in lr[0][2], "a smoke arm that exited non-zero: L0SMOKE FAIL naming the arm and the exit code")
    mk(SM0, log="  something: else\n")
    g9, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=SM0)
    lr = [(n, s, d) for n, s, d in g9.rows if n.startswith("L0SMOKE %s" % SM0)]
    ck(len(lr) == 1 and lr[0][1] in ("FAIL", "MISSING") and "params echo absent" in lr[0][2], "a smoke arm with no params echo: not accepted")
    g9, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=SMA)
    ck(S(g9, "L0SMOKE %s not dispatched" % SMH) == ["NOTRUN"] and S(g9, "L0SMOKE %s not dispatched" % SM0) == ["NOTRUN"], "undispatched smoke arms: NOTRUN")
    # ---- MSGSCANS: INFO for a dc arm
    d = os.path.join(runs, "msg_dc"); mk("msg_dc"); trace("msg_dc")
    L = pd.DataFrame({"msg_index": range(50), "seq": range(50), "header_stamp": 0.0, "scan_time": 0.0, "gap_header_s": 0.1, "n_raw": 1, "n_kept": 1, "t_first": 0.0, "t_last": 0.0, "off_first_s": 0.0, "off_last_s": 0.0, "span_s": 0.1,
                      "prev_accepted_last_t": 0.0, "overlap_s": [0.0] * 30 + [0.01] * 20, "verdict": [0] * 30 + [1] * 20})
    L.to_csv(os.path.join(d, "lidar_msg_log_all.csv.gz"), index=False, compression="gzip")
    gx, _ = run_gates(runs, mdir, dispatched="msg_dc")
    ck(any(n.startswith("MSGSCANS msg_dc dc arm") and s == "INFO" for n, s, _ in gx.rows) and not any(n.startswith("MSGSCANS") and s == "FAIL" for n, s, _ in gx.rows), "MSGSCANS for a dc arm is INFO")
    # ---- LEN keeps the 60 s rule for the short arms
    pd.DataFrame({"scan_id": range(1, 596), "px": 0.0, "py": 0.0, "pz": 0.0, "yaw_rad": 0.0}).to_csv(os.path.join(runs, S_A2, "pos_trace_all.csv.gz"), index=False, compression="gzip")
    gx, _ = run_gates(runs, mdir, grid_retime=GR, dispatched=S_A2)
    ck(S(gx, "LEN %s first 60 s: between 560 and 700" % S_A2) == ["PASS"], "LEN: the 60 s rule applies to a short arm with 595 scans")
    # ---- PFT: arms whose job sets plane_fit_trace_frames
    open(os.path.join(retd, "plane_fit_trace_counts_all.csv"), "w").write("arm,rows\n%s,12\n" % L0b)
    gx, _ = run_gates(runs, mdir, ret_dirs=[retd], dispatched=L0b)
    ck(any(n.startswith("PFT all 1 arms") and s == "PASS" for n, s, _ in gx.rows), "PFT passes for the arm with trace rows: %s" % [n for n, s, _ in gx.rows if "PFT" in n])
    open(os.path.join(retd, "plane_fit_trace_counts_all.csv"), "w").write("arm,rows\nother,3\n")
    gx, _ = run_gates(runs, mdir, ret_dirs=[retd], dispatched=L0b)
    ck(any(n.startswith("PFT %s" % L0b) and s == "MISSING" for n, s, _ in gx.rows), "PFT: no rows for the arm is MISSING")
    # ---- the code of removed gates is gone
    src = open(os.path.abspath(__file__)).read().split("def selftest")[0]
    ck(not re.search(r"def (sweng_rows|tbpn_rows|cmp_static|cmp_pair|smoke_row|sw_value)|K_TBPN|SW_LINE|sw_ms|CMPCFG|CMPPAIR|T3ENG", src), "no code of the removed R84 gates (SWENG, TBPNENG, CMPCFG, CMPPAIR, T3ENG) remains in the gate code")
    shutil.rmtree(root, ignore_errors=True)
    print("SELFTEST r85_gates", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (nck[0], len(fails)),
          "" if fails else "(L0ENG / L2RENG, L2MAP, SHTENG, NBCFG, DSCFG ds-off, SHTSCAN, DETS, DETCFG, IDENT, L0SMOKE, REMOVED with five keys, ZUPTGONE, COMPLETE, REQUIRED, RTGRID / OVLCFG, TRACE / SWEEPCOMP rows PASS on good fixtures and FAIL, MISSING, INFO or NOTRUN on bad ones; "
          "DET / DETPOS / DETS / IDENT FAIL and a FAIL or MISSING SHTSCAN of a smoke arm write HOLD_FROM_GATES.txt; L0SMOKE, REMOVED, sweep and other SHTSCAN failures write none; "
          "synthetic fixtures, planner sandbox, not evidence)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--manifest", default="r85_manifest"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grid-retime", default=None); ap.add_argument("--grid-bag", default=None); ap.add_argument("--ref-return", action="append", default=[]); ap.add_argument("--ref-runs", action="append", default=[])
    ap.add_argument("--ref-manifest", action="append", default=[]); ap.add_argument("--phase-summary", default=None); ap.add_argument("--ret-dir", action="append", default=[])
    ap.add_argument("--dispatched", default=None); ap.add_argument("--hold-file", default=None); ap.add_argument("--no-famcount", action="store_true")
    ap.add_argument("--expect", default=None, help="group counts for FAMCOUNT, e.g. \"L0=288,L2R=198,SHT=144\" (a count may list alternatives with |); default: the SPEC table of this file")
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
        print("HOLD written: %s (a DET, DETS, IDENT or smoke-arm SHTSCAN gate failed)" % a.hold_file)
    return 1 if "FAIL" in [l.split()[0] for l in txt.splitlines() if l] else 0


if __name__ == "__main__":
    sys.exit(main())
