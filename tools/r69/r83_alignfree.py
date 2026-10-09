#!/usr/bin/env python3
"""R83: alignment-free per-arm summary in each run's OWN FIRST-FRAME coordinates (successor of r82_alignfree.py, r81_alignfree.py, r80d_alignfree.py).
No global alignment, no ATE (rule 4.12 items 6 and 11).

    python3 tools/r69/r83_alignfree.py --root <dir, one folder per arm with pos_trace_all.csv.gz> --ref-root <R82 compact dir> [--ref-root <more>] \
        --phase-summary <R82 phase_summary.csv> [--phase-summary <R83 phase_summary.csv>] [--params bag_params.json] [--arm-table r83_manifest/arm_table.tsv] \
        [--state-ref-root <dir with <arm>/state_reference.txt>] [--prefix r83_] --out alignfree_summary.csv
    python3 tools/r69/r83_alignfree.py --selftest

WHAT WAS WRONG IN R82 (planner's error): r82_alignfree.py took only arm folders whose name starts with r81_ or r80d_ (so it wrote a header-only table for the R82 arms) and paired controls and healthy references
by R81 arm names (r81_rt_<...>, r80_<...>), which do not exist in R82 or R83. R83 pairs by an EXPLICIT control map and never looks for an R81 or R80 name; a control that cannot be found is a note in the row, never an error.

METRIC RULE (Bryce, 2026-10-05): initial yaw has no ground truth. Rotation and position deviation are measured against the FIRST FRAME BEFORE IMU PROPAGATION (the calibration state), and every run is compared in
ITS OWN first-frame coordinates: the first-frame position is subtracted and the first-frame heading is removed (rotation about world z by the inverse of the first-frame yaw; yaw = atan2(R10, R00) of the
quaternion). Heading only, not the full first-frame rotation: world z is gravity-up, so tilt is kept as it is. A run started with another heading or position gives the same numbers; two runs are compared point by
point on scan_id (no first-joined-scan offset removal). ONLY these first-frame metrics are written: the R82 columns dep_ctl_oldframe_mm and dep_ref_oldframe_mm (the R80D departure in the runs' own world
frames, which depends on the initial yaw) are not written in R83.

FIRST-FRAME SOURCE (column first_frame_src): `state_reference` = the file state_reference.txt of the arm (lines q_wxyz, p; written once before the first post_imu state, i.e. the state before IMU propagation);
`first_row` = FALLBACK when that file is not found next to the arm (or under --state-ref-root): the first row of pos_trace_all.csv.gz, the posterior of scan 1. Flagged in the row. A comparison across two runs whose
sources differ is flagged in `note`.

CONTROL MAP (R83). For each arm folder under --root whose name starts with --prefix (default r83_):
  ctl_arm  1. the `ctl_arm` column of --arm-table if it has one and the cell is not empty; else
           2. the `control` column (the manifest builder's same-batch control, e.g. r83_rt_ml_<bag>_l2_<q> for D1B and MQS arms); else
           3. the `ref` column (the arm this arm must reproduce: identity arms, determinism repeats); else
           4. the `r82_control` column (the R82 arm the manifest names); else
           5. the R82 control by rule: same bag, same Q, mode rt or dc as the arm, ds first, root voxel 0.425:  r82_<rt|dc>_ds_<bag>_first_<fq|mq>.
  ref_arm  the healthy reference: the `ref_arm` column of the table if present, else `r82_control`, else the R82 control by rule (5). dep_ref_mm and the flags are measured against it.
  Which arms: when the table has an `alignfree` column only the arms with alignfree = 1 get a row (the manifest names them; required_outputs.json counts them); --all-arms writes a row for every folder with the prefix.
  bag      the table's `bag` (else `seq`) column, else the first [a-z]+_NN token of the arm name.   Q: the last fq|mq token of the arm name (else the table's `q` column).   mode: the table's `mode` column
           (rt|dc), else the first _rt_ / _dc_ token of the name.
  An arm that is its own control gets no control (blank). A name the rule cannot parse gets a row with a note and empty control cells. Controls are looked for as <root>/<name>/pos_trace_all.csv.gz in --root and every
  --ref-root (compact/<name>/ layouts: pass the compact directory as --ref-root).

COLUMNS
  arm, fam (the token after rt_/dc_ of the name: ds, ml, sw, pc, d4, mqs, dst, id, ...), src (the rest of the name), bag
  onset_scan       onset_scan_used of the phase_summary row of the arm, else of its control, else of its healthy reference (onset_source says which; blank with a note when none has one)
  n_scans, n_joined: scans in the arm; scans joined (on scan_id, never on t_abs) to the control
  stat_drift_mm    max over stationary scans (scan_id < onset, or <= stat_last of --params) of |p'|, p' = position in the arm's first-frame coordinates, mm
  stat_settle_mm   |mean p' over the stationary scans|, mm
  dep_ctl_mm       max over joined scans of |p' - q'|, q' = the control in ITS OWN first-frame coordinates; mm.   dep_ctl_scan50: first scan_id whose departure from the control is >= 50 mm (blank if none)
  dep_ref_mm       the same departure from the healthy reference.   dep_flag: 1 when dep_ref_mm >= 500
  mode, ctl_arm, ref_arm, first_frame_src, ctl_first_frame_src, ref_first_frame_src, onset_source, stat_last
  stat_yaw_drift_mrad, stat_yaw_settle_mrad (scans 50..stat_last, blank if fewer than 6), stat_yaw_end_mrad
  excursion_mm     max over scans 1..10 of |p'|.   settled_offset_mm: |mean p' over scans 50..stat_last| (blank if fewer than 6 scans)
  gt_max_mm        from the phase_summary row of the arm (blank if the arm has no row)
  collapse_flag    1 when dep_ref_mm >= 500 AND gt_max_mm > 500 (rule 4.15c), 0 otherwise, blank when unknown
  note             warnings (missing control or reference, mixed first-frame sources, onset taken from a control, no onset)
An arm folder without pos_trace_all.csv.gz (summary-only, smoke without a trace, refuse) gets a note-only row; this is never an error. The exit code is 0 unless the tool itself crashes.
Planning agent's script. Run by me only on synthetic traces.
"""
import argparse, csv, gzip, math, os, re, sys
import numpy as np

BAG = r"[a-z]+_\d\d"
COLS = ["arm", "fam", "src", "bag", "onset_scan", "n_scans", "n_joined", "stat_drift_mm", "stat_settle_mm", "dep_ctl_mm", "dep_ctl_scan50", "dep_ref_mm", "dep_flag",
        "mode", "ctl_arm", "ref_arm", "first_frame_src", "ctl_first_frame_src", "ref_first_frame_src", "onset_source", "stat_last",
        "stat_yaw_drift_mrad", "stat_yaw_settle_mrad", "stat_yaw_end_mrad", "excursion_mm", "settled_offset_mm", "gt_max_mm", "collapse_flag", "note"]


# ----------------------------------------------------------------------------------------------------------------- geometry
def wrap(a):
    """wrap an angle (rad) to [-pi, pi)."""
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of_quat(q):
    """q (N,4) w,x,y,z -> yaw = atan2(R10, R00), rad."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


class Trace:
    """ids (scan_id), xyz (m, world), yaw (rad), and the first frame (p0, yaw0, src)."""
    def __init__(self, ids, xyz, yaw, ff_p=None, ff_yaw=None, src="first_row"):
        self.ids, self.xyz, self.yaw = ids, xyz, yaw
        if ff_p is None:
            ff_p, ff_yaw, src = xyz[0].copy(), float(yaw[0]), "first_row"
        self.p0, self.yaw0, self.src = np.asarray(ff_p, float), float(ff_yaw), src
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        d = xyz - self.p0
        self.pf = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)   # R_z(-yaw0) (p - p0), own first-frame coordinates, m
        self.dyaw = wrap(yaw - self.yaw0)                                                                  # yaw relative to the first-frame heading, rad


def read_state_reference(path):
    """-> (p (3,), yaw0 rad) from state_reference.txt (lines q_wxyz and p). Raises ValueError if either is missing."""
    d = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if t and t[0] in ("q_wxyz", "p"):
                d[t[0]] = [float(x) for x in t[1:]]
    if "q_wxyz" not in d or "p" not in d or len(d["q_wxyz"]) != 4 or len(d["p"]) != 3:
        raise ValueError("%s lacks q_wxyz / p" % path)
    return np.array(d["p"]), float(yaw_of_quat(np.array([d["q_wxyz"]]))[0])


def read_pos_trace(path, ref_path=None):
    """pos_trace_all.csv.gz (columns scan_id, px, py, pz, yaw_rad, optional qw qx qy qz) -> Trace. The heading uses the quaternion columns when present, else yaw_rad."""
    import pandas as pd
    D = pd.read_csv(path).drop_duplicates("scan_id").sort_values("scan_id")
    ids = D["scan_id"].to_numpy(np.int64)
    xyz = D[["px", "py", "pz"]].to_numpy(np.float64)
    if all(c in D.columns for c in ("qw", "qx", "qy", "qz")):
        yaw = yaw_of_quat(D[["qw", "qx", "qy", "qz"]].to_numpy(np.float64))
    else:
        yaw = D["yaw_rad"].to_numpy(np.float64)
    if ref_path and os.path.exists(ref_path):
        p0, y0 = read_state_reference(ref_path)
        return Trace(ids, xyz, yaw, p0, y0, "state_reference")
    return Trace(ids, xyz, yaw)


# ----------------------------------------------------------------------------------------------------------------- departures
def departure_ff(ta, tb, thr=50.0):
    """departure in first-frame coordinates: max | p'_a - p'_b | mm over the joined scans (no offset removal), first scan >= thr, n joined."""
    common = np.intersect1d(ta.ids, tb.ids)
    if len(common) < 1:
        return float("nan"), None, 0
    A = ta.pf[np.searchsorted(ta.ids, common)]; B = tb.pf[np.searchsorted(tb.ids, common)]
    d = 1000.0 * np.linalg.norm(A - B, axis=1)
    hit = np.nonzero(d >= thr)[0]
    return float(d.max()), (int(common[hit[0]]) if len(hit) else None), int(len(common))


def stationary_metrics(t, onset, stat_last=None):
    """-> dict of the stationary metrics of a Trace in its own first-frame coordinates. Stationary = scan_id < onset, or <= stat_last."""
    last = (int(stat_last) if stat_last is not None else int(onset) - 1)
    k = int(np.searchsorted(t.ids, last, side="right"))
    k = k if k >= 2 else min(2, len(t.ids))
    st, sy, sid = t.pf[:k], t.dyaw[:k], t.ids[:k]
    r = dict(stat_last=last,
             stat_drift_mm=1000.0 * float(np.linalg.norm(st, axis=1).max()),
             stat_settle_mm=1000.0 * float(np.linalg.norm(st.mean(axis=0))),
             stat_yaw_drift_mrad=1000.0 * float(np.abs(sy).max()),
             stat_yaw_end_mrad=1000.0 * float(sy[-1]))
    m = (sid >= 50)
    r["stat_yaw_settle_mrad"] = 1000.0 * float(sy[m].mean()) if m.sum() >= 6 else None
    r["settled_offset_mm"] = 1000.0 * float(np.linalg.norm(st[m].mean(axis=0))) if m.sum() >= 6 else None
    m10 = (t.ids >= 1) & (t.ids <= 10)
    r["excursion_mm"] = 1000.0 * float(np.linalg.norm(t.pf[m10], axis=1).max()) if m10.any() else None
    return r


# ----------------------------------------------------------------------------------------------------------------- names
def rule_control(bag, q, mode):
    """the R82 control by rule: same bag, same Q, rt (or dc for a dc arm), ds first, root 0.425 -> r82_<mode>_ds_<bag>_first_<q>."""
    return "r82_%s_ds_%s_first_%s" % (mode, bag, q) if (bag and q and mode in ("rt", "dc")) else None


def resolve(arm, table=None):
    """-> dict(mode, fam, src, bag, q, ctl, ref, note). Never raises, never returns None; an arm the rule cannot parse gets empty control cells and a note. No R80 / R81 arm name is used."""
    row = (table or {}).get(arm, {})
    mm = re.match(r"r\d+[a-z]?_(?:(rt|dc)_)?(.*)$", arm)
    nm_mode = mm.group(1) if mm and mm.group(1) else None
    if not nm_mode:
        m2 = re.search(r"(?:^|_)(rt|dc)(?=_)", arm)
        nm_mode = m2.group(1) if m2 else None
    mode = row.get("mode") if row.get("mode") in ("rt", "dc") else nm_mode
    rest = mm.group(2) if mm else arm
    if mm and mm.group(1) is None:
        rest = re.sub(r"^(?:rt|dc)_", "", rest)
    mb = re.search(BAG, arm)
    bag = (row.get("bag") or row.get("seq") or "").strip() or (mb.group(0) if mb else "")
    if bag and not re.fullmatch(BAG, bag):
        bag = mb.group(0) if mb else ""
    qs = re.findall(r"(?:^|_)(fq|mq)(?=_|$)", arm)
    q = qs[-1] if qs else ((row.get("q") or "").strip() or None)
    fam = rest.split("_")[0] if rest else ""
    notes = []
    rc = rule_control(bag, q, mode)
    cell = lambda k: (row.get(k) or "").strip()
    ctl = cell("ctl_arm") or cell("control") or cell("ref") or cell("r82_control") or rc
    ref = cell("ref_arm") or cell("r82_control") or rc
    if rc is None and not (cell("ctl_arm") or cell("control") or cell("ref") or cell("r82_control")):
        notes.append("name not understood by the control rule (bag %r, Q %r, mode %r) and no ctl_arm / ref in the table" % (bag, q, mode))
    if ctl == arm:
        ctl = None
    if ref == arm:
        ref = None
    return dict(mode=mode or "", fam=fam, src=rest, bag=bag, q=q, ctl=ctl, ref=ref, note="; ".join(notes))


class Finder:
    """locates <name>/pos_trace_all.csv.gz and <name>/state_reference.txt in the given roots; caches Traces."""
    def __init__(self, roots, state_roots=()):
        self.roots, self.state_roots, self.cache = list(roots), list(state_roots), {}

    def trace_path(self, name):
        for r in self.roots:
            p = os.path.join(r, name, "pos_trace_all.csv.gz")
            if os.path.exists(p):
                return p
        return None

    def ref_path(self, name):
        cands = []
        for r in self.roots + self.state_roots:
            cands += [os.path.join(r, name, "state_reference.txt"), os.path.join(r, name, "archive", "state_reference.txt"),
                      os.path.join(r, name, name, "state_reference.txt")]
        for p in cands:
            if os.path.exists(p):
                return p
        return None

    def have(self, name):
        return self.trace_path(name) is not None

    def get(self, name):
        if name not in self.cache:
            p = self.trace_path(name)
            self.cache[name] = read_pos_trace(p, self.ref_path(name)) if p else None
        return self.cache[name]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------------------------- summary
def summarize(arm, info, ta, ctl, ref, onset, onset_source, gt_max, stat_last=None):
    """ta, ctl, ref: Trace (ctl / ref may be None). onset may be None (stationary metrics blank). -> row dict."""
    notes = [info["note"]] if info.get("note") else []
    row = dict(arm=arm, fam=info["fam"], src=info["src"], bag=info["bag"], n_scans=int(len(ta.ids)), mode=info["mode"], ctl_arm=info["ctl"] or "", ref_arm=info["ref"] or "",
               first_frame_src=ta.src, onset_source=onset_source or "", gt_max_mm=("" if gt_max is None else round(gt_max, 3)), onset_scan="" if onset is None else int(onset))
    if onset is None and stat_last is None:
        notes.append("no onset_scan_used in the phase summaries for the arm, its control or its reference: stationary metrics not written")
        m10 = (ta.ids >= 1) & (ta.ids <= 10)
        row["excursion_mm"] = round(1000.0 * float(np.linalg.norm(ta.pf[m10], axis=1).max()), 3) if m10.any() else ""
    else:
        s = stationary_metrics(ta, int(onset) if onset is not None else stat_last + 1, stat_last)
        row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    for k in ("stat_yaw_settle_mrad", "settled_offset_mm", "excursion_mm"):
        if row.get(k) is None: row[k] = ""
    for tag, tr in (("ctl", ctl), ("ref", ref)):
        if tr is None:
            row.update({"dep_%s_mm" % tag: "", "%s_first_frame_src" % tag: ""})
            if info[tag]:
                notes.append("no %s trace (%s)" % (tag, info[tag]))
            if tag == "ctl": row["n_joined"], row["dep_ctl_scan50"] = "", ""
            continue
        d, sc, nj = departure_ff(ta, tr)
        row["dep_%s_mm" % tag] = round(d, 3); row["%s_first_frame_src" % tag] = tr.src
        if tag == "ctl": row["n_joined"], row["dep_ctl_scan50"] = nj, ("" if sc is None else sc)
        if tr.src != ta.src: notes.append("%s first-frame source differs (%s vs %s)" % (tag, tr.src, ta.src))
    if ta.src == "first_row":
        notes.append("first frame = posterior of scan 1 (no state_reference.txt)")
    dr = row.get("dep_ref_mm", "")
    row["dep_flag"] = "" if dr == "" else int(dr >= 500.0)
    row["collapse_flag"] = "" if (dr == "" or gt_max is None) else int(dr >= 500.0 and gt_max > 500.0)
    row["note"] = "; ".join(n for n in notes if n)
    return row


def load_phase(paths):
    out = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            out[r["arm"]] = r
    return out


def load_table(path):
    if not path:
        return {}
    return {r["arm"]: r for r in csv.DictReader(open(path), delimiter="\t")}


def load_table(path):
    if not path:
        return {}
    return {r["arm"]: r for r in csv.DictReader(open(path), delimiter="\t")}


def pick_onset(arm, info, phase):
    """onset_scan_used of the arm's own phase row, else of its control, else of its healthy reference."""
    for c, tag in ((arm, "self"), (info["ctl"], "ctl"), (info["ref"], "ref")):
        r = phase.get(c) if c else None
        if r and num(r.get("onset_scan_used")) is not None:
            return int(float(r["onset_scan_used"])), ("self" if tag == "self" else c)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--ref-root", action="append", default=[]); ap.add_argument("--phase-summary", action="append", default=[])
    ap.add_argument("--state-ref-root", action="append", default=[]); ap.add_argument("--params"); ap.add_argument("--arm-table"); ap.add_argument("--prefix", default="r83_")
    ap.add_argument("--out", default="alignfree_summary.csv"); ap.add_argument("--all-arms", action="store_true", help="ignore the table's alignfree column"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.root and a.phase_summary):
        sys.exit("--root and --phase-summary are required")
    import json
    phase = load_phase(a.phase_summary)
    params = json.load(open(a.params)) if a.params else {}
    table = load_table(a.arm_table)
    F = Finder([a.root] + a.ref_root, a.state_ref_root)
    arms = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)) and d.startswith(a.prefix))
    if table and not a.all_arms and any("alignfree" in r for r in table.values()):
        arms = [x for x in arms if (table.get(x, {}).get("alignfree") or "").strip() == "1"]
    rows, noted, nocontrol = [], [], []
    for arm in arms:
        info = resolve(arm, table)
        if not os.path.exists(os.path.join(a.root, arm, "pos_trace_all.csv.gz")):
            noted.append(arm)
            rows.append(dict(arm=arm, mode=info["mode"], ctl_arm=info["ctl"] or "", ref_arm=info["ref"] or "", note="no pos_trace_all.csv.gz (summary-only / smoke / refuse arm)"))
            continue
        ons, osrc = pick_onset(arm, info, phase)
        stat_last = None
        pj = params.get(info["bag"])
        if pj and "stat_last" in pj and (ons is None or int(pj.get("onset_scan", ons)) == ons):
            stat_last = int(pj["stat_last"])
        g = num(phase.get(arm, {}).get("gt_max_mm"))
        rows.append(summarize(arm, info, F.get(arm), F.get(info["ctl"]) if info["ctl"] else None, F.get(info["ref"]) if info["ref"] else None, ons, osrc, g, stat_last))
        if rows[-1].get("ctl_first_frame_src", "") == "" or rows[-1].get("ref_first_frame_src", "") == "":
            nocontrol.append(arm)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS, restval=""); w.writeheader(); w.writerows(rows)
    print("alignfree rows %d; first-frame source state_reference %d, first_row %d; note-only rows (no trace) %d; arms with a missing control or reference (a note in the row, not an error) %d %s"
          % (len(rows), sum(r.get("first_frame_src") == "state_reference" for r in rows), sum(r.get("first_frame_src") == "first_row" for r in rows), len(noted), len(nocontrol), nocontrol[:5]))
    return 0


# ----------------------------------------------------------------------------------------------------------------- selftest
def _write_trace(path, ids, xyz, yaw, tilt=None):
    import pandas as pd
    n = len(ids)
    half = yaw / 2.0
    q = np.stack([np.cos(half), np.zeros(n), np.zeros(n), np.sin(half)], axis=1)      # pure yaw quaternion
    D = pd.DataFrame({"scan_id": ids, "t_abs": 0.1 * ids, "t_rel_s": 0.1 * ids, "px": xyz[:, 0], "py": xyz[:, 1], "pz": xyz[:, 2], "yaw_rad": wrap(yaw),
                      "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2], "qz": q[:, 3]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    D.to_csv(path, index=False, compression="gzip")


def _write_ref(path, p, yaw):
    with open(path, "w") as f:
        f.write("schema state_reference_v1\nq_wxyz %.12g 0 0 %.12g\np %.12g %.12g %.12g\nv 0 0 0\nbag_start_abs 0\nP0_dim 1\nP0_row_major 1\n" % (math.cos(yaw / 2), math.sin(yaw / 2), p[0], p[1], p[2]))


def selftest():
    import tempfile, shutil
    fails = []; nck = [0]

    def chk(c, m):
        nck[0] += 1
        if not c: fails.append(m)
    n = 600; ids = np.arange(1, n + 1)
    base = np.zeros((n, 3)); base[300:, 0] = 0.01 * np.arange(n - 300)
    wob = base.copy(); wob[:, 1] += np.where(ids <= 100, 0.002 * np.sin(ids / 7.0), 0.0)          # 2 mm wobble while stationary
    yaw_true = np.where(ids <= 100, 0.003 * np.sin(ids / 11.0), 0.003 * np.sin(100 / 11.0))        # yaw drifts by up to 3 mrad, in first-frame terms
    Rz = lambda a: np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    def place(pos, yaw, y0, off):
        return (Rz(y0) @ pos.T).T + off, yaw + y0

    t = tempfile.mkdtemp()
    try:
        # (1) yaw/position invariance: arm and control are the same run started with other headings and positions
        wa, ya = place(wob, yaw_true, 2.9, np.array([5.0, -3.0, 0.1]))
        wc, yc = place(wob, yaw_true, -0.7, np.array([-40.0, 22.0, 0.0]))
        A, C = Trace(ids, wa, ya), Trace(ids, wc, yc)
        d, s, nj = departure_ff(A, C)
        chk(d < 1e-6 and s is None and nj == n, "ff departure of the same run started elsewhere must be 0: %r %r" % (d, s))
        # (2) a 200 mm step in the arm's own x at scan 401 is reported as 200 mm at scan 401
        step = wob.copy(); step[400:, 0] += 0.2
        wa2, ya2 = place(step, yaw_true, 2.9, np.array([5.0, -3.0, 0.1]))
        A2 = Trace(ids, wa2, ya2)
        d, s, nj = departure_ff(A2, C)
        chk(abs(d - 200.0) < 1e-6 and s == 401, "200 mm step: %r %r" % (d, s))
        keep = (ids < 10) | (ids >= 20)
        Ak = Trace(ids[keep], wa2[keep], ya2[keep], A2.p0, A2.yaw0, "state_reference")
        d2, _, nj2 = departure_ff(Ak, C)
        chk(nj2 == n - 10 and abs(d2 - 200.0) < 1e-6, "join on scan_id with missing scans: %r %r" % (d2, nj2))
        # (3) stationary metrics in first-frame coordinates
        s_ = stationary_metrics(A, 301)
        chk(1.0 < s_["stat_drift_mm"] < 3.0, "stat drift %r" % s_["stat_drift_mm"])
        chk(abs(s_["stat_yaw_drift_mrad"] - 1000 * np.abs(yaw_true[:300] - yaw_true[0]).max()) < 1e-6, "yaw drift %r" % s_["stat_yaw_drift_mrad"])
        chk(s_["excursion_mm"] is not None and s_["excursion_mm"] < 3.0 and s_["stat_yaw_settle_mrad"] is not None, "excursion / settle present")
        # (4) state_reference.txt overrides the first row
        ref_p = wa[0] + Rz(ya[0]) @ np.array([-0.004, 0.0, 0.0]); ref_y = ya[0] - 0.010
        ra = os.path.join(t, "ra"); _write_trace(os.path.join(ra, "x", "pos_trace_all.csv.gz"), ids, wa, ya)
        _write_ref(os.path.join(ra, "x", "state_reference.txt"), ref_p, ref_y)
        T6 = read_pos_trace(os.path.join(ra, "x", "pos_trace_all.csv.gz"), os.path.join(ra, "x", "state_reference.txt"))
        chk(T6.src == "state_reference" and abs(np.linalg.norm(T6.pf[0]) - 0.004) < 1e-9 and abs(T6.dyaw[0] - 0.010) < 1e-9, "state_reference first frame: %r" % T6.pf[0])
        # (5) the R83 control map: names, the table's ref column, explicit overrides, never an R81 name, never an exception
        r1 = resolve("r83_rt_ml_eee_01_l2_fq")
        chk(r1["ctl"] == "r82_rt_ds_eee_01_first_fq" and r1["ref"] == "r82_rt_ds_eee_01_first_fq" and r1["mode"] == "rt" and r1["fam"] == "ml" and r1["bag"] == "eee_01", "rt rule control: %r" % r1)
        r2 = resolve("r83_dc_sw_nya_02_off_v0p25_pfn3_mq")
        chk(r2["ctl"] == "r82_dc_ds_nya_02_first_mq" and r2["mode"] == "dc" and r2["fam"] == "sw", "dc rule control: %r" % r2)
        r3 = resolve("r83_rt_mqs_sbs_03_g0p3_a0p15_mq")
        chk(r3["ctl"] == "r82_rt_ds_sbs_03_first_mq" and r3["bag"] == "sbs_03", "mqs name (g0p3, a0p15 are not the Q token): %r" % r3)
        r4 = resolve("r83_rt_pc_nya_01_w20ms_fq", {"r83_rt_pc_nya_01_w20ms_fq": {"ref": "r83_rt_id_pc0_eee_01_fq", "mode": "rt", "bag": "nya_01"}})
        chk(r4["ctl"] == "r83_rt_id_pc0_eee_01_fq" and r4["ref"] == "r82_rt_ds_nya_01_first_fq", "table ref column is the control, the rule control is the healthy reference: %r" % r4)
        r5 = resolve("r83_rt_x_eee_01_fq", {"r83_rt_x_eee_01_fq": {"ctl_arm": "CTL", "ref_arm": "REF"}})
        chk(r5["ctl"] == "CTL" and r5["ref"] == "REF", "ctl_arm / ref_arm override: %r" % r5)
        r6 = resolve("r83_own_det_x_eee_01", {"r83_own_det_x_eee_01": {"mode": "rt", "bag": "eee_01", "ref": "r83_rt_ml_eee_01_l2_fq"}})
        chk(r6["ctl"] == "r83_rt_ml_eee_01_l2_fq" and r6["mode"] == "rt", "determinism repeat: the control is the ref column: %r" % r6)
        r8 = resolve("junk"); r9 = resolve("r83_dc_sm_pcw_60s")
        chk(r8["ctl"] is None and "not understood" in r8["note"] and r9["ctl"] is None and "not understood" in r9["note"], "unparseable names give a note, not an error: %r %r" % (r8, r9))
        chk(resolve("r82_rt_ds_eee_01_first_fq")["ctl"] is None, "an arm that is its own rule control gets no control")
        chk(not any(re.match(r"r8[01]d?_", str(x["ctl"]) + "") or re.match(r"r8[01]d?_", str(x["ref"]) + "") for x in (r1, r2, r3, r4, r6)), "no R80 / R81 name appears in any control")
        # (6) end to end with the CLI: R83 arm names, the R82 control found in a ref root, collapse flag via gt_max, summary-only arm, missing control = note
        root, ref82 = os.path.join(t, "ret"), os.path.join(t, "r82c")
        arm = "r83_rt_ml_eee_01_l2_fq"
        big = wob.copy(); big[450:, 0] += 1.0                                           # 1 m step from scan 451
        wb, yb = place(big, yaw_true, 2.9, np.array([5.0, 5.0, 0.0]))
        wr_, yr_ = place(wob, yaw_true, -1.1, np.array([9.0, 1.0, 0.0]))
        _write_trace(os.path.join(root, arm, "pos_trace_all.csv.gz"), ids, wb, yb)
        _write_trace(os.path.join(ref82, "r82_rt_ds_eee_01_first_fq", "pos_trace_all.csv.gz"), ids, wr_, yr_)
        os.makedirs(os.path.join(root, "r83_dc_sw_eee_01_off_v0p25_pfn3_fq"), exist_ok=True)   # no trace: note-only row
        os.makedirs(os.path.join(root, "r83_dc_refuse_zupt_removed"), exist_ok=True)
        os.makedirs(os.path.join(root, "r81_rt_old_eee_01_fq"), exist_ok=True)               # another round's folder: not taken (prefix)
        ps = os.path.join(t, "ps.csv")
        open(ps, "w").write("arm,onset_scan_used,gt_max_mm\nr82_rt_ds_eee_01_first_fq,301,86.6\n%s,301,1200.0\n" % arm)
        out = os.path.join(t, "out.csv")
        sys.argv = ["x", "--root", root, "--ref-root", ref82, "--phase-summary", ps, "--out", out]
        rc_ = main()
        R = list(csv.DictReader(open(out)))
        chk(rc_ == 0 and len(R) == 3 and not any(x["arm"].startswith("r81_") for x in R), "rows %d (the arm and two note-only rows; the r81_ folder is not taken)" % len(R))
        chk(all("no pos_trace" in x["note"] for x in R if x["arm"] != arm), "note-only rows")
        r = {x["arm"]: x for x in R}[arm]
        chk(r["mode"] == "rt" and r["ctl_arm"] == "r82_rt_ds_eee_01_first_fq" and r["first_frame_src"] == "first_row" and r["fam"] == "ml", "cli resolution %r" % r)
        chk(abs(float(r["dep_ctl_mm"]) - 1000.0) < 1e-3 and r["dep_ctl_scan50"] == "451" and r["dep_flag"] == "1" and r["collapse_flag"] == "1", "cli departure / collapse %r" % r)
        chk(r["onset_scan"] == "301" and r["onset_source"] == "self" and "first frame = posterior of scan 1" in r["note"], "onset / note %r" % r)
        chk(set(r) == set(COLS) and "dep_ctl_oldframe_mm" not in r and "dep_ref_oldframe_mm" not in r, "only first-frame columns are written: %r" % sorted(set(r) ^ set(COLS)))
        chk(abs(float(r["stat_yaw_drift_mrad"]) - 1000 * np.abs(yaw_true[:300] - yaw_true[0]).max()) < 1e-2, "cli yaw drift %r" % r["stat_yaw_drift_mrad"])
        open(ps, "w").write("arm,onset_scan_used,gt_max_mm\nr82_rt_ds_eee_01_first_fq,301,86.6\n%s,301,120.0\n" % arm)
        main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[arm]
        chk(r["dep_flag"] == "1" and r["collapse_flag"] == "0", "collapse needs gt_max > 500: %r" % r)
        # table ref column as the control, with a control that does not exist: a note, exit code 0
        open(os.path.join(t, "at.tsv"), "w").write("arm\tmode\tbag\tref\n%s\trt\teee_01\tr83_rt_id_pc0_eee_01_fq\n" % arm)
        sys.argv = ["x", "--root", root, "--ref-root", ref82, "--phase-summary", ps, "--arm-table", os.path.join(t, "at.tsv"), "--out", out]
        rc_ = main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[arm]
        chk(rc_ == 0 and r["ctl_arm"] == "r83_rt_id_pc0_eee_01_fq" and r["dep_ctl_mm"] == "" and "no ctl trace" in r["note"] and r["dep_ref_mm"] != "", "missing table control is a note, not a failure: %r %r" % (rc_, r["note"]))
        # the manifest's extra columns: control (same batch) before ref before r82_control; alignfree = 1 selects the arms
        rA = resolve("r83_rt_pc_eee_01_w20ms_fq", {"r83_rt_pc_eee_01_w20ms_fq": {"control": "r83_rt_ml_eee_01_l2_fq", "r82_control": "r82_rt_ds_eee_01_first_fq", "mode": "rt", "bag": "eee_01"}})
        chk(rA["ctl"] == "r83_rt_ml_eee_01_l2_fq" and rA["ref"] == "r82_rt_ds_eee_01_first_fq", "control column is the control, r82_control the healthy reference: %r" % rA)
        rB = resolve("r83_dc_sw_eee_01_off_v0p2_pfn3_fq", {"r83_dc_sw_eee_01_off_v0p2_pfn3_fq": {"r82_control": "r82_dc_ds_eee_01_first_fq", "mode": "dc", "bag": "eee_01"}})
        chk(rB["ctl"] == "r82_dc_ds_eee_01_first_fq", "r82_control column when nothing nearer is named: %r" % rB)
        os.makedirs(os.path.join(root, "r83_rt_other_eee_01_fq"), exist_ok=True); _write_trace(os.path.join(root, "r83_rt_other_eee_01_fq", "pos_trace_all.csv.gz"), ids, wob, yaw_true)
        open(os.path.join(t, "at2.tsv"), "w").write("arm\tmode\tbag\talignfree\n%s\trt\teee_01\t1\nr83_rt_other_eee_01_fq\trt\teee_01\t0\n" % arm)
        sys.argv = ["x", "--root", root, "--ref-root", ref82, "--phase-summary", ps, "--arm-table", os.path.join(t, "at2.tsv"), "--out", out]
        main()
        chk([x["arm"] for x in csv.DictReader(open(out))] == [arm], "alignfree column selects the arms: %r" % [x["arm"] for x in csv.DictReader(open(out))])
        sys.argv = ["x", "--root", root, "--ref-root", ref82, "--phase-summary", ps, "--arm-table", os.path.join(t, "at2.tsv"), "--all-arms", "--out", out]
        main()
        chk(len(list(csv.DictReader(open(out)))) == 4, "--all-arms writes every folder with the prefix")
        # no control and no onset anywhere: still a row, exit 0
        shutil.rmtree(os.path.join(ref82, "r82_rt_ds_eee_01_first_fq"))
        open(ps, "w").write("arm,onset_scan_used,gt_max_mm\n")
        rc_ = main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[arm]
        chk(rc_ == 0 and r["stat_drift_mm"] == "" and "no onset_scan_used" in r["note"], "no onset: stationary metrics blank, note, exit 0: %r" % r)
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r83_alignfree", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (nck[0], len(fails)),
          "" if fails else "(own first-frame coordinates: a run started with another heading and position departs 0 mm from itself; 200 mm step reported at scan 401; join on scan_id; state_reference.txt overrides the first row; "
          "control map from the table ref column or the R82 rule control; R83 names read; no R81 name needed; missing control / no onset are notes with exit 0; only first-frame columns)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
