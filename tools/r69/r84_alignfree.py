#!/usr/bin/env python3
"""R84: alignment-free per-arm summary in each run's OWN FIRST-FRAME coordinates (successor of r83_alignfree.py, r82_alignfree.py, r81_alignfree.py, r80d_alignfree.py).
No global alignment, no ATE (rule 4.12 items 6 and 11).

    python3 tools/r69/r84_alignfree.py --root <dir, one folder per arm with pos_trace_all.csv.gz> --ref-root <R82 compact dir> [--ref-root <more>] \
        --phase-summary <R82 phase_summary.csv> [--phase-summary <R84 phase_summary.csv>] [--params bag_params.json] [--arm-table r84_manifest/arm_table.tsv] \
        [--state-ref-root <dir with <arm>/state_reference.txt>] [--prefix r84_] --out alignfree_summary.csv
    python3 tools/r69/r84_alignfree.py --selftest

GROUP: the 216 comparison arms of R84 (group CMP, arm names r84_rt_cmp_<bag>_w<ms>ms_<vacc|none>_<fq|mq>) and nothing else. Selection: arm folders under --root whose name starts with --prefix and that are CMP arms
(the table's `group` column = CMP, else the token _cmp_ in the name); when the table has an `alignfree` column only the arms with alignfree = 1 (the manifest names them; required_outputs.json counts them).

PAIRING. Two controls per arm, both columns of the arm table:
  ctl_arm  the R82 control: the table's `control` column (the R82 rt first arm of the bag and Q), else `r82_control`, else the R82 control by rule  r82_<rt|dc>_ds_<bag>_first_<fq|mq>.
  sbc_arm  the same-batch control: the table's `same_batch_control` column (the w0 / var_acc arm of the same bag and Q); blank when the arm is its own same-batch control.
  ref_arm  the healthy reference for dep_flag / collapse_flag: `r82_control`, else the R82 control by rule (the same arm as ctl_arm for CMP arms).
A control that cannot be found is a note in the row, never an error. Controls are looked for as <root>/<name>/pos_trace_all.csv.gz in --root and every --ref-root (compact/<name>/ layouts: pass the compact directory as --ref-root).

METRIC RULE (Bryce, 2026-10-05): initial yaw has no ground truth. Rotation and position deviation are measured against the FIRST FRAME BEFORE IMU PROPAGATION (the calibration state), and every run is compared in
ITS OWN first-frame coordinates: the first-frame position is subtracted and the first-frame heading is removed (rotation about world z by the inverse of the first-frame yaw; yaw = atan2(R10, R00) of the
quaternion). Heading only, not the full first-frame rotation: world z is gravity-up, so tilt is kept as it is. A run started with another heading or position gives the same numbers; two runs are compared point by
point on scan_id (no first-joined-scan offset removal). ONLY these first-frame metrics are written: the R82 columns dep_ctl_oldframe_mm and dep_ref_oldframe_mm (the R80D departure in the runs' own world
frames, which depends on the initial yaw) are not written.

FIRST-FRAME SOURCE (column first_frame_src): `state_reference` = the file state_reference.txt of the arm (lines q_wxyz, p; written once before the first post_imu state, i.e. the state before IMU propagation);
`first_row` = FALLBACK when that file is not found next to the arm (or under --state-ref-root): the first row of pos_trace_all.csv.gz, the posterior of scan 1. Flagged in the row. A comparison across two runs whose
sources differ is flagged in `note`.

COLUMNS
  arm, fam (cmp), src (the rest of the name after r84_rt_), bag, sw_ms (sample window in ms: the table's `sw_ms`, else the w<N>ms token of the name), tbpn (time_based_process_noise: the table's `tbpn`, else the vacc|none token)
  onset_scan       onset_scan_used of the phase_summary row of the arm, else of its R82 control, else of its same-batch control, else of its healthy reference (onset_source says which; blank with a note when none has one)
  n_scans, n_joined: scans in the arm; scans joined (on scan_id, never on t_abs) to the control
  stat_drift_mm    max over stationary scans (scan_id < onset, or <= stat_last of --params) of |p'|, p' = position in the arm's first-frame coordinates, mm
  stat_settle_mm   |mean p' over the stationary scans|, mm
  dep_ctl_mm       max over joined scans of |p' - q'|, q' = the control in ITS OWN first-frame coordinates; mm.   dep_ctl_scan50: first scan_id whose departure from the control is >= 50 mm (blank if none)
  dep_ref_mm       the same departure from the healthy reference.   dep_flag: 1 when dep_ref_mm >= 500
  dep_sbc_mm       the same departure from the same-batch control (sbc_arm), mm.   dep_sbc_scan50: first scan_id whose departure from the same-batch control is >= 50 mm (blank if none)
  mode, ctl_arm, sbc_arm, ref_arm, first_frame_src, ctl_first_frame_src, sbc_first_frame_src, ref_first_frame_src, onset_source, stat_last
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
COLS = ["arm", "fam", "src", "bag", "sw_ms", "tbpn", "onset_scan", "n_scans", "n_joined", "stat_drift_mm", "stat_settle_mm", "dep_ctl_mm", "dep_ctl_scan50", "dep_sbc_mm", "dep_sbc_scan50", "dep_ref_mm", "dep_flag",
        "mode", "ctl_arm", "sbc_arm", "ref_arm", "first_frame_src", "ctl_first_frame_src", "sbc_first_frame_src", "ref_first_frame_src", "onset_source", "stat_last",
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


def is_cmp(arm, table=None):
    row = (table or {}).get(arm)
    if row is not None and (row.get("group") or "").strip():
        return row["group"].strip() == "CMP"
    return "_cmp_" in arm


def resolve(arm, table=None):
    """-> dict(mode, fam, src, bag, q, sw_ms, tbpn, ctl, sbc, ref, note). Never raises, never returns None; an arm the rule cannot parse gets empty control cells and a note."""
    row = (table or {}).get(arm, {})
    cell = lambda k: (row.get(k) or "").strip()
    mm = re.match(r"r\d+[a-z]?_(?:(rt|dc)_)?(.*)$", arm)
    nm_mode = mm.group(1) if mm and mm.group(1) else None
    mode = row.get("mode") if row.get("mode") in ("rt", "dc") else nm_mode
    rest = mm.group(2) if mm else arm
    mb = re.search(BAG, arm)
    bag = cell("bag") or cell("seq") or (mb.group(0) if mb else "")
    if bag and not re.fullmatch(BAG, bag):
        bag = mb.group(0) if mb else ""
    qs = re.findall(r"(?:^|_)(fq|mq)(?=_|$)", arm)
    q = qs[-1] if qs else (cell("q") or None)
    mw = re.search(r"_w(\d+)ms_(vacc|none)(?=_|$)", arm)
    sw_ms = cell("sw_ms") or (mw.group(1) if mw else "")
    tbpn = cell("tbpn") or ({"vacc": "var_acc", "none": "none"}[mw.group(2)] if mw else "")
    if sw_ms.endswith(".0"):
        sw_ms = sw_ms[:-2]
    notes = []
    rc = rule_control(bag, q, mode)
    ctl = cell("control") or cell("r82_control") or rc
    sbc = cell("same_batch_control") or None
    ref = cell("r82_control") or rc
    if rc is None and not (cell("control") or cell("r82_control")):
        notes.append("name not understood by the control rule (bag %r, Q %r, mode %r) and no control in the table" % (bag, q, mode))
    if ctl == arm:
        ctl = None
    if sbc == arm:
        sbc = None
    if ref == arm:
        ref = None
    return dict(mode=mode or "", fam="cmp" if is_cmp(arm, table) else rest.split("_")[0], src=rest, bag=bag, q=q, sw_ms=sw_ms, tbpn=tbpn, ctl=ctl, sbc=sbc, ref=ref, note="; ".join(notes))


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
def summarize(arm, info, ta, ctl, sbc, ref, onset, onset_source, gt_max, stat_last=None):
    """ta, ctl, sbc, ref: Trace (ctl / sbc / ref may be None). onset may be None (stationary metrics blank). -> row dict."""
    notes = [info["note"]] if info.get("note") else []
    row = dict(arm=arm, fam=info["fam"], src=info["src"], bag=info["bag"], sw_ms=info["sw_ms"], tbpn=info["tbpn"], n_scans=int(len(ta.ids)), mode=info["mode"],
               ctl_arm=info["ctl"] or "", sbc_arm=info["sbc"] or "", ref_arm=info["ref"] or "",
               first_frame_src=ta.src, onset_source=onset_source or "", gt_max_mm=("" if gt_max is None else round(gt_max, 3)), onset_scan="" if onset is None else int(onset))
    if onset is None and stat_last is None:
        notes.append("no onset_scan_used in the phase summaries for the arm, its controls or its reference: stationary metrics not written")
        m10 = (ta.ids >= 1) & (ta.ids <= 10)
        row["excursion_mm"] = round(1000.0 * float(np.linalg.norm(ta.pf[m10], axis=1).max()), 3) if m10.any() else ""
    else:
        s = stationary_metrics(ta, int(onset) if onset is not None else stat_last + 1, stat_last)
        row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    for k in ("stat_yaw_settle_mrad", "settled_offset_mm", "excursion_mm"):
        if row.get(k) is None: row[k] = ""
    for tag, tr in (("ctl", ctl), ("sbc", sbc), ("ref", ref)):
        if tr is None:
            row.update({"dep_%s_mm" % tag: "", "%s_first_frame_src" % tag: ""})
            if info[tag]:
                notes.append("no %s trace (%s)" % (tag, info[tag]))
            if tag == "ctl": row["n_joined"], row["dep_ctl_scan50"] = "", ""
            if tag == "sbc": row["dep_sbc_scan50"] = ""
            continue
        d, sc, nj = departure_ff(ta, tr)
        row["dep_%s_mm" % tag] = round(d, 3); row["%s_first_frame_src" % tag] = tr.src
        if tag == "ctl": row["n_joined"], row["dep_ctl_scan50"] = nj, ("" if sc is None else sc)
        if tag == "sbc": row["dep_sbc_scan50"] = "" if sc is None else sc
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


def pick_onset(arm, info, phase):
    """onset_scan_used of the arm's own phase row, else of its R82 control, else of its same-batch control, else of its healthy reference."""
    for c, tag in ((arm, "self"), (info["ctl"], "ctl"), (info["sbc"], "sbc"), (info["ref"], "ref")):
        r = phase.get(c) if c else None
        if r and num(r.get("onset_scan_used")) is not None:
            return int(float(r["onset_scan_used"])), ("self" if tag == "self" else c)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--ref-root", action="append", default=[]); ap.add_argument("--phase-summary", action="append", default=[])
    ap.add_argument("--state-ref-root", action="append", default=[]); ap.add_argument("--params"); ap.add_argument("--arm-table"); ap.add_argument("--prefix", default="r84_")
    ap.add_argument("--out", default="alignfree_summary.csv"); ap.add_argument("--selftest", action="store_true")
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
    arms = [x for x in arms if is_cmp(x, table)]                                    # group CMP only
    if table and any("alignfree" in r for r in table.values()):
        arms = [x for x in arms if (table.get(x, {}).get("alignfree") or "").strip() in ("1", "1.0")]
    rows, noted, nocontrol = [], [], []
    for arm in arms:
        info = resolve(arm, table)
        if not os.path.exists(os.path.join(a.root, arm, "pos_trace_all.csv.gz")):
            noted.append(arm)
            rows.append(dict(arm=arm, fam=info["fam"], bag=info["bag"], sw_ms=info["sw_ms"], tbpn=info["tbpn"], mode=info["mode"], ctl_arm=info["ctl"] or "", sbc_arm=info["sbc"] or "", ref_arm=info["ref"] or "", note="no pos_trace_all.csv.gz (summary-only / smoke / refuse arm)"))
            continue
        ons, osrc = pick_onset(arm, info, phase)
        stat_last = None
        pj = params.get(info["bag"])
        if pj and "stat_last" in pj and (ons is None or int(pj.get("onset_scan", ons)) == ons):
            stat_last = int(pj["stat_last"])
        g = num(phase.get(arm, {}).get("gt_max_mm"))
        rows.append(summarize(arm, info, F.get(arm), F.get(info["ctl"]) if info["ctl"] else None, F.get(info["sbc"]) if info["sbc"] else None, F.get(info["ref"]) if info["ref"] else None, ons, osrc, g, stat_last))
        if rows[-1].get("ctl_first_frame_src", "") == "" or rows[-1].get("ref_first_frame_src", "") == "" or (info["sbc"] and rows[-1].get("sbc_first_frame_src", "") == ""):
            nocontrol.append(arm)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS, restval=""); w.writeheader(); w.writerows(rows)
    print("alignfree rows %d; first-frame source state_reference %d, first_row %d; note-only rows (no trace) %d; arms with a missing control, same-batch control or reference (a note in the row, not an error) %d %s"
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
        # (5) the R84 control map: R82 control from the table's control column, same-batch control, rule fallback, never an exception
        C0 = "r84_rt_cmp_eee_01_w0ms_vacc_fq"; C5 = "r84_rt_cmp_eee_01_w5ms_vacc_fq"; R82 = "r82_rt_ds_eee_01_first_fq"
        tb = {C5: {"group": "CMP", "mode": "rt", "bag": "eee_01", "control": R82, "r82_control": R82, "same_batch_control": C0, "sw_ms": "5.0", "tbpn": "var_acc", "alignfree": "1"},
              C0: {"group": "CMP", "mode": "rt", "bag": "eee_01", "control": R82, "r82_control": R82, "same_batch_control": C0, "sw_ms": "0.0", "tbpn": "var_acc", "alignfree": "1"}}
        r1 = resolve(C5, tb)
        chk(r1["ctl"] == R82 and r1["sbc"] == C0 and r1["ref"] == R82 and r1["mode"] == "rt" and r1["fam"] == "cmp" and r1["bag"] == "eee_01" and r1["sw_ms"] == "5" and r1["tbpn"] == "var_acc", "CMP resolution from the table: %r" % r1)
        r0 = resolve(C0, tb)
        chk(r0["sbc"] is None and r0["ctl"] == R82 and r0["sw_ms"] == "0", "the w0 arm is its own same-batch control (blank): %r" % r0)
        r2 = resolve("r84_rt_cmp_nya_02_w20ms_none_mq")
        chk(r2["ctl"] == "r82_rt_ds_nya_02_first_mq" and r2["sbc"] is None and r2["sw_ms"] == "20" and r2["tbpn"] == "none" and r2["fam"] == "cmp" and r2["bag"] == "nya_02", "name only: rule control, window and tbpn from the name: %r" % r2)
        r3 = resolve(C5, {C5: {"control": "CTL", "r82_control": "REF", "same_batch_control": "SBC", "group": "CMP"}})
        chk(r3["ctl"] == "CTL" and r3["sbc"] == "SBC" and r3["ref"] == "REF", "control / same_batch_control / r82_control columns: %r" % r3)
        r8 = resolve("junk")
        chk(r8["ctl"] is None and "not understood" in r8["note"], "unparseable names give a note, not an error: %r" % r8)
        chk(resolve("r82_rt_ds_eee_01_first_fq")["ctl"] is None, "an arm that is its own rule control gets no control")
        chk(is_cmp(C5, tb) and not is_cmp("r84_rt_t3_eee_01_pca_off_v0p3_fq", {"r84_rt_t3_eee_01_pca_off_v0p3_fq": {"group": "T3"}}) and not is_cmp("r84_rt_t3_eee_01_pca_off_v0p3_fq") and is_cmp(C5), "CMP selection by group column or by name")
        # (6) end to end with the CLI: the R82 control in a ref root, the same-batch control in --root, collapse flag via gt_max, summary-only arm, missing control = note
        root, ref82 = os.path.join(t, "ret"), os.path.join(t, "r82c")
        big = wob.copy(); big[450:, 0] += 1.0                                           # 1 m step from scan 451
        wb, yb = place(big, yaw_true, 2.9, np.array([5.0, 5.0, 0.0]))
        wr_, yr_ = place(wob, yaw_true, -1.1, np.array([9.0, 1.0, 0.0]))
        w0_, y0_ = place(wob, yaw_true, 0.4, np.array([1.0, 1.0, 0.0]))
        _write_trace(os.path.join(root, C5, "pos_trace_all.csv.gz"), ids, wb, yb)
        _write_trace(os.path.join(root, C0, "pos_trace_all.csv.gz"), ids, w0_, y0_)
        _write_trace(os.path.join(ref82, R82, "pos_trace_all.csv.gz"), ids, wr_, yr_)
        os.makedirs(os.path.join(root, "r84_rt_cmp_eee_01_w10ms_none_fq"), exist_ok=True)   # no trace: note-only row
        os.makedirs(os.path.join(root, "r84_dc_refuse_sample_window_neg"), exist_ok=True)   # not a CMP arm: not taken
        os.makedirs(os.path.join(root, "r83_rt_old_eee_01_fq"), exist_ok=True)               # another round's folder: not taken (prefix)
        ps = os.path.join(t, "ps.csv")
        open(ps, "w").write("arm,onset_scan_used,gt_max_mm\n%s,301,86.6\n%s,301,1200.0\n" % (R82, C5))
        out = os.path.join(t, "out.csv")
        sys.argv = ["x", "--root", root, "--ref-root", ref82, "--phase-summary", ps, "--out", out]
        rc_ = main()
        R = list(csv.DictReader(open(out)))
        chk(rc_ == 0 and len(R) == 3 and not any(x["arm"].startswith(("r83_", "r84_dc_")) for x in R), "rows %d (two CMP arms with traces and one note-only CMP row; the refuse folder and the r83_ folder are not taken)" % len(R))
        chk(all("no pos_trace" in x["note"] for x in R if x["arm"] == "r84_rt_cmp_eee_01_w10ms_none_fq"), "note-only row")
        r = {x["arm"]: x for x in R}[C5]
        chk(r["mode"] == "rt" and r["ctl_arm"] == R82 and r["first_frame_src"] == "first_row" and r["fam"] == "cmp" and r["sw_ms"] == "5" and r["tbpn"] == "var_acc", "cli resolution %r" % r)
        chk(abs(float(r["dep_ctl_mm"]) - 1000.0) < 1e-3 and r["dep_ctl_scan50"] == "451" and r["dep_flag"] == "1" and r["collapse_flag"] == "1", "cli departure / collapse %r" % r)
        chk(r["sbc_arm"] == "" and r["dep_sbc_mm"] == "", "without the table the same-batch control is blank: %r" % r)
        chk(r["onset_scan"] == "301" and r["onset_source"] == "self" and "first frame = posterior of scan 1" in r["note"], "onset / note %r" % r)
        chk(set(r) == set(COLS) and "dep_ctl_oldframe_mm" not in r and "dep_ref_oldframe_mm" not in r, "only first-frame columns are written: %r" % sorted(set(r) ^ set(COLS)))
        chk(abs(float(r["stat_yaw_drift_mrad"]) - 1000 * np.abs(yaw_true[:300] - yaw_true[0]).max()) < 1e-2, "cli yaw drift %r" % r["stat_yaw_drift_mrad"])
        # with the arm table: same-batch control departure, alignfree column selects the arms, group column keeps CMP only
        cols = ["arm", "group", "mode", "bag", "control", "r82_control", "same_batch_control", "sw_ms", "tbpn", "alignfree"]
        with open(os.path.join(t, "at.tsv"), "w") as f:
            f.write("\t".join(cols) + "\n")
            for a_, v in tb.items():
                f.write("\t".join([a_] + [v.get(c_, "") for c_ in cols[1:]]) + "\n")
            f.write("\t".join(["r84_rt_cmp_eee_01_w10ms_none_fq", "CMP", "rt", "eee_01", R82, R82, C0, "10.0", "none", "0"]) + "\n")
            f.write("\t".join(["r84_dc_refuse_sample_window_neg", "OWN", "dc", "eee_01", "", "", "", "", "", "0"]) + "\n")
        sys.argv = ["x", "--root", root, "--ref-root", ref82, "--phase-summary", ps, "--arm-table", os.path.join(t, "at.tsv"), "--out", out]
        rc_ = main()
        R2 = {x["arm"]: x for x in csv.DictReader(open(out))}
        chk(rc_ == 0 and sorted(R2) == sorted([C0, C5]), "alignfree = 1 selects the arms: %r" % sorted(R2))
        r = R2[C5]
        chk(r["sbc_arm"] == C0 and abs(float(r["dep_sbc_mm"]) - 1000.0) < 1e-3 and r["dep_sbc_scan50"] == "451" and r["sbc_first_frame_src"] == "first_row", "same-batch control departure: %r" % r)
        chk(R2[C0]["sbc_arm"] == "" and R2[C0]["dep_sbc_mm"] == "" and R2[C0]["dep_ctl_mm"] != "" and "no sbc" not in R2[C0]["note"], "the w0 arm has no same-batch control and no note about it: %r" % R2[C0])
        # a same-batch control without a trace is a note, exit 0
        shutil.rmtree(os.path.join(root, C0))
        rc_ = main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[C5]
        chk(rc_ == 0 and r["dep_sbc_mm"] == "" and "no sbc trace" in r["note"] and r["dep_ctl_mm"] != "", "missing same-batch control is a note, not a failure: %r" % r["note"])
        # onset from the same-batch control when the arm and the R82 control have none
        open(ps, "w").write("arm,onset_scan_used,gt_max_mm\n%s,350,5.0\n" % C0)
        main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[C5]
        chk(r["onset_scan"] == "350" and r["onset_source"] == C0, "onset taken from the same-batch control: %r" % r)
        # no control and no onset anywhere: still a row, exit 0
        shutil.rmtree(os.path.join(ref82, R82))
        open(ps, "w").write("arm,onset_scan_used,gt_max_mm\n")
        rc_ = main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[C5]
        chk(rc_ == 0 and r["stat_drift_mm"] == "" and "no onset_scan_used" in r["note"], "no onset: stationary metrics blank, note, exit 0: %r" % r)
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r84_alignfree", ("FAIL " + "; ".join(fails) + " | ") if fails else "PASS", "checks: %d, failures: %d" % (nck[0], len(fails)),
          "" if fails else "(own first-frame coordinates: a run started with another heading and position departs 0 mm from itself; 200 mm step reported at scan 401; join on scan_id; state_reference.txt overrides the first row; "
          "CMP only; R82 control from the table control column (rule fallback), same-batch control departure; missing control / no onset are notes with exit 0; only first-frame columns)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
