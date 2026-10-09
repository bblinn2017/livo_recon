#!/usr/bin/env python3
"""R81: alignment-free per-arm summary in each run's OWN FIRST-FRAME coordinates (successor of r80d_alignfree.py).
No global alignment, no ATE (rule 4.12 items 6 and 11).

    python3 tools/r69/r81_alignfree.py --root <dir, one folder per arm with pos_trace_all.csv.gz> --ref-root <R80 compact dir> [--ref-root <more>] \
        --phase-summary <R80 phase_summary.csv> [--phase-summary <R81 phase_summary.csv>] [--params bag_params.json] [--arm-table arm_table.tsv] \
        [--state-ref-root <dir with <arm>/state_reference.txt>] --out alignfree_summary.csv
    python3 tools/r69/r81_alignfree.py --selftest

METRIC RULE (Bryce, 2026-10-05): initial yaw has no ground truth. Rotation and position deviation are measured against the FIRST FRAME BEFORE IMU PROPAGATION (the
calibration state), and every run is compared in ITS OWN first-frame coordinates: the first-frame position is subtracted and the first-frame heading is removed
(rotation about world z by the inverse of the first-frame yaw; yaw = atan2(R10, R00) of the quaternion). Heading only, not the full first-frame rotation: world z is gravity-up
(R80 state), so tilt is kept as it is. Consequences: a run started with another heading or position gives the same numbers; two runs are compared point by point (no
first-joined-scan offset removal, because the origin is the calibration state of each run).

FIRST-FRAME SOURCE (per run, written to the column first_frame_src):
  state_reference : the file state_reference.txt of the arm (lines `q_wxyz`, `p`, `v`, `bag_start_abs`, `P0_*`). It is written ONCE, by LioProcBase::ensureStationaryReference()
                    (src/processing/lio_base.cpp, called from lio_iekf.cpp 857 / lio_coupled_process.cpp 78 right BEFORE the first logStateTrace("post_imu")), from
                    MeasureGroup rot_before_imu / pos_before_imu / vel_before_imu of the first frame with a non-empty voxel map: the state before IMU propagation.
                    pos_trace_all.csv.gz holds the post_lio row only (state_trace.csv phase post_lio, via r74_frame_errors.py), so it does NOT hold that state.
  first_row       : FALLBACK when state_reference.txt is not found next to the arm (or under --state-ref-root): the first row of pos_trace_all.csv.gz, i.e. the POSTERIOR of
                    scan 1 (after IMU propagation and the first LiDAR update). It differs from the calibration state by the first-scan propagation and update. Flagged in the
                    row; with this source stat_drift_mm and stat_settle_mm are exactly the R80D definitions (|p - p(scan 1)|).
A comparison across two runs whose sources differ is flagged in `note` (ctl_first_frame_src / ref_first_frame_src are written).

COLUMNS (the first 13 are the columns of r80d_alignfree.py; their definitions are those of R80D except where marked [ff] = now measured from the first frame, own coordinates)
  arm, fam, src, bag
  onset_scan       : onset_scan_used of the phase_summary row of the arm, else of its control (onset_source says which)
  n_scans, n_joined: scans in the arm; scans joined (on scan_id, never on t_abs) to the control
  stat_drift_mm    [ff] max over stationary scans (scan_id < onset, or <= stat_last of --params) of |p'|, p' = position in the arm's first-frame coordinates, mm
  stat_settle_mm   [ff] |mean p' over the stationary scans|, mm
  dep_ctl_mm       [ff] max over joined scans of |p' - q'|, q' = the control arm in ITS OWN first-frame coordinates; mm
  dep_ctl_scan50   first scan_id whose departure from the control is >= 50 mm (blank if none)
  dep_ref_mm       [ff] the same departure from the healthy reference arm of the same mode
  dep_flag         1 when dep_ref_mm >= 500
NEW COLUMNS
  mode             dc (overlap_policy discard) | rt (retime_trim) | legacy (R80D names)
  ctl_arm, ref_arm the control (same data, the dc control = the R80 arm, or the matching plain control of the mode) and the healthy reference used
  first_frame_src, ctl_first_frame_src, ref_first_frame_src
  stat_last        last stationary scan used
  stat_yaw_drift_mrad   max over stationary scans of |wrap(yaw - yaw_first_frame)|, mrad
  stat_yaw_settle_mrad  mean of wrap(yaw - yaw_first_frame) over scans 50..stat_last (signed, mrad; blank if fewer than 6 scans)
  stat_yaw_end_mrad     the same difference at the last stationary scan (signed)
  excursion_mm     [ff] max over scans 1..10 of |p'| (r69_reduce excursion, against the first frame)
  settled_offset_mm [ff] |mean p' over scans 50..stat_last| (r69_reduce settled offset, against the first frame; blank if fewer than 6 scans)
  dep_ctl_oldframe_mm, dep_ref_oldframe_mm : the R80D departure (position only, as the runs wrote it, start offset removed at the first joined scan, no rotation). These reproduce the R80D
                    dep_ctl_mm / dep_ref_mm exactly (selftest and the R80D return). Do not rank on them: they depend on the initial yaw.
  gt_max_mm        from the phase_summary row of the arm (largest ground-truth error; blank if the arm has no row)
  collapse_flag    1 when dep_ref_mm >= 500 AND gt_max_mm > 500 (rule 4.15c), 0 otherwise, blank when gt_max_mm is unknown
  note             warnings (missing control, mixed first-frame sources, onset taken from a control)

ARM RESOLUTION (names; an --arm-table with columns arm and ctl_arm / ref_arm overrides every rule)
  r81_<mode>_<rest>  mode dc|rt, <rest> = the rest of the id, bag = [a-z]+_NN inside it, Q = the last fq|mq token. With the R81 arm_table.tsv (columns src_arm, group; use --arm-table):
    group RR or SWT (rerun of an R80 arm): ctl = src_arm (the R80 dc control; shows the effect of retime_trim), ref = r81_rt_<refsrc> (rt) or r80_<refsrc> (dc)
    other groups (new tasks): dc -> ctl = src_arm (R80 baseline); rt -> ctl = r81_rt_<src_arm without r80_> (the rt rerun of the same baseline) when its trace exists, else src_arm
    <refsrc> = mg_<bag>_fq for a b3 src, else b3_<bag>_mq (as R80D, from the src: BASE3 fq = r80_mg_<bag>_fq, BASE3 mq = r80_b3_<bag>_mq).
    Without a table row: rt with an existing r80_<rest> -> RR rule; else BASE3 of the Q (mg_<bag>_fq | b3_<bag>_mq), dc as r80_..., rt as r81_rt_....
  r80d_*  legacy names of R80D (src_of of r80d_alignfree.py): ctl = r80_<src>, ref = r80_<refsrc>.
Controls are looked for as <root>/<name>/pos_trace_all.csv.gz in --root and every --ref-root.
Arms whose id contains _sw_, _id_, _smoke_, _sm_, _refuse_ or _det (summary-only, identity, smoke, refuse and determinism arms) and have no trace get a note-only row (the generator's gate counts one row per arm) and no error.
Planning agent's script. Run by me only on synthetic traces (and, for the reproduction check, read-only on the returned R80D files).
"""
import argparse, csv, gzip, math, os, re, sys
import numpy as np

BAG = r"[a-z]+_\d\d"
SKIP_RE = re.compile(r"_sw_|_id_|_smoke_|_sm_|_refuse_|_det|_own_det")
COLS = ["arm", "fam", "src", "bag", "onset_scan", "n_scans", "n_joined", "stat_drift_mm", "stat_settle_mm", "dep_ctl_mm", "dep_ctl_scan50", "dep_ref_mm", "dep_flag",
        "mode", "ctl_arm", "ref_arm", "first_frame_src", "ctl_first_frame_src", "ref_first_frame_src", "onset_source", "stat_last",
        "stat_yaw_drift_mrad", "stat_yaw_settle_mrad", "stat_yaw_end_mrad", "excursion_mm", "settled_offset_mm",
        "dep_ctl_oldframe_mm", "dep_ref_oldframe_mm", "gt_max_mm", "collapse_flag", "note"]


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
def departure_old(ia, pa, ib, pb):
    """R80D definition (identical): max | (p - p(first joined)) - (q - q(first joined)) | mm, first scan >= 50 mm, n joined."""
    common = np.intersect1d(ia, ib)
    if len(common) < 2:
        return float("nan"), None, int(len(common))
    A = pa[np.searchsorted(ia, common)]; B = pb[np.searchsorted(ib, common)]
    d = 1000.0 * np.linalg.norm((A - A[0]) - (B - B[0]), axis=1)
    hit = np.nonzero(d >= 50.0)[0]
    return float(d.max()), (int(common[hit[0]]) if len(hit) else None), int(len(common))


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
def ref_src(src, bag=None):
    bag = bag or re.search(BAG, src).group(0)
    return "mg_%s_fq" % bag if src.startswith("b3_") else "b3_%s_mq" % bag


def legacy_src_of(arm):
    m = re.match(r"r80d_rs_(.+)$", arm)
    if m: return "rs", m.group(1)
    m = re.match(r"r80d_off_(mgfq|b3mq)_(%s)_o(\d{3})$" % BAG, arm)
    if m: return "off", ("mg_%s_fq" if m.group(1) == "mgfq" else "b3_%s_mq") % m.group(2)
    m = re.match(r"r80d_e3_(?:rtt|rt)_(.+)$", arm)
    if m: return "e3", m.group(1)
    return None, None


def base3(bag, q):
    """the R80 same-bag baseline of a Q: fixed-Q = mg_<bag>_fq (group MG), motion-Q = b3_<bag>_mq (group B3)."""
    return "mg_%s_fq" % bag if q == "fq" else "b3_%s_mq" % bag


def resolve(arm, table=None, have=None):
    """-> dict(mode, fam, src, bag, ctl, ref) or None when the name is not understood. `have(name)` -> bool tells whether a control trace exists.
    With an arm table row: src_arm (the R80 source arm) and group (RR / SWT = rerun of that R80 arm) are used (the R81 arm_table.tsv of make_r81_manifest.py has both);
      group RR / SWT   : ctl = src_arm (the dc control; shows the effect of retime_trim)
      other groups     : dc -> ctl = src_arm; rt -> ctl = r81_rt_<src_arm without r80_> (the rt rerun of the same baseline) when it exists, else src_arm
    Without a table row, by name (see the module docstring). ref (the healthy reference) follows the R80D rule on the src."""
    table = table or {}
    row = table.get(arm, {})
    fam, src = legacy_src_of(arm)
    if fam is not None:
        bag = re.search(BAG, src).group(0)
        out = dict(mode="legacy", fam=fam, src=src, bag=bag, ctl="r80_" + src, ref="r80_" + ref_src(src))
    else:
        m = re.match(r"r81_(dc|rt)_(.+)$", arm)
        if m:
            mode, rest = m.group(1), m.group(2)
        elif row.get("mode") in ("dc", "rt") and arm.startswith("r81_"):      # e.g. the determinism arm r81_own_det_x_<bag> (mode from the table)
            mode, rest = row["mode"], arm[len("r81_"):]
        else:
            return None
        mb = re.search(BAG, rest)
        bag = mb.group(0) if mb else (row.get("bag") or row.get("seq") or "")      # ids without a bag token (r81_dc_pca_fq, r81_dc_f2b_pfn3_fq): the table's bag / seq column
        if not re.fullmatch(BAG, bag):
            return None
        q = re.findall(r"(?:^|_)(fq|mq)(?=_|$)", rest)
        q = q[-1] if q else None
        fam = rest.split("_")[0]
        s80 = row.get("src_arm") if (row.get("src_arm") or "").startswith("r80_") else None
        basesrc = s80[4:] if s80 else (rest if re.match(r"(b3|mg)_", rest) else (base3(bag, q) if q else rest))
        rr = (row.get("group") in ("RR", "SWT")) if row.get("group") else bool(have is None or have("r80_" + rest))
        if rr:
            ctl = s80 or ("r80_" + rest)
        elif mode == "dc":
            ctl = s80 or ("r80_" + base3(bag, q) if q else None)
        else:
            rtb = "r81_rt_" + basesrc
            ctl = rtb if (have is None or have(rtb)) else (s80 or "r80_" + basesrc)
        ref = ("r81_rt_" if mode == "rt" else "r80_") + ref_src(basesrc, bag)
        out = dict(mode=mode, fam=fam, src=rest, bag=bag, ctl=ctl, ref=ref)
    if out["ctl"] == arm: out["ctl"] = None          # never a control of itself
    if row.get("arm_kind") in ("id", "det_fb") and row.get("ref"): out["ctl"] = row["ref"]   # identity / determinism arms: the arm they must reproduce
    if row.get("ctl_arm"): out["ctl"] = row["ctl_arm"]
    if row.get("ref_arm"): out["ref"] = row["ref_arm"]
    return out


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


# ----------------------------------------------------------------------------------------------------------------- summary
def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def summarize(arm, info, ta, ctl, ref, onset, onset_source, gt_max, stat_last=None):
    """ta, ctl, ref: Trace (ctl / ref may be None). -> row dict."""
    ons = int(onset)
    s = stationary_metrics(ta, ons, stat_last)
    notes = []
    row = dict(arm=arm, fam=info["fam"], src=info["src"], bag=info["bag"], onset_scan=ons, n_scans=int(len(ta.ids)), mode=info["mode"], ctl_arm=info["ctl"] or "", ref_arm=info["ref"] or "",
               first_frame_src=ta.src, onset_source=onset_source, gt_max_mm=("" if gt_max is None else round(gt_max, 3)))
    row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    for k in ("stat_yaw_settle_mrad", "settled_offset_mm", "excursion_mm"):
        if row.get(k) is None: row[k] = ""
    for tag, tr in (("ctl", ctl), ("ref", ref)):
        if tr is None:
            row.update({"dep_%s_mm" % tag: "", "dep_%s_oldframe_mm" % tag: "", "%s_first_frame_src" % tag: ""})
            notes.append("no %s trace (%s)" % (tag, info["ctl" if tag == "ctl" else "ref"]))
            if tag == "ctl": row["n_joined"], row["dep_ctl_scan50"] = "", ""
            continue
        d, sc, nj = departure_ff(ta, tr)
        do, _, _ = departure_old(ta.ids, ta.xyz, tr.ids, tr.xyz)
        row["dep_%s_mm" % tag] = round(d, 3); row["dep_%s_oldframe_mm" % tag] = round(do, 3); row["%s_first_frame_src" % tag] = tr.src
        if tag == "ctl": row["n_joined"], row["dep_ctl_scan50"] = nj, ("" if sc is None else sc)
        if tr.src != ta.src: notes.append("%s first-frame source differs (%s vs %s)" % (tag, tr.src, ta.src))
    if ta.src == "first_row":
        notes.append("first frame = posterior of scan 1 (no state_reference.txt)")
    dr = row["dep_ref_mm"]
    row["dep_flag"] = "" if dr == "" else int(dr >= 500.0)
    row["collapse_flag"] = "" if (dr == "" or gt_max is None) else int(dr >= 500.0 and gt_max > 500.0)
    row["note"] = "; ".join(notes)
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
    cands = [arm, "r80_" + info["src"], info["ctl"] or "", "r80_b3_%s_mq" % info["bag"], "r80_mg_%s_fq" % info["bag"]]
    for c in cands:
        r = phase.get(c)
        if r and num(r.get("onset_scan_used")) is not None:
            return int(float(r["onset_scan_used"])), ("self" if c == arm else c)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--ref-root", action="append", default=[]); ap.add_argument("--phase-summary", action="append", default=[])
    ap.add_argument("--state-ref-root", action="append", default=[]); ap.add_argument("--params"); ap.add_argument("--arm-table")
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
    arms = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)) and (d.startswith("r81_") or d.startswith("r80d_")))
    rows, bad, skipped = [], [], []
    for arm in arms:
        if F.trace_path(arm) is None or not os.path.exists(os.path.join(a.root, arm, "pos_trace_all.csv.gz")):
            if SKIP_RE.search(arm):
                skipped.append(arm)
                rows.append(dict(arm=arm, mode=(re.match(r"r81_(dc|rt)_", arm).group(1) if re.match(r"r81_(dc|rt)_", arm) else ""), note="no pos_trace_all.csv.gz (summary-only / identity / smoke / refuse / determinism arm)"))
            else:
                bad.append(arm)
            continue
        info = resolve(arm, table, F.have)
        if info is None:
            bad.append(arm); continue
        ons, osrc = pick_onset(arm, info, phase)
        if ons is None:
            sys.exit("no onset_scan_used for %s or its controls in the phase summaries" % arm)
        stat_last = None
        pj = params.get(info["bag"])
        if pj and "stat_last" in pj and int(pj.get("onset_scan", ons)) == ons:
            stat_last = int(pj["stat_last"])
        g = num(phase.get(arm, {}).get("gt_max_mm"))
        rows.append(summarize(arm, info, F.get(arm), F.get(info["ctl"]) if info["ctl"] else None, F.get(info["ref"]) if info["ref"] else None, ons, osrc, g, stat_last))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS, restval=""); w.writeheader(); w.writerows(rows)
    nocontrol = [r["arm"] for r in rows if r["arm"] not in skipped and (r["ctl_first_frame_src"] == "" or r["ref_first_frame_src"] == "")]
    print("alignfree rows %d; first-frame source state_reference %d, first_row %d; arms without trace written as note-only rows (summary-only/identity/smoke/refuse/det) %d; "
          "arms bad (name not understood or no trace) %d %s; arms with a missing control or reference %d %s"
          % (len(rows), sum(r.get("first_frame_src") == "state_reference" for r in rows), sum(r.get("first_frame_src") == "first_row" for r in rows), len(skipped), len(bad), bad[:5],
             len(nocontrol), nocontrol[:5]))
    return 1 if (bad or nocontrol) else 0


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
    fails = []

    def chk(c, m):
        if not c: fails.append(m)
    n = 600; ids = np.arange(1, n + 1)
    # base trajectory in ITS OWN first-frame coordinates: still, then moves in x
    base = np.zeros((n, 3)); base[300:, 0] = 0.01 * np.arange(n - 300)
    wob = base.copy(); wob[:, 1] += np.where(ids <= 100, 0.002 * np.sin(ids / 7.0), 0.0)          # 2 mm wobble while stationary
    yaw_true = np.where(ids <= 100, 0.003 * np.sin(ids / 11.0), 0.003 * np.sin(100 / 11.0))        # yaw drifts by up to 3 mrad, in first-frame terms
    Rz = lambda a: np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    def place(pos, yaw, y0, off):
        """world trace of a run whose first-frame trajectory is (pos, yaw) but whose start heading is y0 and start position off."""
        return (Rz(y0) @ pos.T).T + off, yaw + y0

    t = tempfile.mkdtemp()
    try:
        # (1) yaw/position invariance: arm and control are the same run started with other headings and positions
        wa, ya = place(wob, yaw_true, 2.9, np.array([5.0, -3.0, 0.1]))
        wc, yc = place(wob, yaw_true, -0.7, np.array([-40.0, 22.0, 0.0]))
        A, C = Trace(ids, wa, ya), Trace(ids, wc, yc)
        d, s, nj = departure_ff(A, C)
        chk(d < 1e-6 and s is None and nj == n, "ff departure of the same run started elsewhere must be 0: %r %r" % (d, s))
        do, _, _ = departure_old(ids, wa, ids, wc)
        chk(do > 1000.0, "old-frame departure must see the initial yaw difference (%.1f)" % do)
        # (2) a 200 mm step in the arm's own x at scan 401 is reported as 200 mm at scan 401, whatever the start
        step = wob.copy(); step[400:, 0] += 0.2
        wa2, ya2 = place(step, yaw_true, 2.9, np.array([5.0, -3.0, 0.1]))
        A2 = Trace(ids, wa2, ya2)
        d, s, nj = departure_ff(A2, C)
        chk(abs(d - 200.0) < 1e-6 and s == 401, "200 mm step: %r %r" % (d, s))
        # (3) joined on scan_id with missing scans
        keep = (ids < 10) | (ids >= 20)
        Ak = Trace(ids[keep], wa2[keep], ya2[keep], A2.p0, A2.yaw0, "state_reference")
        d2, _, nj2 = departure_ff(Ak, C)
        chk(nj2 == n - 10 and abs(d2 - 200.0) < 1e-6, "join on scan_id with missing scans: %r %r" % (d2, nj2))
        # (4) old definition reproduces R80D exactly on a same-heading pair (copy of the R80D selftest numbers)
        b = base.copy(); b2 = wob + np.array([5.0, 5.0, 0.0]); b2[400:, 0] += 0.2
        do, so, _ = departure_old(ids, b2, ids, b)
        chk(abs(do - 200.0) < 5.0 and so == 401, "old definition: %r %r" % (do, so))
        # (5) stationary metrics in first-frame coordinates, yaw drift and the first-frame source
        s_ = stationary_metrics(A, 301)
        chk(1.0 < s_["stat_drift_mm"] < 3.0, "stat drift %r" % s_["stat_drift_mm"])
        chk(abs(s_["stat_yaw_drift_mrad"] - 1000 * np.abs(yaw_true[:300] - yaw_true[0]).max()) < 1e-6, "yaw drift %r" % s_["stat_yaw_drift_mrad"])   # first_row source: relative to scan 1
        chk(abs(s_["stat_yaw_end_mrad"] - 1000 * (yaw_true[299] - yaw_true[0])) < 1e-6, "yaw end %r" % s_["stat_yaw_end_mrad"])
        chk(s_["excursion_mm"] is not None and s_["excursion_mm"] < 3.0 and s_["stat_yaw_settle_mrad"] is not None, "excursion / settle present")
        # (6) state_reference.txt overrides the first row: reference 4 mm behind in x and 10 mrad away in yaw from the first row
        ref_p = wa[0] + Rz(ya[0]) @ np.array([-0.004, 0.0, 0.0]); ref_y = ya[0] - 0.010
        ra = os.path.join(t, "ra"); _write_trace(os.path.join(ra, "r81_rt_mg_eee_01_fq", "pos_trace_all.csv.gz"), ids, wa, ya)
        _write_ref(os.path.join(ra, "r81_rt_mg_eee_01_fq", "state_reference.txt"), ref_p, ref_y)
        T6 = read_pos_trace(os.path.join(ra, "r81_rt_mg_eee_01_fq", "pos_trace_all.csv.gz"), os.path.join(ra, "r81_rt_mg_eee_01_fq", "state_reference.txt"))
        chk(T6.src == "state_reference", "source")
        exp = Rz(-ref_y) @ (wa[0] - ref_p)
        chk(np.abs(T6.pf[0] - exp).max() < 1e-9 and abs(np.linalg.norm(T6.pf[0]) - 0.004) < 1e-9, "scan 1 must sit 4 mm from the reference in its first-frame coordinates: %r" % T6.pf[0])
        chk(abs(T6.dyaw[0] - 0.010) < 1e-9, "yaw of scan 1 relative to the reference: %r" % T6.dyaw[0])
        # (7) name resolution
        ra_ = resolve("r81_rt_b3_nya_02_mq", None, lambda n: n == "r80_b3_nya_02_mq")
        chk(ra_["ctl"] == "r80_b3_nya_02_mq" and ra_["ref"] == "r81_rt_mg_nya_02_fq" and ra_["mode"] == "rt", "resolve rt twin: %r" % ra_)
        rb = resolve("r81_dc_nis_eee_01_fq", None, lambda n: False)
        chk(rb["ctl"] == "r80_mg_eee_01_fq" and rb["ref"] == "r80_b3_eee_01_mq" and rb["fam"] == "nis" and rb["src"] == "nis_eee_01_fq", "resolve dc new: %r" % rb)
        rc = resolve("r81_rt_nis_eee_01_mq", None, lambda n: n.startswith("r81_rt_b3"))
        chk(rc["ctl"] == "r81_rt_b3_eee_01_mq" and rc["ref"] == "r81_rt_mg_eee_01_fq", "resolve rt new family (rt baseline of the Q): %r" % rc)
        tab = {"r81_dc_nis_eee_01_mq": dict(src_arm="r80_b3_eee_01_mq", group="N478"), "r81_rt_nis_eee_01_mq": dict(src_arm="r80_b3_eee_01_mq", group="N478"),
               "r81_rt_pc_ps_eee_01_fq": dict(src_arm="r80_pc_ps_eee_01_fq", group="RR")}
        r1, r2 = resolve("r81_dc_nis_eee_01_mq", tab, lambda n: True), resolve("r81_rt_nis_eee_01_mq", tab, lambda n: True)
        chk(r1["ctl"] == "r80_b3_eee_01_mq" and r1["ref"] == "r80_mg_eee_01_fq" and r2["ctl"] == "r81_rt_b3_eee_01_mq" and r2["ref"] == "r81_rt_mg_eee_01_fq", "table: new family dc / rt: %r %r" % (r1, r2))
        r3 = resolve("r81_rt_pc_ps_eee_01_fq", tab, lambda n: True)
        chk(r3["ctl"] == "r80_pc_ps_eee_01_fq", "table: RR group control = the R80 source arm: %r" % r3)
        rd = resolve("r80d_off_b3mq_nya_02_o085")
        chk(rd["mode"] == "legacy" and rd["ctl"] == "r80_b3_nya_02_mq" and rd["ref"] == "r80_mg_nya_02_fq", "resolve legacy: %r" % rd)
        chk(resolve("r81_rt_x") is None and resolve("junk") is None, "unknown names")
        rdet = resolve("r81_own_det_x_eee_01", {"r81_own_det_x_eee_01": dict(mode="rt", arm_kind="det_fb", ref="r81_rt_mg_eee_01_fq", src_arm="r80_mg_eee_01_fq")}, lambda n: True)
        chk(rdet is not None and rdet["ctl"] == "r81_rt_mg_eee_01_fq" and rdet["mode"] == "rt", "determinism arm: %r" % rdet)
        chk(resolve("r81_dc_z_eee_01_fq", {"r81_dc_z_eee_01_fq": {"ctl_arm": "CTL", "ref_arm": "REF"}})["ctl"] == "CTL", "arm table override")
        # (8) end to end with the CLI: one rt arm, its R80 dc twin, its rt reference; collapse flag via gt_max
        root, ref80, rtroot = os.path.join(t, "ret"), os.path.join(t, "r80"), os.path.join(t, "rt")
        arm = "r81_rt_b3_eee_01_mq"
        big = wob.copy(); big[450:, 0] += 1.0                                           # 1 m step from scan 451: collapse
        wb, yb = place(big, yaw_true, 2.9, np.array([5.0, 5.0, 0.0]))
        wr, yr = place(wob, yaw_true, -1.1, np.array([9.0, 1.0, 0.0]))
        _write_trace(os.path.join(root, arm, "pos_trace_all.csv.gz"), ids, wb, yb)
        _write_trace(os.path.join(rtroot, "r81_rt_mg_eee_01_fq", "pos_trace_all.csv.gz"), ids, wr, yr)   # rt reference (another root: it is a control here, not an arm of this return)
        _write_trace(os.path.join(ref80, "r80_b3_eee_01_mq", "pos_trace_all.csv.gz"), ids, wc, yc)      # dc control = R80 twin
        os.makedirs(os.path.join(root, "r81_rt_sw_eee_01_off_v0p25_pfn3_fq"), exist_ok=True)   # summary-only arm: a folder without a trace is skipped, not an error
        ps = os.path.join(t, "ps.csv")
        with open(ps, "w") as f:
            f.write("arm,onset_scan_used,gt_max_mm\nr80_b3_eee_01_mq,301,86.6\n%s,301,1200.0\n" % arm)
        out = os.path.join(t, "out.csv")
        sys.argv = ["x", "--root", root, "--ref-root", ref80, "--ref-root", rtroot, "--phase-summary", ps, "--out", out]
        rc_ = main()
        R = list(csv.DictReader(open(out)))
        chk(len(R) == 2 and any(x["arm"] == "r81_rt_sw_eee_01_off_v0p25_pfn3_fq" and "no pos_trace" in x["note"] for x in R), "rows %d (the arm and a note-only row for the summary-only arm)" % len(R))
        r = {x["arm"]: x for x in R}[arm]
        chk(rc_ == 0, "cli exit code %r" % rc_)
        chk(r["mode"] == "rt" and r["ctl_arm"] == "r80_b3_eee_01_mq" and r["ref_arm"] == "r81_rt_mg_eee_01_fq" and r["first_frame_src"] == "first_row", "cli resolution %r" % r)
        chk(abs(float(r["dep_ctl_mm"]) - 1000.0) < 1e-3 and r["dep_ctl_scan50"] == "451" and r["dep_flag"] == "1" and r["collapse_flag"] == "1", "cli departure / collapse %r" % r)
        chk(float(r["dep_ctl_oldframe_mm"]) > 1500.0, "old frame departure differs (initial yaw): %r" % r["dep_ctl_oldframe_mm"])
        chk(r["onset_scan"] == "301" and r["onset_source"] == "self" and "first frame = posterior of scan 1" in r["note"], "onset / note %r" % r)
        chk(abs(float(r["stat_yaw_drift_mrad"]) - 1000 * np.abs(yaw_true[:300] - yaw_true[0]).max()) < 1e-2, "cli yaw drift %r" % r["stat_yaw_drift_mrad"])
        # gt_max below 500 -> departure large but collapse flag 0 (the second condition of rule 4.15c)
        with open(ps, "w") as f:
            f.write("arm,onset_scan_used,gt_max_mm\nr80_b3_eee_01_mq,301,86.6\n%s,301,120.0\n" % arm)
        main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[arm]
        chk(r["dep_flag"] == "1" and r["collapse_flag"] == "0", "collapse needs gt_max > 500: %r" % r)
        # a missing control is an error exit with a note
        shutil.rmtree(os.path.join(ref80, "r80_b3_eee_01_mq"))
        rc_ = main()
        r = {x["arm"]: x for x in csv.DictReader(open(out))}[arm]
        chk(rc_ == 1 and r["dep_ctl_mm"] == "" and "no ctl trace" in r["note"], "missing control must fail visibly: %r %r" % (rc_, r["note"]))
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("SELFTEST r81_alignfree", ("FAIL " + "; ".join(fails)) if fails else
          "PASS (own first-frame coordinates: a run started with another heading and position departs 0 mm from itself, the old frame sees > 1 m; 200 mm step reported as 200 mm at scan 401; "
          "join on scan_id; state_reference.txt overrides the first row; yaw drift; name resolution incl. R80 twin / legacy / table override; collapse needs gt_max > 500; missing control fails visibly)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
