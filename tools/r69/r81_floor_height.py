#!/usr/bin/env python3
"""R81 F-480 record: the floor height of each bag, in the map frame (world z; the map's origin is the start pose). ANALYSIS ONLY: nothing here feeds a run. Host tool, reads returned early-arm files.

    python3 r81_floor_height.py --root <dir holding the returned early arms> --out <dir> [--map-frames 1] [--nz-min 0.95] [--bin 0.02] [--halfwidth 0.10] [--source auto|snapshot|dump]
    python3 r81_floor_height.py --arm eee_01=<arm dir> [--arm eee_01=<arm dir2> ...] --out <dir>
    python3 r81_floor_height.py --selftest

Bryce wants the floor height of every bag kept as a record (the ground/above-ground split of F-480 uses H = 0.0 for every bag; this file says where the floor really is relative to that cut).

Two measurements, the same estimator on both
  snapshot  a voxel snapshot: post_calibration_voxelmap.csv (map frame 1) or snapshot_frames_all.csv.gz (column map_frame; r79_early_reduce.py OUT_COLS). Voxel planes with |normal_z| >= --nz-min are floor
            candidates; the height of a voxel is its plane centre centre_z, its weight its point_count. Columns used: normal_z, center_z, point_count (+ map_frame when present, + distinct_frames if present).
            --map-frames (default 1 = the post-calibration snapshot, the first map) picks which snapshots.
  dump      the residual dump of the early arm (residual_dump_all.csv.gz, residual_dump.csv[.gz] or window_ground_rows_all.csv.gz): rows with |hpos_z| >= --nz-min (the row's world normal), height = wz (the
            world point z), weight 1 per row, ONLY the first dumped iteration of each frame (the iterations of one frame are the same points and would be counted again).
Estimator (identical for both): weighted histogram of the heights with bin width --bin (default 0.02 m = range_err, bins anchored at multiples of the width); the MODE bin (the largest weight; a tie goes to
  the lower bin: the floor is the lowest large horizontal surface) -> mode_z (bin centre). floor_z = weighted MEDIAN of the heights within +-halfwidth (default 0.10 m) of the mode_z; spread_m = weighted standard
  deviation of the same heights about floor_z; p25_z / p75_z = weighted quartiles of the same heights. n_points = total weight of those heights, n_voxels = number of voxels (rows) in them, n_flat_points =
  weight of ALL |normal| >= nz-min candidates, peak_share = n_points / n_flat_points (a small share = several horizontal surfaces, check second_peak_*). second_peak_z / second_peak_share = the largest bin that lies
  more than 2*halfwidth from the mode and its weight / n_flat_points (a table top, a ceiling, a second floor level; no decision is taken on it).
Outputs (--out)
  r81_floor_heights_by_arm.csv  one row per (bag, arm, source): the values above, plus frames used and the file read.
  r81_floor_heights.csv         ONE row per bag: floor_z_m, how (method string: source + file kind + frames + bin/halfwidth/nz-min), n_points, n_voxels, spread_m, p25_z, p75_z, n_arms, arm_range_mm (max - min of
                                floor_z over the arms of the bag, same source), other_source_z_m and other_source_diff_mm (the other source's value for the same bag when both exist), second_peak_z, second_peak_share,
                                floor_minus_H_m for H = 0.0 (floor_z - 0.0, the Bryce cut). The representative arm is the lower median arm; source preference snapshot over dump (--source forces one).
A bag with no usable arm, an arm directory with no readable file, a missing column or an empty candidate set is a nonzero exit naming the bag (never an empty row). Nothing here picks, ranks or recommends a height."""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd

BAGS = ["eee_01", "eee_02", "eee_03", "nya_01", "nya_02", "nya_03", "sbs_01", "sbs_02", "sbs_03"]
SNAP_FILES = ["snapshot_frames_all.csv.gz", "snapshot_frames_all.csv", "post_calibration_voxelmap.csv", "post_calibration_voxelmap.csv.gz"]
DUMP_FILES = ["residual_dump_all.csv.gz", "residual_dump_all.csv", "residual_dump.csv", "residual_dump.csv.gz", "window_ground_rows_all.csv.gz"]
H_BRYCE = 0.0
BY_ARM_COLS = ["bag", "arm", "source", "file", "frames", "floor_z_m", "mode_z_m", "n_points", "n_voxels", "n_flat_points", "peak_share", "spread_m", "p25_z", "p75_z", "second_peak_z", "second_peak_share"]
BAG_COLS = ["bag", "floor_z_m", "method", "n_points", "n_voxels", "spread_m", "p25_z", "p75_z", "n_arms", "arm_range_mm", "other_source_z_m", "other_source_diff_mm", "second_peak_z", "second_peak_share",
            "floor_minus_H_m", "representative_arm", "n_flat_points", "peak_share"]


def wquantile(x, w, q):
    o = np.argsort(x, kind="stable"); x, w = x[o], w[o]
    c = np.cumsum(w); t = q * c[-1]
    return float(x[min(int(np.searchsorted(c, t, side="left")), len(x) - 1)])


def estimate(z, w, binw, half):
    """-> dict of the estimator on heights z with weights w. Raises SystemExit on an empty set."""
    z, w = np.asarray(z, float), np.asarray(w, float)
    ok = np.isfinite(z) & np.isfinite(w) & (w > 0)
    z, w = z[ok], w[ok]
    if len(z) == 0:
        raise SystemExit("no floor candidates (no finite heights with positive weight)")
    k = np.floor(z / binw).astype(np.int64)
    ks, inv = np.unique(k, return_inverse=True)
    bw = np.bincount(inv, weights=w)
    i = int(np.argmax(bw))                       # first maximum = lowest bin on a tie (ks ascending)
    mode_z = (ks[i] + 0.5) * binw
    m = np.abs(z - mode_z) <= half
    zz, ww = z[m], w[m]
    fz = wquantile(zz, ww, 0.5)
    tot = float(w.sum())
    far = np.abs((ks + 0.5) * binw - mode_z) > 2 * half
    sp_z, sp_sh = float("nan"), 0.0
    if far.any():
        j = int(np.argmax(np.where(far, bw, -1.0)))
        sp_z, sp_sh = float((ks[j] + 0.5) * binw), float(bw[j] / tot)
    return dict(floor_z_m=fz, mode_z_m=float(mode_z), n_points=float(ww.sum()), n_voxels=int(m.sum()), n_flat_points=tot, peak_share=float(ww.sum() / tot),
                spread_m=float(np.sqrt(np.sum(ww * (zz - fz) ** 2) / ww.sum())), p25_z=wquantile(zz, ww, 0.25), p75_z=wquantile(zz, ww, 0.75), second_peak_z=sp_z, second_peak_share=sp_sh)


def _first(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def read_snapshot(path, nz_min, map_frames):
    D = pd.read_csv(path, usecols=lambda c: c in ("map_frame", "normal_z", "center_z", "point_count"))
    miss = [c for c in ("normal_z", "center_z", "point_count") if c not in D.columns]
    if miss:
        raise SystemExit("%s lacks columns %s" % (os.path.basename(path), miss))
    if "map_frame" in D.columns:
        D = D[D.map_frame.isin(map_frames)]
        frames = sorted(int(f) for f in D.map_frame.unique())
    else:                                        # post_calibration_voxelmap.csv: map frame 1
        frames = [1]
        if 1 not in map_frames:
            raise SystemExit("%s is map frame 1 only but --map-frames is %s" % (os.path.basename(path), sorted(map_frames)))
    D = D[D.normal_z.abs() >= nz_min]
    if len(D) == 0:
        raise SystemExit("%s: no voxel with |normal_z| >= %g in map frames %s" % (os.path.basename(path), nz_min, sorted(map_frames)))
    return D.center_z.to_numpy(float), D.point_count.to_numpy(float), frames


def read_dump(path, nz_min):
    parts = []
    for ch in pd.read_csv(path, chunksize=200000, usecols=lambda c: c in ("frame", "iter", "hpos_z", "wz")):
        miss = [c for c in ("frame", "iter", "hpos_z", "wz") if c not in ch.columns]
        if miss:
            raise SystemExit("%s lacks columns %s" % (os.path.basename(path), miss))
        parts.append(ch)
    D = pd.concat(parts, ignore_index=True)
    D = D[D["iter"] == D.groupby("frame")["iter"].transform("min")]
    D = D[D.hpos_z.abs() >= nz_min]
    if len(D) == 0:
        raise SystemExit("%s: no dump row with |hpos_z| >= %g" % (os.path.basename(path), nz_min))
    return D.wz.to_numpy(float), np.ones(len(D)), sorted(int(f) for f in D.frame.unique())


def measure_arm(bag, arm, d, sources, nz_min, map_frames, binw, half):
    rows = []
    if "snapshot" in sources:
        p = _first(d, SNAP_FILES)
        if p:
            z, w, fr = read_snapshot(p, nz_min, map_frames)
            rows.append(dict(bag=bag, arm=arm, source="snapshot", file=os.path.basename(p), frames=" ".join(map(str, fr)), **estimate(z, w, binw, half)))
    if "dump" in sources:
        p = _first(d, DUMP_FILES)
        if p:
            z, w, fr = read_dump(p, nz_min)
            fs = " ".join(map(str, fr)) if len(fr) <= 12 else "%d..%d (%d frames)" % (fr[0], fr[-1], len(fr))
            rows.append(dict(bag=bag, arm=arm, source="dump", file=os.path.basename(p), frames=fs, **estimate(z, w, binw, half)))
    return rows


def find_arms(root):
    """-> [(bag, arm name, dir)]: every directory under root with a snapshot or dump file; the bag is the one bag name found in its path relative to root."""
    out, seen = [], set()
    for dp, _dn, files in os.walk(root):
        if any(f in files for f in SNAP_FILES + DUMP_FILES):
            rel = os.path.relpath(dp, root)
            hit = [b for b in BAGS if b in rel]
            if len(hit) != 1:
                raise SystemExit("cannot assign %s to one bag (bag names found in the path: %s); use --arm BAG=DIR" % (dp, hit))
            out.append((hit[0], os.path.basename(dp.rstrip("/")) if rel != "." else hit[0], dp))
    return sorted(out)


def aggregate(A, source_pref, binw, half, nz_min):
    """A = by-arm frame -> per-bag frame."""
    out = []
    for bag, g in A.groupby("bag", sort=True):
        srcs = [s for s in ("snapshot", "dump") if (g.source == s).any()]
        use = source_pref if source_pref in ("snapshot", "dump") else srcs[0]
        if use not in srcs:
            raise SystemExit("bag %s: no %s measurement" % (bag, use))
        gs = g[g.source == use].sort_values("floor_z_m")
        rep = gs.iloc[(len(gs) - 1) // 2]            # lower median
        other = [s for s in srcs if s != use]
        oz = float(g[g.source == other[0]].floor_z_m.median()) if other else float("nan")
        out.append(dict(bag=bag, floor_z_m=float(rep.floor_z_m),
                        method="%s (%s, map frames/frames %s), weighted histogram mode bin %g m, median within +-%g m, |normal z| >= %g; lower-median arm of %d" % (use, rep.file, rep.frames, binw, half, nz_min, len(gs)),
                        n_points=float(rep.n_points), n_voxels=int(rep.n_voxels), spread_m=float(rep.spread_m), p25_z=float(rep.p25_z), p75_z=float(rep.p75_z), n_arms=int(len(gs)),
                        arm_range_mm=float(1000 * (gs.floor_z_m.max() - gs.floor_z_m.min())), other_source_z_m=oz, other_source_diff_mm=float(1000 * (oz - rep.floor_z_m)) if other else float("nan"),
                        second_peak_z=float(rep.second_peak_z), second_peak_share=float(rep.second_peak_share), floor_minus_H_m=float(rep.floor_z_m - H_BRYCE), representative_arm=rep.arm,
                        n_flat_points=float(rep.n_flat_points), peak_share=float(rep.peak_share)))
    return pd.DataFrame(out)[BAG_COLS]


def run(arms, out, sources=("snapshot", "dump"), nz_min=0.95, map_frames=(1,), binw=0.02, half=0.10, source_pref="auto", need_bags=None):
    rows, failed = [], []
    for bag, name, d in arms:
        try:
            r = measure_arm(bag, name, d, sources, nz_min, set(map_frames), binw, half)
            if not r:
                raise SystemExit("no snapshot or dump file in %s" % d)
            rows += r
        except SystemExit as e:
            failed.append("%s/%s: %s" % (bag, name, e)); print("FAILED %s/%s: %s" % (bag, name, e), file=sys.stderr)
    have = {r["bag"] for r in rows}
    for b in (need_bags or []):
        if b not in have:
            failed.append("%s: no measurement" % b)
    if failed:
        raise SystemExit("floor height failed for: " + "; ".join(failed))
    A = pd.DataFrame(rows)[BY_ARM_COLS]
    B = aggregate(A, source_pref, binw, half, nz_min)
    os.makedirs(out, exist_ok=True)
    A.to_csv(os.path.join(out, "r81_floor_heights_by_arm.csv"), index=False, float_format="%.6g")
    B.to_csv(os.path.join(out, "r81_floor_heights.csv"), index=False, float_format="%.6g")
    return A, B


# ---------------------------------------------------------------------------------------------- selftest
def _synth_snapshot(path, floor, rng, frames=(1,), extra_table=-0.70):
    rows = []
    for f in frames:
        for _ in range(400):                      # floor voxels: horizontal, heights scatter 1 cm around the floor
            rows.append(dict(map_frame=f, normal_z=rng.choice([1.0, -1.0]) * rng.uniform(0.96, 1.0), center_z=floor + rng.normal(0, 0.01), point_count=int(rng.integers(50, 150))))
        for _ in range(120):                      # a table top: horizontal, fewer points
            rows.append(dict(map_frame=f, normal_z=0.99, center_z=extra_table + rng.normal(0, 0.01), point_count=int(rng.integers(10, 40))))
        for _ in range(300):                      # walls: not candidates
            rows.append(dict(map_frame=f, normal_z=rng.uniform(-0.3, 0.3), center_z=rng.uniform(floor, floor + 3), point_count=int(rng.integers(50, 200))))
        for _ in range(30):                       # a sloped (|nz| = 0.9) patch at floor height: below nz-min, excluded
            rows.append(dict(map_frame=f, normal_z=0.9, center_z=floor + 0.5, point_count=500))
    D = pd.DataFrame(rows); D["distinct_frames"] = 1; D["eig0"] = 0.0
    D.to_csv(path, index=False)
    return D


def _synth_dump(path, floor, rng):
    rows = []
    for fr in range(1, 6):
        for it in (0, 1):
            for _ in range(500):
                rows.append(dict(frame=fr, iter=it, hpos_z=rng.choice([1.0, -1.0]) * rng.uniform(0.96, 1.0), wz=floor + rng.normal(0, 0.02)))
            for _ in range(800 if it == 1 else 200):  # iteration 1 repeats must not be counted: put an offset population only in iter 1 so a double count would shift the result
                rows.append(dict(frame=fr, iter=it, hpos_z=0.99, wz=(floor + 0.08 + rng.normal(0, 0.005)) if it == 1 else (floor + rng.uniform(1.0, 2.0))))
            for _ in range(300):
                rows.append(dict(frame=fr, iter=it, hpos_z=0.1, wz=rng.uniform(floor, floor + 3)))
    pd.DataFrame(rows).to_csv(path, index=False)


def selftest():
    import tempfile, subprocess
    fails = []
    rng = np.random.default_rng(11)
    root = tempfile.mkdtemp()
    floors = {"eee_01": -1.80, "nya_02": 0.35, "sbs_03": 0.0}
    for bag, fz in floors.items():
        for arm in ("a_n400", "a_n800"):
            d = os.path.join(root, "x", bag, arm); os.makedirs(d)
            _synth_snapshot(os.path.join(d, "snapshot_frames_all.csv"), fz + (0.004 if arm == "a_n800" else 0.0), rng, frames=(1, 2))
            _synth_dump(os.path.join(d, "residual_dump_all.csv.gz"), fz, rng)
    arms = find_arms(os.path.join(root, "x"))
    if len(arms) != 6:
        fails.append("find_arms %s" % arms)
    A, B = run(arms, os.path.join(root, "out"), need_bags=list(floors))
    for bag, fz in floors.items():
        r = B[B.bag == bag].iloc[0]
        if abs(r.floor_z_m - fz) > 0.012 or r.n_arms != 2 or abs(r.arm_range_mm - 4.0) > 2.5 or r.spread_m > 0.03 or r.peak_share < 0.7:
            fails.append("%s snapshot floor %r (want %g) arms %d range %.2f mm spread %.4f share %.2f" % (bag, r.floor_z_m, fz, r.n_arms, r.arm_range_mm, r.spread_m, r.peak_share))
        if abs(r.other_source_z_m - fz) > 0.012:
            fails.append("%s dump cross-check %r" % (bag, r.other_source_z_m))
        if abs(r.second_peak_z + 0.70) > 0.02 and bag == "eee_01" and r.second_peak_share > 0:
            pass
        if abs(r.floor_minus_H_m - r.floor_z_m) > 1e-12:
            fails.append("floor_minus_H")
    # the dump alone, double counting guard: iteration 1 holds an extra population at floor + 0.08 inside the window; counting both iterations would pull the median up
    Ad, Bd = run(arms[:2], os.path.join(root, "out_d"), sources=("dump",), source_pref="dump")
    rd = Bd.iloc[0]
    if abs(rd.floor_z_m - floors["eee_01"]) > 0.01 or "dump" not in rd.method:
        fails.append("dump-only floor %r" % rd.floor_z_m)
    # the table top (-0.70, fewer points) is the second peak for the eee_01 floor at -1.80 (distance 1.1 m > 0.2)
    ra = A[(A.bag == "eee_01") & (A.source == "snapshot")].iloc[0]
    if abs(ra.second_peak_z + 0.70) > 0.02 or not (0 < ra.second_peak_share < 0.5):
        fails.append("second peak %r share %r" % (ra.second_peak_z, ra.second_peak_share))
    # map_frames filter: the same values in frame 2 -> n_points doubles with --map-frames 1 2
    A2, _ = run(arms[:1], os.path.join(root, "out2"), sources=("snapshot",), map_frames=(1, 2))
    A1, _ = run(arms[:1], os.path.join(root, "out1"), sources=("snapshot",), map_frames=(1,))
    if not (1.6 < A2.n_flat_points.iloc[0] / A1.n_flat_points.iloc[0] < 2.4):
        fails.append("map-frames filter: %r vs %r" % (A2.n_flat_points.iloc[0], A1.n_flat_points.iloc[0]))
    # post_calibration_voxelmap.csv layout (no map_frame column)
    d = os.path.join(root, "pc", "sbs_01"); os.makedirs(d)
    S = _synth_snapshot(os.path.join(d, "post_calibration_voxelmap.csv"), -0.9, rng).drop(columns=["map_frame"]); S.to_csv(os.path.join(d, "post_calibration_voxelmap.csv"), index=False)
    _, Bp = run(find_arms(os.path.join(root, "pc")), os.path.join(root, "outp"))
    if abs(Bp.floor_z_m.iloc[0] + 0.9) > 0.012 or Bp.n_arms.iloc[0] != 1:
        fails.append("post_calibration layout %r" % Bp.floor_z_m.iloc[0])
    # tie goes to the lower bin
    e = estimate([0.01, 1.01], [5.0, 5.0], 0.02, 0.1)
    if abs(e["floor_z_m"] - 0.01) > 1e-12:
        fails.append("tie: %r" % e["floor_z_m"])
    # failures: a missing column, nothing above nz-min, a missing bag, an arm dir with no file, an unassignable path
    d = os.path.join(root, "bad", "eee_02"); os.makedirs(d)
    pd.DataFrame({"normal_z": [1.0], "center_z": [0.0]}).to_csv(os.path.join(d, "post_calibration_voxelmap.csv"), index=False)
    for what, fn in (("missing column", lambda: run(find_arms(os.path.join(root, "bad")), os.path.join(root, "o"))),
                     ("missing bag", lambda: run(arms, os.path.join(root, "o"), need_bags=["eee_01", "nya_03"])),
                     ("no candidates", lambda: run([("eee_01", "z", os.path.join(root, "x", "eee_01", "a_n400"))], os.path.join(root, "o"), nz_min=2.0)),
                     ("empty dir", lambda: run([("eee_01", "e", tempfile.mkdtemp())], os.path.join(root, "o")))):
        try:
            fn(); fails.append("%s did not fail" % what)
        except SystemExit:
            pass
    amb = os.path.join(root, "amb", "eee_01_nya_01"); os.makedirs(amb); open(os.path.join(amb, "post_calibration_voxelmap.csv"), "w").write("x\n")
    try:
        find_arms(os.path.join(root, "amb")); fails.append("ambiguous path accepted")
    except SystemExit:
        pass
    # CLI end to end
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--root", os.path.join(root, "x"), "--out", os.path.join(root, "cli"), "--require-all-bags"], capture_output=True, text=True)
    if r.returncode == 0 or "no measurement" not in r.stderr:
        fails.append("--require-all-bags with 3 of 9 bags did not fail (rc %s)" % r.returncode)
    r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--root", os.path.join(root, "x"), "--out", os.path.join(root, "cli")], capture_output=True, text=True)
    C = pd.read_csv(os.path.join(root, "cli", "r81_floor_heights.csv")) if r.returncode == 0 else None
    if C is None or list(C.columns) != BAG_COLS or len(C) != 3:
        fails.append("cli run rc %s %s" % (r.returncode, r.stderr[-200:]))
    print("SELFTEST r81_floor_height", "FAIL " + "; ".join(fails) if fails else
          "PASS (planted floors -1.80 / 0.35 / 0.0 m recovered within 12 mm from snapshots (table top at -0.70 is the second peak, walls and |nz| 0.9 patches excluded) and from the dump (first iteration only), "
          "two arms per bag -> arm range, map-frame filter, post_calibration layout, tie to the lower bin; missing column, no candidates, missing bag, empty dir, ambiguous path each fail)")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--arm", action="append", default=[], metavar="BAG=DIR"); ap.add_argument("--out")
    ap.add_argument("--map-frames", type=int, nargs="+", default=[1]); ap.add_argument("--nz-min", type=float, default=0.95)
    ap.add_argument("--bin", type=float, default=0.02); ap.add_argument("--halfwidth", type=float, default=0.10)
    ap.add_argument("--source", choices=["auto", "snapshot", "dump"], default="auto")
    ap.add_argument("--require-all-bags", action="store_true", help="fail unless all nine bags have a measurement")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.out or not (a.root or a.arm):
        sys.exit("--out and --root (or --arm BAG=DIR) are required")
    arms = find_arms(a.root) if a.root else []
    for s in a.arm:
        b, _, d = s.partition("=")
        if b not in BAGS or not os.path.isdir(d):
            sys.exit("--arm %r: need <one of %s>=<existing dir>" % (s, BAGS))
        arms.append((b, os.path.basename(d.rstrip("/")), d))
    srcs = ("snapshot", "dump") if a.source == "auto" else (a.source,)
    A, B = run(arms, a.out, srcs, a.nz_min, tuple(a.map_frames), a.bin, a.halfwidth, a.source, BAGS if a.require_all_bags else None)
    print(B[["bag", "floor_z_m", "n_points", "spread_m", "n_arms", "arm_range_mm", "other_source_diff_mm", "second_peak_z", "second_peak_share"]].to_string(index=False, float_format=lambda x: "%.4f" % x))
