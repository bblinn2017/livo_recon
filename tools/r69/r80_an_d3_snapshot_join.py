#!/usr/bin/env python3
"""R80 D3 (F-465 / F-441): why only 2-8 percent of the frame-1 nodes match at 200 calibration samples but 94 percent at 400. HOST analysis, no arm of its own.

Approved text (planner): 'Host analysis first. The map snapshot already carries point count, eigenvalues, status, retired flag and the plane test values per node. Join it to the frame-1 dump
for the R79 n200 and n400 arms. First check whether the post-calibration snapshot is among the returned files. If it isn't, ask the user for it. Arms (in R80): calibration samples 250, 300,
350 on eee_01, fixed-Q, 0.425 m, lines off, pfn 1 and 3, with the frame 1-5 dump and per-frame snapshots on. What it settles: whether the jump comes from the number of calibration frames or
from the minimum-points rule on the frame-0 map.'

    python3 r80_an_d3_snapshot_join.py --root <dir holding one sub-directory per arm> [--out <dir>] [--iter first|N] [--min-init-points 5]
    python3 r80_an_d3_snapshot_join.py --preflight --root <dir>
    python3 r80_an_d3_snapshot_join.py --selftest

Arm discovery: every sub-directory of --root whose name contains _n<samples>_pfn<pfn> (R79 name form r79_ec_n200_pfn1_fq; the R80 arms are expected to follow the same form: check with --preflight).
Inputs per arm (compact R79 return names, or the raw files as fallback):
  dump       residual_dump_all.csv.gz  | residual_dump.csv[.gz]            (columns frame, iter, node_id, ...; src/processing/lio_base.cpp logResidualDump header)
  snapshot   snapshot_frames_all.csv.gz (map_frame + the 28 columns of r79_early_reduce.COLS)
             | post_calibration_voxelmap.csv (map_frame 1) + voxelmap_snapshot_f<N>.csv (map_frame N)   (src/map/voxelmap.cpp writeSnapshotFile)
Join: dump rows of frame N with the snapshot rows of map_frame N by node_id (voxelmap.cpp:779-781: the file written for frame_idx_ = N is the map frame N queries; the dump uses the frame index
the map has DURING the update, lio_base.cpp residual_dump 'frame' = voxel_map_->frame_idx_). Map frame 1 = post_calibration_voxelmap.csv.
Dump iteration: the dump holds one block per dumped iteration; by default the LOWEST iteration present in each frame (iteration 0, the propagated state, as logResidualDump's comment says);
--iter N picks iteration N instead. This is a read choice, not a threshold.

Minimum-points rule (the real constant): VoxelPlane::isInit() is points_size_ > opts_->min_init_points (include/livo_recon/lio/voxelplane.h:53); the value is voxel_map/points/min_init = 5
(config/ntu_viral.yaml:197; the C++ struct default voxelmap_utils.h:309 is 5, the paramWarn fallback voxelmap.cpp:206 is 3, so the yaml value is the one a run uses). A node's `initialized` column
in the snapshot IS isInit(). This script reports BOTH the snapshot `initialized` share and the share with point_count > MIN_INIT_POINTS (they must agree: the check is printed). The first-fit
rule outside frame 0 additionally needs distinct_frames >= voxel_map/plane/min_frames_to_init (default 1, voxelmap.cpp:202; skipped for global frame 0, voxelnode.cpp); distinct_frames is
reported in the post-calibration table so that the frame-0 exemption can be seen.
No other threshold is used. Quantile levels in the distribution table are display choices."""
import argparse, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r79_early_reduce as E

MIN_INIT_POINTS = 5                  # voxel_map/points/min_init, config/ntu_viral.yaml:197 ; isInit(): point_count > this (voxelplane.h:53)
FRAMES = (1, 2, 3, 4, 5)
SAMPLES_ORDER = (200, 250, 300, 350, 400)
ARM_RE = re.compile(r"_n(\d+)_pfn(\d+)")
DUMP_NEED = ["frame", "iter", "node_id"]
SNAP_NEED = ["node_id", "point_count", "is_plane", "initialized", "distinct_frames", "layer", "eig0", "eig1", "eig2"]
Q_LEVELS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


# ---------------------------------------------------------------------------------------- discovery and loading
def _first(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def find_arms(root):
    arms = []
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n)
        m = ARM_RE.search(n)
        if os.path.isdir(p) and m:
            arms.append((int(m.group(1)), int(m.group(2)), n, p))
    return sorted(arms)


def inventory(d):
    """-> dict of what is present in one arm dir."""
    dump = _first(d, ["residual_dump_all.csv.gz", "residual_dump.csv", "residual_dump.csv.gz"])
    comp = _first(d, ["snapshot_frames_all.csv.gz"])
    raw = E.snapshot_files(d)
    post_raw = os.path.exists(os.path.join(d, "post_calibration_voxelmap.csv"))
    post_comp = False
    comp_frames = []
    if comp:
        try:
            mf = pd.read_csv(comp, usecols=["map_frame"])["map_frame"]
            comp_frames = sorted(int(x) for x in mf.unique())
            post_comp = 1 in comp_frames
        except Exception:
            pass
    return dict(dump=dump, snap_compact=comp, snap_compact_frames=comp_frames, snap_raw_frames=[f for f, _ in raw], post_cal=post_raw or post_comp)


def preflight(root, out=None):
    """Prints, per arm, which required files are present; returns True when every arm has the post-calibration snapshot and a dump."""
    out = out or sys.stdout
    arms = find_arms(root)
    if not arms:
        print("NO ARM DIRECTORIES (name containing _n<samples>_pfn<pfn>) under %s" % root, file=out)
        return False
    ok = True
    for ns, pfn, name, d in arms:
        inv = inventory(d)
        print("arm %s (samples %d, pfn %d)" % (name, ns, pfn), file=out)
        print("   dump (residual_dump_all.csv.gz | residual_dump.csv):          %s" % (os.path.basename(inv["dump"]) if inv["dump"] else "ABSENT"), file=out)
        print("   snapshot_frames_all.csv.gz:                                   %s" % ("map frames %s" % inv["snap_compact_frames"] if inv["snap_compact"] else "ABSENT"), file=out)
        print("   raw snapshots (post_calibration_voxelmap.csv, voxelmap_snapshot_f<N>.csv): map frames %s" % (inv["snap_raw_frames"] or "ABSENT"), file=out)
        have = set(inv["snap_compact_frames"]) | set(inv["snap_raw_frames"])
        miss = [f for f in FRAMES if f not in have]
        if miss:
            print("   snapshot map frames missing of 1..5: %s" % miss, file=out)
        if not inv["post_cal"]:
            print("   POST-CALIBRATION SNAPSHOT MISSING: ask the user", file=out)
            ok = False
        if not inv["dump"]:
            print("   DUMP MISSING: ask the user", file=out)
            ok = False
    return ok


def load_dump(d):
    p = _first(d, ["residual_dump_all.csv.gz", "residual_dump.csv", "residual_dump.csv.gz"])
    if p is None:
        raise SystemExit("residual dump absent in %s" % d)
    D = pd.read_csv(p)
    miss = [c for c in DUMP_NEED if c not in D.columns]
    if miss:
        raise SystemExit("%s lacks columns %s" % (p, miss))
    return D


def load_snaps(d, need=SNAP_NEED, extra=()):
    """-> DataFrame map_frame + columns. Compact file preferred; raw files for any frame missing from it."""
    cols = list(dict.fromkeys(list(need) + list(extra)))
    parts, have = [], set()
    comp = _first(d, ["snapshot_frames_all.csv.gz"])
    if comp:
        S = pd.read_csv(comp)
        miss = [c for c in cols + ["map_frame"] if c not in S.columns]
        if miss:
            raise SystemExit("%s lacks columns %s" % (comp, miss))
        parts.append(S[["map_frame"] + cols]); have |= set(int(x) for x in S.map_frame.unique())
    for fr, p in E.snapshot_files(d):
        if fr in have:
            continue
        R = pd.read_csv(p)
        miss = [c for c in cols if c not in R.columns]
        if miss:
            raise SystemExit("%s lacks columns %s" % (p, miss))
        R = R[cols].copy(); R.insert(0, "map_frame", fr); parts.append(R); have.add(fr)
    if not parts:
        raise SystemExit("no snapshot in %s" % d)
    return pd.concat(parts, ignore_index=True)


def pick_iter(D, it):
    """Dump rows of one iteration per frame: it == 'first' -> the lowest iteration present in that frame."""
    if it == "first":
        lo = D.groupby("frame")["iter"].transform("min")
        return D[D["iter"] == lo]
    return D[D["iter"] == int(it)]


# ---------------------------------------------------------------------------------------- the analysis
def per_frame(D, S, min_init=MIN_INIT_POINTS, it="first"):
    """One row per frame in FRAMES (a frame with no dump rows or no snapshot is reported with NaN, never dropped)."""
    D = pick_iter(D, it)
    out = []
    for f in FRAMES:
        d = D[D["frame"] == f]
        s = S[S["map_frame"] == f]
        r = dict(frame=f, residual_rows=int(len(d)), distinct_nodes_hit=int(d["node_id"].nunique()) if len(d) else 0, snapshot_nodes=int(len(s)))
        if len(s):
            enough = s["point_count"] > min_init
            r.update(snap_plane_share=float(s["is_plane"].astype(bool).mean()), snap_initialized_share=float(s["initialized"].astype(bool).mean()),
                     snap_pointcount_gt_min_share=float(enough.mean()), snap_plane_and_gt_min_share=float((s["is_plane"].astype(bool) & enough).mean()),
                     init_flag_equals_pointcount_rule=bool((s["initialized"].astype(bool) == enough).all()),
                     snap_nodes_hit_share=float(len(set(d["node_id"]) & set(s["node_id"])) / len(s)))
            if len(d):
                isp = d["node_id"].map(s.drop_duplicates("node_id").set_index("node_id")["is_plane"])
                matched = isp.notna()
                r.update(rows_node_in_snapshot_share=float(matched.mean()), rows_node_is_plane_share=float(isp.fillna(False).astype(bool).mean()),
                         rows_node_is_plane_of_matched=float(isp[matched].astype(bool).mean()) if matched.any() else float("nan"))
        out.append(r)
    cols = ["frame", "residual_rows", "distinct_nodes_hit", "snapshot_nodes", "snap_plane_share", "snap_initialized_share", "snap_pointcount_gt_min_share", "snap_plane_and_gt_min_share",
            "init_flag_equals_pointcount_rule", "snap_nodes_hit_share", "rows_node_in_snapshot_share", "rows_node_is_plane_share", "rows_node_is_plane_of_matched"]
    return pd.DataFrame(out).reindex(columns=cols)


def postcal_distribution(S, min_init=MIN_INIT_POINTS):
    """Point-count distribution per node of the post-calibration snapshot (map_frame 1): quantiles over all nodes, over initialized nodes, over plane nodes, plus counts by value."""
    s = S[S["map_frame"] == 1]
    if not len(s):
        return None, None
    rows = []
    for label, sub in (("all nodes", s), ("is_plane", s[s["is_plane"].astype(bool)]), ("initialized", s[s["initialized"].astype(bool)]), ("point_count <= min", s[s["point_count"] <= min_init])):
        pc = sub["point_count"].to_numpy(float)
        row = dict(group=label, nodes=int(len(sub)))
        for q in Q_LEVELS:
            row["q%g" % q] = float(np.quantile(pc, q)) if len(pc) else float("nan")
        row["mean"] = float(pc.mean()) if len(pc) else float("nan")
        row["distinct_frames_max"] = float(sub["distinct_frames"].max()) if len(sub) else float("nan")
        rows.append(row)
    vc = s["point_count"].value_counts().sort_index().rename_axis("point_count").reset_index(name="nodes")
    vc["share"] = vc["nodes"] / len(s)
    vc["cum_share"] = vc["share"].cumsum()
    return pd.DataFrame(rows), vc


def across_arms(results):
    """results {(samples, pfn): per_frame DataFrame}. One row per arm ordered by samples (200, 250, 300, 350, 400, then any other), then pfn; one column group per frame 1..5."""
    keys = sorted(results, key=lambda k: (SAMPLES_ORDER.index(k[0]) if k[0] in SAMPLES_ORDER else len(SAMPLES_ORDER) + k[0], k[1]))
    metrics = ["residual_rows", "distinct_nodes_hit", "snapshot_nodes", "snap_plane_share", "snap_pointcount_gt_min_share", "snap_nodes_hit_share", "rows_node_is_plane_share"]
    rows = []
    for k in keys:
        r = dict(samples=k[0], pfn=k[1])
        pf = results[k].set_index("frame")
        for f in FRAMES:
            for m in metrics:
                r["f%d_%s" % (f, m)] = pf.loc[f, m] if f in pf.index else float("nan")
        rows.append(r)
    T = pd.DataFrame(rows)
    missing = [s for s in SAMPLES_ORDER if s not in {k[0] for k in results}]
    return T, missing


def run(root, out=None, it="first", min_init=MIN_INIT_POINTS):
    if not preflight(root):
        print("preflight failed: not every arm has the post-calibration snapshot and the dump; nothing joined for the arms that lack them")
    results, dists = {}, {}
    for ns, pfn, name, d in find_arms(root):
        inv = inventory(d)
        if not (inv["post_cal"] and inv["dump"]):
            continue
        D, S = load_dump(d), load_snaps(d)
        pf = per_frame(D, S, min_init, it)
        results[(ns, pfn)] = pf
        dist, vc = postcal_distribution(S, min_init)
        dists[(ns, pfn)] = (dist, vc)
        print("\n=== arm %s  samples %d  pfn %d  (min-points rule: point_count > %d)" % (name, ns, pfn, min_init))
        print(pf.to_string(index=False, float_format=lambda x: "%.4f" % x))
        print("post-calibration snapshot (map_frame 1) point_count distribution")
        print(dist.to_string(index=False, float_format=lambda x: "%.4g" % x))
        if out:
            os.makedirs(out, exist_ok=True)
            pf.to_csv(os.path.join(out, "d3_per_frame_n%d_pfn%d.csv" % (ns, pfn)), index=False)
            dist.to_csv(os.path.join(out, "d3_postcal_pointcount_n%d_pfn%d.csv" % (ns, pfn)), index=False)
            vc.to_csv(os.path.join(out, "d3_postcal_pointcount_values_n%d_pfn%d.csv" % (ns, pfn)), index=False)
    if not results:
        raise SystemExit("no arm could be analysed")
    T, missing = across_arms(results)
    print("\n=== across arms, ordered by calibration samples (frame 1 columns first below, all frames in the CSV)")
    show = ["samples", "pfn"] + ["f1_" + m for m in ("residual_rows", "distinct_nodes_hit", "snapshot_nodes", "snap_plane_share", "snap_pointcount_gt_min_share", "snap_nodes_hit_share", "rows_node_is_plane_share")]
    print(T[show].to_string(index=False, float_format=lambda x: "%.4f" % x))
    if missing:
        print("sample counts of the expected list %s with NO arm in this root: %s" % (list(SAMPLES_ORDER), missing))
    if out:
        T.to_csv(os.path.join(out, "d3_across_arms.csv"), index=False)
    return results, T


# ---------------------------------------------------------------------------------------- selftest
def _write_arm(d, ns, pfn, n_nodes, n_pc_small, n_plane, rng, omit_postcal=False, raw=False):
    """Synthetic arm with KNOWN composition. Snapshot map frames 1..5 share the node set; nodes 0..n_pc_small-1 have point_count 3 (<= 5), the rest 12; the first n_plane nodes by index among
    the point_count 12 nodes are planes. Dump: frame f has rows only on plane nodes (first 20 rows per node) plus 10 rows on one non-plane node plus 7 rows on a node absent from the snapshot."""
    os.makedirs(d, exist_ok=True)
    snaps = []
    for f in FRAMES:
        pc = np.where(np.arange(n_nodes) < n_pc_small, 3, 12)
        isp = np.zeros(n_nodes, bool); isp[n_pc_small:n_pc_small + n_plane] = True
        S = pd.DataFrame(dict(map_frame=f, node_id=np.arange(n_nodes), layer=0, status=0, retired=0, initialized=(pc > MIN_INIT_POINTS).astype(int), is_plane=isp.astype(int), full=0,
                              point_count=pc, distinct_frames=1 if f == 1 else f, eig0=0.001, eig1=0.01, eig2=0.1))
        snaps.append(S)
    rows = []
    plane_ids = list(range(n_pc_small, n_pc_small + n_plane))
    non_plane = n_nodes - 1
    for f in FRAMES:
        for it in (0, 1):
            for nid in plane_ids:
                rows += [dict(frame=f, iter=it, node_id=nid)] * 20
            rows += [dict(frame=f, iter=it, node_id=non_plane)] * 10 + [dict(frame=f, iter=it, node_id=n_nodes + 500)] * 7
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(d, "residual_dump_all.csv.gz" if not raw else "residual_dump.csv"), index=False)
    allS = pd.concat(snaps, ignore_index=True)
    if raw:
        for f, S in zip(FRAMES, snaps):
            if f == 1 and omit_postcal:
                continue
            S.drop(columns="map_frame").assign(snapshot_frame=f).to_csv(os.path.join(d, "post_calibration_voxelmap.csv" if f == 1 else "voxelmap_snapshot_f%d.csv" % f), index=False)
    else:
        if omit_postcal:
            allS = allS[allS.map_frame != 1]
        allS.to_csv(os.path.join(d, "snapshot_frames_all.csv.gz"), index=False)


def selftest():
    import tempfile, io
    fails = []
    rng = np.random.default_rng(1)
    root = tempfile.mkdtemp()
    # arm compositions: samples -> (nodes, nodes with point_count <= min, planes among the rest)
    spec = {200: (100, 90, 4), 250: (100, 60, 20), 300: (100, 30, 40), 350: (100, 10, 60), 400: (100, 4, 90)}
    for ns, (n, ps, npl) in spec.items():
        for pfn in (1, 3):
            _write_arm(os.path.join(root, "r80_ec_n%d_pfn%d_fq" % (ns, pfn)), ns, pfn, n, ps, npl, rng)
    buf = io.StringIO()
    if not preflight(root, out=buf) or "MISSING" in buf.getvalue():
        fails.append("preflight on a complete root reported a problem:\n" + buf.getvalue())
    # numbers: arm 250 pfn 3
    d = os.path.join(root, "r80_ec_n250_pfn3_fq")
    D, S = load_dump(d), load_snaps(d)
    pf = per_frame(D, S).set_index("frame")
    f1 = pf.loc[1]
    exp = dict(residual_rows=20 * 20 + 10 + 7, distinct_nodes_hit=20 + 2, snapshot_nodes=100, snap_plane_share=0.20, snap_initialized_share=0.40, snap_pointcount_gt_min_share=0.40,
               snap_plane_and_gt_min_share=0.20, snap_nodes_hit_share=21 / 100, rows_node_in_snapshot_share=410 / 417, rows_node_is_plane_share=400 / 417, rows_node_is_plane_of_matched=400 / 410)
    for k, v in exp.items():
        if abs(float(f1[k]) - v) > 1e-12:
            fails.append("n250 frame 1 %s = %r want %r" % (k, f1[k], v))
    if not bool(f1["init_flag_equals_pointcount_rule"]):
        fails.append("initialized flag vs point_count rule check false")
    if len(pf) != 5 or not (pf["residual_rows"] == f1["residual_rows"]).all():
        fails.append("frames 1..5 rows")
    # iteration choice: iter 0 and 1 both dumped, default must NOT double count
    p1 = per_frame(D, S, it=1).set_index("frame").loc[1, "residual_rows"]
    if p1 != f1["residual_rows"]:
        fails.append("iter selection differs in count")
    # distribution
    dist, vc = postcal_distribution(S)
    allr = dist[dist.group == "all nodes"].iloc[0]
    if allr.nodes != 100 or allr["q0"] != 3 or allr["q1"] != 12 or abs(allr["q0.5"] - 3) > 1e-12 or abs(allr["q0.75"] - 12) > 1e-12 or vc.set_index("point_count").loc[3, "nodes"] != 60 or abs(vc["cum_share"].iloc[-1] - 1) > 1e-12:
        fails.append("post-cal distribution %s" % allr.to_dict())
    # across arms: ordering and jump visible
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()) as cap:
        results, T = run(root, out=os.path.join(root, "out"))
    if list(T.samples) != [200, 200, 250, 250, 300, 300, 350, 350, 400, 400] or list(T.pfn) != [1, 3] * 5:
        fails.append("across-arm order %s" % list(zip(T.samples, T.pfn)))
    sh = T[T.pfn == 1]["f1_snap_pointcount_gt_min_share"].tolist()
    if not np.allclose(sh, [0.10, 0.40, 0.70, 0.90, 0.96]):
        fails.append("across-arm share column %s" % sh)
    if not os.path.exists(os.path.join(root, "out", "d3_across_arms.csv")):
        fails.append("across-arm csv not written")
    # missing samples reported
    shutil_root = tempfile.mkdtemp()
    _write_arm(os.path.join(shutil_root, "a_n200_pfn1_fq"), 200, 1, 50, 40, 3, rng)
    _write_arm(os.path.join(shutil_root, "a_n400_pfn1_fq"), 400, 1, 50, 2, 40, rng)
    with contextlib.redirect_stdout(io.StringIO()) as cap:
        run(shutil_root)
    if "[250, 300, 350]" not in cap.getvalue():
        fails.append("absent sample counts not reported")
    # missing post-calibration snapshot: compact and raw
    for raw in (False, True):
        r2 = tempfile.mkdtemp()
        _write_arm(os.path.join(r2, "b_n200_pfn3_fq"), 200, 3, 50, 40, 3, rng, omit_postcal=True, raw=raw)
        buf = io.StringIO(); ok = preflight(r2, out=buf)
        if ok or "POST-CALIBRATION SNAPSHOT MISSING: ask the user" not in buf.getvalue():
            fails.append("missing post-cal snapshot not flagged (raw=%s): %s" % (raw, buf.getvalue()))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run(r2)
            fails.append("run on an arm without the post-cal snapshot did not stop")
        except SystemExit:
            pass
    # raw-file path gives the same numbers
    r3 = tempfile.mkdtemp()
    _write_arm(os.path.join(r3, "c_n250_pfn3_fq"), 250, 3, 100, 60, 20, rng, raw=True)
    pf3 = per_frame(load_dump(os.path.join(r3, "c_n250_pfn3_fq")), load_snaps(os.path.join(r3, "c_n250_pfn3_fq")))
    if not np.allclose(pf3.fillna(-1).select_dtypes("number").to_numpy(float), pf.reset_index().fillna(-1).select_dtypes("number").to_numpy(float)):
        fails.append("raw-file path differs from compact path")
    # missing column fails
    r4 = tempfile.mkdtemp(); _write_arm(os.path.join(r4, "d_n200_pfn1_fq"), 200, 1, 20, 10, 3, rng)
    p = os.path.join(r4, "d_n200_pfn1_fq", "snapshot_frames_all.csv.gz"); pd.read_csv(p).drop(columns=["point_count"]).to_csv(p, index=False)
    try:
        load_snaps(os.path.join(r4, "d_n200_pfn1_fq")); fails.append("snapshot without point_count did not fail")
    except SystemExit:
        pass
    print("SELFTEST r80_an_d3_snapshot_join", "FAIL " + "; ".join(fails) if fails else
          "PASS (planted composition recovered exactly for every per-frame column; initialized flag == point_count > %d rule; iteration choice does not double count; ordering 200..400 by samples and a planted jump in the "
          "across-arm column; absent sample counts listed; missing post-cal snapshot flagged in compact and raw form and stops the run; raw-file path equals compact path; a missing column fails)" % MIN_INIT_POINTS)
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--iter", default="first"); ap.add_argument("--min-init-points", type=int, default=MIN_INIT_POINTS)
    ap.add_argument("--preflight", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.root:
        sys.exit("--root is required")
    if a.preflight:
        sys.exit(0 if preflight(a.root) else 2)
    run(a.root, a.out, a.iter, a.min_init_points)
