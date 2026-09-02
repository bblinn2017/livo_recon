#!/usr/bin/env python3
"""Turn spline_traj.csv into a .tum, and measure how continuous it actually is.

Two jobs, and the second is the interesting one.

(1) --to-tum  The dense dump is the spline evaluated as a function of time
    across each scan's own window. Written as a .tum it can be fed to
    traj_divergence.py like any estimate, except that landing on a ground-truth
    timestamp now costs ~1/hz of interpolation instead of interpolating across
    a ~0.1 s scan gap. On a spline run that gap is precisely the thing the
    spline exists to remove, and every comparison this project has made has
    paid it.

    Overlaps: consecutive scan windows abut and may overlap. Later scans win by
    default (--overlap last), because a later fit has seen more data; --overlap
    first and --overlap mean are available for the same reason any of this is
    ablatable.

(2) --discontinuity  Nothing in this project has ever checked that the
    concatenation of per-scan splines is continuous. `anchorTo()` is supposed to
    make it so. If consecutive fits disagree at their shared boundary by more
    than sensor noise, then the "continuous-time trajectory" is a sequence of
    short splines with steps between them, and G2's central claim is weaker than
    it reads -- deskewing does not "fall out of the representation" if the
    representation jumps every 0.1 s.

    Reported as position and rotation jumps at every scan boundary. This costs
    one pass over a file that already exists.
"""
import argparse, json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R


def load(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return dict(scan=a[:, 0].astype(int), t=a[:, 1], p=a[:, 2:5], q=a[:, 5:9])


def discontinuity(d):
    """At each scan boundary, compare the two fits where their windows meet.

    For consecutive scans k and k+1, take the overlap of their time spans (or,
    if they merely abut, the last sample of k and the first of k+1) and report
    the position and rotation disagreement.
    """
    scans = np.unique(d["scan"])
    rows = []
    for i in range(len(scans) - 1):
        m0, m1 = d["scan"] == scans[i], d["scan"] == scans[i + 1]
        t0, t1 = d["t"][m0], d["t"][m1]
        if len(t0) < 2 or len(t1) < 2:
            continue
        lo, hi = max(t0[0], t1[0]), min(t0[-1], t1[-1])
        if hi > lo:                                    # genuine overlap
            grid = np.linspace(lo, hi, 5)
            p0 = np.column_stack([np.interp(grid, t0, d["p"][m0][:, k]) for k in range(3)])
            p1 = np.column_stack([np.interp(grid, t1, d["p"][m1][:, k]) for k in range(3)])
            i0 = np.searchsorted(t0, grid).clip(0, len(t0) - 1)
            i1 = np.searchsorted(t1, grid).clip(0, len(t1) - 1)
            q0, q1 = d["q"][m0][i0], d["q"][m1][i1]
            gap = 0.0
        else:                                          # abutting: end vs start
            p0, p1 = d["p"][m0][-1:], d["p"][m1][:1]
            q0, q1 = d["q"][m0][-1:], d["q"][m1][:1]
            gap = float(t1[0] - t0[-1])
        dp = float(np.max(np.linalg.norm(p0 - p1, axis=1)))
        rel = R.from_quat(q0).inv() * R.from_quat(q1)
        dr = float(np.degrees(np.max(np.linalg.norm(rel.as_rotvec(), axis=1))))
        rows.append(dict(scan_a=int(scans[i]), scan_b=int(scans[i + 1]),
                         t=float(t1[0]), overlap=bool(hi > lo), gap_s=gap,
                         d_pos_m=dp, d_rot_deg=dr))
    if not rows:
        return dict(error="no consecutive scan pairs found")
    dp = np.array([r["d_pos_m"] for r in rows])
    dr = np.array([r["d_rot_deg"] for r in rows])
    return dict(
        n_boundaries=len(rows),
        overlapping=int(sum(r["overlap"] for r in rows)),
        d_pos_m=dict(p50=float(np.median(dp)), p90=float(np.percentile(dp, 90)),
                     max=float(dp.max()), argmax_t=rows[int(dp.argmax())]["t"]),
        d_rot_deg=dict(p50=float(np.median(dr)), p90=float(np.percentile(dr, 90)),
                       max=float(dr.max()), argmax_t=rows[int(dr.argmax())]["t"]),
        note=("A continuous trajectory has boundary jumps at sensor-noise scale. "
              "Compare d_pos_m p50 against the LiDAR range noise (~0.02-0.05 m) "
              "and against redeskew_dp_rms from spline_q.csv: a jump much larger "
              "than either means anchorTo() is not delivering continuity and the "
              "'continuous-time' framing needs qualifying."),
        worst=sorted(rows, key=lambda r: -r["d_pos_m"])[:10])


def to_tum(d, overlap="last"):
    order = np.lexsort((d["scan"], d["t"])) if overlap == "last" else \
            np.lexsort((-d["scan"], d["t"]))
    t, p, q, sc = d["t"][order], d["p"][order], d["q"][order], d["scan"][order]
    keep = np.concatenate([np.diff(t) > 0, [True]])       # last wins per timestamp
    if overlap == "mean":
        ut, inv = np.unique(t, return_inverse=True)
        pm = np.zeros((len(ut), 3)); cnt = np.zeros(len(ut))
        np.add.at(pm, inv, p); np.add.at(cnt, inv, 1.0)
        pm /= cnt[:, None]
        qi = np.zeros((len(ut), 4)); qi[inv] = q          # rotation: nearest, not averaged
        return ut, pm, qi
    return t[keep], p[keep], q[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", help="spline_traj.csv")
    ap.add_argument("--to-tum", metavar="OUT.tum")
    ap.add_argument("--overlap", default="last", choices=["last", "first", "mean"])
    ap.add_argument("--discontinuity", action="store_true")
    a = ap.parse_args()

    d = load(a.traj)
    out = dict(rows=int(len(d["t"])), scans=int(len(np.unique(d["scan"]))),
               t_span=[float(d["t"].min()), float(d["t"].max())])
    if a.discontinuity:
        out["discontinuity"] = discontinuity(d)
    if a.to_tum:
        t, p, q = to_tum(d, a.overlap)
        np.savetxt(a.to_tum, np.column_stack([t, p, q]), fmt="%.9f")
        out["wrote"] = dict(path=a.to_tum, n=int(len(t)), overlap=a.overlap)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
