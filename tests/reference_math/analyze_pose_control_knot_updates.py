#!/usr/bin/env python3
"""Compare per-knot pose-spline motion across coupled and decoupled modes.

Inputs:
  coupled_full_diag.csv: unified pose_control_full_diagnostics.csv
  decoupled_knot_update.csv: decoupled_knot_update.csv

The analysis reports iteration-local and frame-cumulative position/rotation
motion for each knot, plus the ratio to the direct-tail physical correction.
It can split the run at a supplied stationary-duration threshold.
"""
import argparse, csv, math
from collections import defaultdict


def f(r,k):
    try:
        x=float(r[k]); return x if math.isfinite(x) else None
    except (KeyError,TypeError,ValueError): return None

def q(v,p):
    v=sorted(v)
    if not v: return float('nan')
    x=p*(len(v)-1); lo=int(math.floor(x)); hi=int(math.ceil(x))
    return v[lo]+(v[hi]-v[lo])*(x-lo)

def summarize(label, rows, split):
    groups=defaultdict(lambda:{"iter_p":[],"iter_r":[],"frame_p":[],"frame_r":[],"tail":[]})
    for r in rows:
        t=f(r,"t_abs_frame_end")
        regions=["combined"]
        if t is not None:
            regions.append("stationary" if t < split else "movement")
        for region in regions:
            key=(region,r.get("control_point_index","NA"))
            if r.get("scope")=="iteration":
                groups[key]["iter_p"].append(f(r,"delta_cp_norm") or 0.0)
                groups[key]["iter_r"].append(f(r,"delta_cp_phi_norm") or 0.0)
            elif r.get("scope")=="frame_cumulative":
                groups[key]["frame_p"].append(f(r,"delta_cp_norm") or 0.0)
                groups[key]["frame_r"].append(f(r,"delta_cp_phi_norm") or 0.0)
            tail=f(r,"delta_tail_physical_norm")
            if tail is not None: groups[key]["tail"].append(tail)
    out=[]
    for (region,k),g in sorted(groups.items()):
        out.append({"architecture":label,"region":region,"knot":k,
                    "iter_p_median":q(g["iter_p"],.5),"iter_p_p95":q(g["iter_p"],.95),
                    "iter_r_median":q(g["iter_r"],.5),"iter_r_p95":q(g["iter_r"],.95),
                    "frame_p_median":q(g["frame_p"],.5),"frame_p_p95":q(g["frame_p"],.95),
                    "frame_r_median":q(g["frame_r"],.5),"frame_r_p95":q(g["frame_r"],.95),
                    "tail_physical_median":q(g["tail"],.5)})
    return out

def load_frame_epoch(path):
    # Coupled knot_update/knot_update_frame rows carry scan_id + a wall-clock
    # "timestamp" but no bag-relative-or-absolute t_abs_frame_end field (unlike
    # the decoupled logger, which the patch gave that field directly) -- found
    # this round while trying to region-split the coupled side. frame_stats.txt's
    # "t" column, keyed by frame_idx, is absolute epoch; scan_id == frame_idx+1
    # (same mapping confirmed for position_covariance_full elsewhere this round).
    out={}
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            try: out[int(float(r["frame_idx"]))+1]=float(r["t"])
            except (KeyError,TypeError,ValueError): continue
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--coupled", required=True)
    ap.add_argument("--decoupled", required=True)
    ap.add_argument("--stationary-end", type=float, required=True, help="absolute timestamp separating stationary and movement")
    ap.add_argument("--coupled-frame-stats", default="", help="frame_stats.txt for the coupled run, to translate scan_id to absolute epoch for region splitting")
    args=ap.parse_args()
    frame_epoch=load_frame_epoch(args.coupled_frame_stats) if args.coupled_frame_stats else None
    allrows=[]
    with open(args.coupled,newline="") as fd:
        for r in csv.DictReader(fd):
            if r.get("row_type") in ("knot_update","knot_update_frame"):
                if frame_epoch is not None and not f(r,"t_abs_frame_end"):
                    try:
                        sid=int(float(r["scan_id"]))
                        if sid in frame_epoch: r["t_abs_frame_end"]=str(frame_epoch[sid])
                    except (KeyError,TypeError,ValueError): pass
                allrows.append(r)
    coupled=summarize("coupled",allrows,args.stationary_end)
    dec=[]
    with open(args.decoupled,newline="") as fd:
        for r in csv.DictReader(fd):
            dec.append(r)
    decoupled=summarize("decoupled",dec,args.stationary_end)
    print("architecture,region,knot,iter_p_median,iter_p_p95,iter_r_median,iter_r_p95,frame_p_median,frame_p_p95,frame_r_median,frame_r_p95,tail_physical_median")
    for r in coupled+decoupled:
        print("{architecture},{region},{knot},{iter_p_median:.9g},{iter_p_p95:.9g},{iter_r_median:.9g},{iter_r_p95:.9g},{frame_p_median:.9g},{frame_p_p95:.9g},{frame_r_median:.9g},{frame_r_p95:.9g},{tail_physical_median:.9g}".format(**r))

if __name__=="__main__": main()
