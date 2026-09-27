#!/usr/bin/env python3
"""Offline consistency analysis for physical LiDAR architecture diagnostics.

NIS is summarized from covariance_calibration rows.  Position NEES is computed
at each position_covariance_full timestamp by LINEARLY INTERPOLATING the GT
position trajectory at that exact timestamp; frame-index matching is never used.
"""
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np

CHI2_1_95 = 3.841458820694124
CHI2_1_99 = 6.6348966010212145
CHI2_3_95 = 7.814727903251179
CHI2_3_99 = 11.344866730144373

def finite(rows, key):
    out=[]
    for r in rows:
        try:
            x=float(r[key])
            if math.isfinite(x): out.append(x)
        except (KeyError, TypeError, ValueError): pass
    return out

def qtile(xs, q):
    xs=sorted(xs)
    if not xs: return float("nan")
    x=q*(len(xs)-1)
    lo=int(math.floor(x)); hi=int(math.ceil(x))
    return xs[lo]+(xs[hi]-xs[lo])*(x-lo)

def inv3(a,b,c,d,e,f):
    A=d*f-e*e; B=c*e-b*f; C=b*e-c*d
    D=a*f-c*c; E=b*c-a*e; F=a*d-b*b
    det=a*A+b*B+c*C
    if not math.isfinite(det) or abs(det)<1e-15: return None
    return ((A,B,C),(B,D,E),(C,E,F)), det

def nees_3(e, p):
    # BUG FIXED this round: inv3() returns the ADJUGATE matrix plus det
    # separately (inverse = adjugate/det) -- this previously used the raw
    # adjugate as if it were already the normalized inverse, silently
    # discarding the division by det. With det ~ (1e-4)^3 for a well-localized
    # covariance, that under-scaled every NEES value by ~1/det (order 1e8-1e12
    # here), making severe overconfidence look like near-zero NEES. Never
    # exercised end-to-end before this round because the GT join itself was
    # 100%-excluded until the scan_id->epoch translation fix above.
    inv=inv3(*p)
    if inv is None: return None
    m,det=inv
    return sum(e[i]*m[i][j]*e[j] for i in range(3) for j in range(3))/det

def kabsch_align(est, gt):
    # Mirrors evo_processing.cpp's computeAte() EXACTLY (Kabsch/Horn, no
    # scale): finds R,t minimizing sum||R*est_i+t - gt_i||^2, i.e. the
    # transform mapping the estimator's own trajectory frame onto GT's frame.
    est=np.asarray(est); gt=np.asarray(gt)
    c_est=est.mean(axis=0); c_gt=gt.mean(axis=0)
    H=(est-c_est).T @ (gt-c_gt)
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T @ U.T
    if np.linalg.det(R)<0:
        Vt2=Vt.copy(); Vt2[2,:]*=-1
        R=Vt2.T @ U.T
    t=c_gt - R @ c_est
    return R,t

def load_gt(path):
    # Schema note (found and fixed this round): the actual lio_gt_matched.csv
    # written by evo_processing.cpp has columns t_abs,est_px..,gt_px,gt_py,gt_pz
    # (absolute epoch seconds, positions already in meters) -- NOT the
    # t/gt_matched/px_mm/py_mm/pz_mm schema this loader originally assumed
    # (that schema belongs to ate_per_frame_<stage>_<mode>.csv instead). Every
    # row of lio_gt_matched.csv is already a matched GT sample, so there is no
    # separate "gt_matched" flag to filter on here.
    #
    # SECOND BUG FOUND AND FIXED this round: est_px/py/pz and gt_px/py/pz in
    # this file are each RAW, in their own sensor's coordinate frame (Leica's
    # frame for GT, the estimator's own world/map frame for est) -- they are
    # NOT already Kabsch-aligned onto a shared frame (that alignment happens
    # only inside computeAte(), transiently, and is never itself written out).
    # Directly differencing raw est (position_covariance_full's own p_x/y/z,
    # which matches this file's est_px/y/z) against raw gt_px/y/z therefore
    # measures a ~10-20m frame offset, not a real position error -- this
    # produced spurious NEES in the tens of millions on first attempt (see
    # runD_local_spline_corroff's uncorrected trial). Fixed by replicating
    # evo_processing.cpp's own computeAte() Kabsch/Horn alignment (same
    # algorithm, no scale) over this file's full matched (est,gt) trajectory,
    # then transforming GT into the ESTIMATOR's frame (the inverse of the
    # est->gt transform) before returning it -- so the caller's e = est - gt
    # is expressed in the same frame P (position_covariance_full's Pp_*) is.
    with open(path,newline="") as f: rows=list(csv.DictReader(f))
    ts=[]; ests=[]; gts=[]
    for r in rows:
        try:
            t=float(r.get("t_abs", "nan"))
            if not math.isfinite(t): continue
            ts.append(t)
            ests.append((float(r["est_px"]),float(r["est_py"]),float(r["est_pz"])))
            gts.append((float(r["gt_px"]),float(r["gt_py"]),float(r["gt_pz"])))
        except (KeyError,TypeError,ValueError): continue
    if len(ts)<3: return []
    R,t_vec=kabsch_align(ests,gts)
    Rt=R.T
    pts=[]
    for tt,g in zip(ts,gts):
        g_est_frame = Rt @ (np.asarray(g)-t_vec)
        pts.append((tt, float(g_est_frame[0]), float(g_est_frame[1]), float(g_est_frame[2])))
    pts.sort()
    return pts

def load_frame_epoch(path):
    # position_covariance_full's own t_abs is BAG-RELATIVE (t1 relative to the
    # spline fit window), not absolute epoch -- frame_stats.txt's own "t"
    # column, keyed by frame_idx, IS absolute epoch, and position_covariance_
    # full's scan_id == frame_idx+1 (confirmed by direct inspection this
    # round). Returns {scan_id: absolute_epoch_t}.
    out={}
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[int(float(r["frame_idx"]))+1]=float(r["t"])
            except (KeyError,TypeError,ValueError): continue
    return out

def interp_gt(gt, t):
    if len(gt)<2 or t<gt[0][0] or t>gt[-1][0]: return None
    lo,hi=0,len(gt)-1
    while lo+1<hi:
        mid=(lo+hi)//2
        if gt[mid][0] <= t: lo=mid
        else: hi=mid
    t0,x0,y0,z0=gt[lo]; t1,x1,y1,z1=gt[hi]
    if t1<=t0:
        return (x0,y0,z0)
    a=(t-t0)/(t1-t0)
    return ((1-a)*x0+a*x1,(1-a)*y0+a*y1,(1-a)*z0+a*z1)

def region_of(t_abs, stationary_end_abs):
    if stationary_end_abs is None or t_abs is None: return "combined"
    return "stationary" if t_abs < stationary_end_abs else "movement"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("diag_csv")
    ap.add_argument("--gt", default="")
    ap.add_argument("--frame-stats", default="", help="frame_stats.txt, to translate position_covariance_full's bag-relative t_abs to the GT file's absolute epoch via scan_id=frame_idx+1")
    ap.add_argument("--stationary-end-abs", type=float, default=None, help="absolute epoch timestamp separating stationary/movement, derived from the reference trajectory")
    ap.add_argument("--out", default="")
    args=ap.parse_args()
    with open(args.diag_csv,newline="") as f: rows=list(csv.DictReader(f))
    frame_epoch=load_frame_epoch(args.frame_stats) if args.frame_stats else None
    lines=["metric,update_mode,region,count,mean,median,p95,p99,fraction_gt_95,fraction_gt_99"]
    groups=defaultdict(list)
    for r in rows:
        if r.get("row_type")!="covariance_calibration": continue
        region="combined"
        if frame_epoch is not None and args.stationary_end_abs is not None:
            try:
                sid=int(float(r["scan_id"]))
                if sid in frame_epoch: region=region_of(frame_epoch[sid], args.stationary_end_abs)
            except (KeyError,TypeError,ValueError): pass
        for reg in ({region,"combined"} if region!="combined" else {"combined"}):
            groups[(r.get("update_mode","unknown"),reg)].append(r)
    for (mode,region),rs in sorted(groups.items()):
        xs=finite(rs,"nis_mean")
        tail95=finite(rs,"nis_gt_3p84_fraction")
        tail99=finite(rs,"nis_gt_6p63_fraction")
        if xs:
            lines.append("NIS_mean,{},{},{},{:.9g},{:.9g},{:.9g},{:.9g},{:.9g},{:.9g}".format(mode,region,len(xs),sum(xs)/len(xs),qtile(xs,.5),qtile(xs,.95),qtile(xs,.99),qtile(tail95,.5) if tail95 else float('nan'),qtile(tail99,.5) if tail99 else float('nan')))
        tr=finite(rs,"trace_Pz_reduction_fraction")
        if tr:
            lines.append("Pz_trace_reduction,{},{},{},{:.9g},{:.9g},{:.9g},{:.9g},NA,NA".format(mode,region,len(tr),sum(tr)/len(tr),qtile(tr,.5),qtile(tr,.95),qtile(tr,.99)))
    if args.gt:
        gt=load_gt(args.gt)
        pg=defaultdict(list); excluded=defaultdict(int)
        for r in rows:
            if r.get("row_type")!="position_covariance_full": continue
            if r.get("scan_end_t1") in (None,"","NA"): continue  # per-GN-iteration emission, not the end-of-frame tail row this analysis targets
            mode=r.get("update_mode",r.get("test_id","unknown"))
            try:
                t=float(r["t_abs"])
                sid=int(float(r["scan_id"]))
                if frame_epoch is not None:
                    if sid not in frame_epoch:
                        excluded[(mode,"combined")]+=1; continue
                    t=frame_epoch[sid]
                region = region_of(t, args.stationary_end_abs) if args.stationary_end_abs is not None else "combined"
                egt=interp_gt(gt,t)
                if egt is None:
                    excluded[(mode,region)]+=1; excluded[(mode,"combined")]+=1; continue
                est=(float(r["p_x"]),float(r["p_y"]),float(r["p_z"]))
                e=tuple(est[i]-egt[i] for i in range(3))
                p=(float(r["Pp_xx"]),float(r["Pp_xy"]),float(r["Pp_xz"]),float(r["Pp_yy"]),float(r["Pp_yz"]),float(r["Pp_zz"]))
            except (KeyError,TypeError,ValueError):
                excluded[(mode,"combined")]+=1; continue
            n=nees_3(e,p)
            if n is None or not math.isfinite(n): excluded[(mode,region)]+=1; excluded[(mode,"combined")]+=1; continue
            pg[(mode,region)].append(n)
            pg[(mode,"combined")].append(n)
        for (mode,region),ns in sorted(pg.items()):
            lines.append("NEES3,{},{},{},{:.9g},{:.9g},{:.9g},{:.9g},{:.9g},{:.9g}".format(mode,region,len(ns),sum(ns)/len(ns),qtile(ns,.5),qtile(ns,.95),qtile(ns,.99),sum(n>CHI2_3_95 for n in ns)/len(ns),sum(n>CHI2_3_99 for n in ns)/len(ns)))
        for (mode,region),n in sorted(excluded.items()):
            lines.append("NEES3_excluded_outside_gt_bracket,{},{},{},NA,NA,NA,NA,NA,NA".format(mode,region,n))
        if gt:
            lines.append("GT_interpolation_range,all,combined,{},{:.9g},{:.9g},NA,NA,NA,NA".format(1,gt[0][0],gt[-1][0]))
    text="\n".join(lines)+"\n"
    if args.out: Path(args.out).write_text(text)
    else: print(text,end="")

if __name__=="__main__": main()
