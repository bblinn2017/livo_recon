#!/usr/bin/env python3
"""R80 D7 (F-467): which input blocks the ZUPT stationarity detector on the bags where it never fired. Host-only; planning agent's script, not run on real data by its author.

    python3 r80_an_d7_zupt.py --arm eee_01=<arm dir> --arm nya_02=<arm dir> ... --params r69_inputs/bag_params.json --out <dir>
                              [--gyro-thresh 0.02] [--acc-thresh 0.15] [--enter-scans 3] [--first-frame 1] [--frame-offset 0]
    python3 r80_an_d7_zupt.py --selftest

Input per arm: <arm dir>/zupt.csv (or .gz)  columns frame,gyro_mean,acc_mean,run,stationary,applied,vnorm_before   (src/processing/lio_iekf.cpp applyZuptIfStationary).
  gyro_mean = mean over the scan's IMU poses of |bias-corrected gyro|; acc_mean = mean of |0.5 (acc_head + acc_tail)| (world-frame specific force + gravity, ~0 at rest).
Detector (include/livo_recon/lio/zupt.h): candidate = gyro_mean < gyro_thresh AND acc_mean < acc_thresh (strict); run = candidate ? run + 1 : 0; stationary = run >= enter_scans.
Known-stationary window: defined as in r69_reduce.py ("stationary = scans 1..stat_last") with stat_last = bag_params.json[bag]["stat_last"] (bag_stats.py: onset_scan - 5, onset =
first time the smoothed IMU dynamic acceleration exceeds 0.25 m/s^2 for 39 samples). Window = frames --first-frame .. stat_last (default first-frame 1, i.e. scans 1..stat_last).
frame = scan + --frame-offset (default 0; zupt.csv's frame is voxel_map frame_idx_, which may differ from scan_id by one; check).
The detector is re-run here from gyro_mean / acc_mean over the WHOLE file (so run lengths carry in from before the window) with the thresholds given on the command line; the
recomputed flag is compared with the logged one (mismatches are reported; they are expected to be 0 at the defaults, which are the logged thresholds).
Blocker per window frame (exactly one class): ok (both inputs below threshold), gyro (only gyro >= thresh), acc (only acc >= thresh), both. A frame with both inputs ok but
run < enter_scans is 'ok' for the inputs and shows in run_short (the start-up ramp, or the frames just after a failure).
Outputs: d7_frames.csv (every window frame of every bag: inputs, ratios to the thresholds, three condition flags, blocker, recomputed run and flag, logged run and flag),
d7_summary.csv (per bag), d7_report.txt."""
import argparse, json, os, sys
import numpy as np
import pandas as pd

NAMES = ["zupt.csv", "zupt.csv.gz"]
COLS = ["frame", "gyro_mean", "acc_mean", "run", "stationary"]


def load(d):
    for n in NAMES:
        p = os.path.join(d, n)
        if os.path.exists(p):
            D = pd.read_csv(p)
            miss = [c for c in COLS if c not in D.columns]
            if miss:
                raise SystemExit("%s lacks columns %s" % (p, miss))
            if len(D) == 0:
                raise SystemExit("%s has no rows" % p)
            return D.drop_duplicates("frame", keep="last").sort_values("frame").reset_index(drop=True)
    raise SystemExit("no zupt.csv in %s (was lio/zupt/mode detect_only set?)" % d)


def redetect(g, a, gt, at, enter):
    run, runs, flags = 0, [], []
    for x, y in zip(g, a):
        run = run + 1 if (x < gt and y < at) else 0
        runs.append(run)
        flags.append(run >= enter)
    return np.array(runs), np.array(flags)


def analyse(bag, D, stat_last, a):
    g, c = D.gyro_mean.to_numpy(float), D.acc_mean.to_numpy(float)
    run, flag = redetect(g, c, a.gyro_thresh, a.acc_thresh, a.enter_scans)
    D = D.copy()
    D["scan"] = D.frame - a.frame_offset
    D["gyro_ok"] = g < a.gyro_thresh
    D["acc_ok"] = c < a.acc_thresh
    D["gyro_over_thresh"] = g / a.gyro_thresh
    D["acc_over_thresh"] = c / a.acc_thresh
    D["run_re"], D["stationary_re"] = run, flag
    D["run_ok"] = run >= a.enter_scans
    D["blocker"] = np.where(D.gyro_ok & D.acc_ok, "ok", np.where(~D.gyro_ok & D.acc_ok, "gyro", np.where(D.gyro_ok & ~D.acc_ok, "acc", "both")))
    W = D[(D.scan >= a.first_frame) & (D.scan <= stat_last)].copy()
    W.insert(0, "bag", bag)
    S = {"bag": bag, "stat_last": stat_last, "window_first": a.first_frame, "frames_in_zupt_csv": len(D), "window_frames": len(W),
         "logged_flag_mismatch_whole_file": int((D.stationary.astype(int).to_numpy() != flag.astype(int)).sum()),
         "logged_ever_stationary": bool(D.stationary.astype(int).any())}
    if len(W) == 0:
        S["status"] = "EMPTY WINDOW (no zupt.csv frame in %d..%d)" % (a.first_frame, stat_last)
        return W, S
    n = len(W)
    for k in ("ok", "gyro", "acc", "both"):
        S["frac_blocker_" + k] = float((W.blocker == k).sum() / n)
    S["frac_gyro_fails"] = float((~W.gyro_ok).sum() / n)
    S["frac_acc_fails"] = float((~W.acc_ok).sum() / n)
    S["frac_run_short"] = float((~W.run_ok).sum() / n)
    S["frac_candidate_but_run_short"] = float(((W.blocker == "ok") & ~W.run_ok).sum() / n)
    for nm in ("gyro_mean", "acc_mean"):
        x = W[nm].to_numpy(float)
        S[nm + "_min"], S[nm + "_median"], S[nm + "_p90"], S[nm + "_max"] = float(x.min()), float(np.median(x)), float(np.quantile(x, 0.9)), float(x.max())
    S["max_run_in_window"] = int(W.run_re.max())
    fire = W[W.stationary_re]
    S["first_fire_scan_in_window"] = int(fire.scan.iloc[0]) if len(fire) else "NEVER"
    anyf = D[D.stationary_re]
    S["first_fire_scan_whole_file"] = int(anyf.scan.iloc[0]) if len(anyf) else "NEVER"
    if S["first_fire_scan_in_window"] == "NEVER":
        top = max(("gyro", "acc", "both"), key=lambda k: S["frac_blocker_" + k])
        S["main_blocker"] = top if S["frac_blocker_" + top] > 0 else "run_length_only"
    else:
        S["main_blocker"] = "n/a (fires)"
    S["status"] = "ok"
    return W, S


def run_all(a):
    P = json.load(open(a.params))
    os.makedirs(a.out, exist_ok=True)
    Ws, Ss = [], []
    for spec in a.arm:
        bag, _, d = spec.partition("=")
        if not d:
            raise SystemExit("--arm wants bag=dir, got %r" % spec)
        if bag not in P or "stat_last" not in P[bag]:
            raise SystemExit("%s has no stat_last in %s" % (bag, a.params))
        W, S = analyse(bag, load(d), int(P[bag]["stat_last"]), a)
        Ws.append(W); Ss.append(S)
    pd.concat(Ws, ignore_index=True).to_csv(os.path.join(a.out, "d7_frames.csv"), index=False, float_format="%.9g")
    SS = pd.DataFrame(Ss)
    SS.to_csv(os.path.join(a.out, "d7_summary.csv"), index=False, float_format="%.6g")
    L = ["D7 (F-467) ZUPT detector inputs on the known-stationary window (scans %d..stat_last)" % a.first_frame,
         "thresholds: gyro < %g rad/s, acc < %g m/s^2, run >= %d (strict <)" % (a.gyro_thresh, a.acc_thresh, a.enter_scans), ""]
    for S in Ss:
        L.append("== %s: stat_last %s, %s" % (S["bag"], S["stat_last"], S["status"]))
        if S["status"] != "ok":
            continue
        L.append("  first fire in window: %s; anywhere in file: %s; logged flag ever set: %s; recomputed-vs-logged flag mismatches: %d" %
                 (S["first_fire_scan_in_window"], S["first_fire_scan_whole_file"], S["logged_ever_stationary"], S["logged_flag_mismatch_whole_file"]))
        L.append("  blocker fractions (window frames): ok %.3f  gyro-only %.3f  acc-only %.3f  both %.3f   | gyro fails %.3f, acc fails %.3f, run short %.3f (candidate but run short %.3f)" %
                 tuple(S[k] for k in ("frac_blocker_ok", "frac_blocker_gyro", "frac_blocker_acc", "frac_blocker_both", "frac_gyro_fails", "frac_acc_fails", "frac_run_short", "frac_candidate_but_run_short")))
        L.append("  gyro_mean min/med/p90/max %.4g %.4g %.4g %.4g ; acc_mean %.4g %.4g %.4g %.4g ; max run %d ; main blocker: %s" %
                 (S["gyro_mean_min"], S["gyro_mean_median"], S["gyro_mean_p90"], S["gyro_mean_max"], S["acc_mean_min"], S["acc_mean_median"],
                  S["acc_mean_p90"], S["acc_mean_max"], S["max_run_in_window"], S["main_blocker"]))
    open(os.path.join(a.out, "d7_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def selftest():
    import tempfile
    t = tempfile.mkdtemp()
    rng = np.random.RandomState(0)
    n = 200
    params = {k: {"stat_last": 100} for k in ("fires", "gyrobad", "accbad", "bothbad")}
    json.dump(params, open(os.path.join(t, "p.json"), "w"))

    def mk(name, g, c):
        d = os.path.join(t, name); os.makedirs(d)
        run, rows = 0, []
        for i in range(n):
            cand = g[i] < 0.02 and c[i] < 0.15
            run = run + 1 if cand else 0
            rows.append((i + 1, g[i], c[i], run, int(run >= 3), 0, 0.0))
        pd.DataFrame(rows, columns=["frame", "gyro_mean", "acc_mean", "run", "stationary", "applied", "vnorm_before"]).to_csv(os.path.join(d, "zupt.csv"), index=False)
        return d
    lo_g, lo_c = np.full(n, 0.005), np.full(n, 0.05)
    hi_g, hi_c = np.full(n, 0.03), np.full(n, 0.30)
    arms = [("fires", mk("fires", lo_g, lo_c)), ("gyrobad", mk("gyrobad", hi_g, lo_c)), ("accbad", mk("accbad", lo_g, hi_c)), ("bothbad", mk("bothbad", hi_g, hi_c))]
    g2 = lo_g.copy(); g2[0::2] = 0.03                      # a fifth arm: alternating, run never reaches 3, inputs 50 % gyro-limited
    params["alt"] = {"stat_last": 100}; json.dump(params, open(os.path.join(t, "p.json"), "w"))
    arms.append(("alt", mk("alt", g2, lo_c)))
    a = argparse.Namespace(arm=["%s=%s" % x for x in arms], params=os.path.join(t, "p.json"), out=os.path.join(t, "out"), gyro_thresh=0.02, acc_thresh=0.15,
                           enter_scans=3, first_frame=1, frame_offset=0)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        run_all(a)
    S = pd.read_csv(os.path.join(a.out, "d7_summary.csv")).set_index("bag")
    F = pd.read_csv(os.path.join(a.out, "d7_frames.csv"))
    bad = []
    if str(S.loc["fires", "first_fire_scan_in_window"]) != "3": bad.append("fires: first fire %s" % S.loc["fires", "first_fire_scan_in_window"])
    if abs(S.loc["fires", "frac_run_short"] - 0.02) > 1e-9 or abs(S.loc["fires", "frac_candidate_but_run_short"] - 0.02) > 1e-9: bad.append("start-up run-short fraction")
    for b, k in (("gyrobad", "gyro"), ("accbad", "acc"), ("bothbad", "both")):
        if S.loc[b, "first_fire_scan_in_window"] != "NEVER" or S.loc[b, "main_blocker"] != k or abs(S.loc[b, "frac_blocker_" + k] - 1.0) > 1e-9: bad.append(b)
    if S.loc["bothbad", "frac_gyro_fails"] != 1.0 or S.loc["gyrobad", "frac_acc_fails"] != 0.0: bad.append("per-input fail fractions")
    if S.loc["alt", "first_fire_scan_in_window"] != "NEVER" or abs(S.loc["alt", "frac_blocker_gyro"] - 0.5) > 1e-9 or S.loc["alt", "max_run_in_window"] != 1: bad.append("alt arm")
    if (S.logged_flag_mismatch_whole_file != 0).any(): bad.append("recomputed flag disagrees with logged at default thresholds")
    if len(F[F.bag == "fires"]) != 100 or F.scan.max() != 100: bad.append("window length / stat_last cut")
    # thresholds are honoured: raising the gyro threshold lets gyrobad fire, and the mismatch with the logged flag is then reported
    a.gyro_thresh = 0.04; a.out = os.path.join(t, "out2")
    with contextlib.redirect_stdout(io.StringIO()):
        run_all(a)
    S2 = pd.read_csv(os.path.join(a.out, "d7_summary.csv")).set_index("bag")
    if str(S2.loc["gyrobad", "first_fire_scan_in_window"]) != "3" or S2.loc["gyrobad", "logged_flag_mismatch_whole_file"] == 0: bad.append("threshold arg not honoured")
    # refusals: missing file, missing stat_last, missing column
    for bad_case in ("nofile", "noparam", "nocol"):
        b = argparse.Namespace(**vars(a)); b.out = os.path.join(t, "out3")
        if bad_case == "nofile": b.arm = ["fires=" + os.path.join(t, "nonexistent")]
        elif bad_case == "noparam": b.arm = ["zzz=" + arms[0][1]]
        else:
            d = os.path.join(t, "nc"); os.makedirs(d)
            pd.read_csv(os.path.join(arms[0][1], "zupt.csv")).drop(columns=["acc_mean"]).to_csv(os.path.join(d, "zupt.csv"), index=False)
            b.arm = ["fires=" + d]
        try:
            run_all(b); bad.append("not refused: " + bad_case)
        except SystemExit:
            pass
    print("SELFTEST", "FAIL: " + "; ".join(bad) if bad else "PASS")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", action="append", default=[], help="bag=<arm dir holding zupt.csv>; repeat per bag")
    ap.add_argument("--params", help="bag_params.json (stat_last per bag)")
    ap.add_argument("--out")
    ap.add_argument("--gyro-thresh", type=float, default=0.02)
    ap.add_argument("--acc-thresh", type=float, default=0.15)
    ap.add_argument("--enter-scans", type=int, default=3)
    ap.add_argument("--first-frame", type=int, default=1)
    ap.add_argument("--frame-offset", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.arm and a.params and a.out):
        ap.error("--arm, --params and --out are required")
    return run_all(a)


if __name__ == "__main__":
    sys.exit(main())
