#!/usr/bin/env python3
"""Checks VibrationCanceller output (vib_apply) against its input.
usage: validate.py RAW CLEAN [--t0 STAMP] [--still A B]... [--moving A B]...
  times in seconds from --t0 (default: the first IMU stamp; pass the first image stamp for 'frames 1-5' = 0-0.5 s)
Prints per window: per-sample variance raw -> cleaned (acc e-3 m^2/s^4, gyro e-6 rad^2/s^2), the 26-30.2 Hz band rms
ratio cleaned/raw (the main line and its sidebands; < 1 removed, > 1 added), the mean shift, and for moving windows the
power below 10 Hz cleaned/raw (the motion; must be 1). Line/neighbour PSD ratio after cleaning: ~1 = removed down to
the surrounding noise, < 1 = noise removed too (overfitting), > 1 = line left."""
import argparse
import numpy as np
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("clean")
ap.add_argument("--t0", type=float, default=None)
ap.add_argument("--still", nargs=2, type=float, action="append", default=[])
ap.add_argument("--moving", nargs=2, type=float, action="append", default=[])
ap.add_argument("--fs", type=float, default=199.998)
ap.add_argument("--lines", type=float, nargs="*", default=[28.025, 27.235, 28.815, 56.05, 13.40, 18.124])
a = ap.parse_args()
R = np.loadtxt(a.raw); C = np.loadtxt(a.clean)
assert len(R) == len(C), "raw and cleaned differ in length"
t = R[:, 0] - (a.t0 if a.t0 is not None else R[0, 0])
x = np.hstack([R[:, 4:7] * 9.81, R[:, 1:4]]); y = np.hstack([C[:, 4:7] * 9.81, C[:, 1:4]])
bb, aa = signal.butter(4, [26.0, 30.2], "bandpass", fs=a.fs)
xb = signal.filtfilt(bb, aa, x, axis=0); yb = signal.filtfilt(bb, aa, y, axis=0)
f3 = lambda v, s=1.0: "[" + " ".join("%.3f" % (u * s) for u in v) + "]"
for kind, wins in (("still", a.still), ("moving", a.moving)):
    for w0, w1 in wins:
        m = (t >= w0) & (t < w1)
        if m.sum() < 20:
            print("%s %.1f-%.1f s: too few samples" % (kind, w0, w1)); continue
        vx, vy = x[m].var(0), y[m].var(0)
        print("%s %.1f-%.1f s (%d samples)" % (kind, w0, w1, m.sum()))
        print("  variance acc %se-3 -> %se-3   gyro %se-6 -> %se-6" % (f3(vx[:3], 1e3), f3(vy[:3], 1e3), f3(vx[3:], 1e6), f3(vy[3:], 1e6)))
        print("  26-30 Hz band rms cleaned/raw  acc %s gyro %s" % (f3(np.sqrt((yb[m] ** 2).mean(0) / (xb[m] ** 2).mean(0))[:3]),
                                                                  f3(np.sqrt((yb[m] ** 2).mean(0) / (xb[m] ** 2).mean(0))[3:])))
        print("  mean shift acc %s mm/s^2" % f3(y[m, :3].mean(0) - x[m, :3].mean(0), 1e3))
        if kind == "moving":
            f, Px = signal.welch(x[m], fs=a.fs, nperseg=min(1024, m.sum()), axis=0)
            _, Py = signal.welch(y[m], fs=a.fs, nperseg=min(1024, m.sum()), axis=0)
            lo = f < 10
            print("  power below 10 Hz cleaned/raw %s" % f3(Py[lo].sum(0) / Px[lo].sum(0)))
        if m.sum() >= 1024:
            f, Py = signal.welch(y[m] - y[m].mean(0), fs=a.fs, nperseg=1024, axis=0)
            out = []
            for L in a.lines:
                at = np.abs(f - L) < 0.3; nb = (np.abs(f - L) > 1.2) & (np.abs(f - L) < 3.0)
                for L2 in a.lines: nb &= np.abs(f - L2) > 1.2
                out.append("%.2f:%.2f" % (L, np.median(Py[at].mean(0) / np.median(Py[nb], 0))))
            print("  line/neighbour PSD after: " + " ".join(out))
