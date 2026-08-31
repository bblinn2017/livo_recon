#!/usr/bin/env python3
"""Synthetic fixtures with a KNOWN Q/R mismatch, to check the diagnosis works.

A real scalar KF is run: the true state evolves with q_true, the filter believes
q_mod / r_mod. Correspondences within a scan share the true prior error, which is
the correlation structure a real LIO scan match has.
"""
import numpy as np, pandas as pd, sys

def make(case, K=1500, n_corr=400, seed=1):
    rng = np.random.default_rng(seed)
    q_true, r_true = 1e-4, 4e-4
    q_mod, r_mod = q_true, r_true
    if case == "q_low":  q_mod = q_true / 25
    if case == "q_high": q_mod = q_true * 25
    if case == "r_low":  r_mod = r_true / 4

    t = np.arange(K) * 0.1
    exc = 0.4 + 1.2 * (1 + np.sin(2 * np.pi * t / 40)) / 2      # motion, in [0.4, 1.6]

    x = 0.0            # true state
    xh = 0.0           # filter mean
    P = r_true         # filter covariance
    rows, srows = [], []
    for k in range(K):
        s = exc[k] ** 2
        x += rng.normal(0, np.sqrt(q_true * s))                 # true propagation
        xh_m, Pm = xh, P + q_mod * s                            # filter propagation
        e = x - xh_m                                            # TRUE prior error
        z = x + rng.normal(0, np.sqrt(r_true), n_corr)
        nu = z - xh_m
        S = Pm + r_mod
        gated = (np.abs(nu) > 3 * np.sqrt(S)).astype(int)
        for nu_i, g in zip(nu, gated):
            rows.append((k, nu_i, S, r_mod, 0.6 * Pm, 0.4 * Pm, 0.0, g))
        Rn = r_mod / n_corr
        Kg = Pm / (Pm + Rn)
        xh = xh_m + Kg * (z.mean() - xh_m)
        P = (1 - Kg) * Pm
        srows.append((k, t[k], 0.1, P, q_mod * s, exc[k], exc[k] * 0.7, e))
    corr = pd.DataFrame(rows, columns=["scan_id","nu","S","S_sensor","S_plane_tilt","S_plane_d","S_pose","gated"])
    corr["range"] = np.abs(rng.normal(15, 6, len(corr)))
    corr["aniso"] = np.exp(rng.normal(1.0, 0.8, len(corr)))
    scan = pd.DataFrame(srows, columns=["scan_id","t","dt","trP_vel","trQdt_vel","omega_norm","acc_norm","true_err"])
    return corr, scan

if __name__ == "__main__":
    case = sys.argv[1]
    c, s = make(case)
    c.to_csv(f"corr_{case}.csv", index=False); s.drop(columns=["true_err"]).to_csv(f"scan_{case}.csv", index=False)
    print(f"{case:8s} rows={len(c):,}  scans={len(s)}  rms_true_prior_err={np.sqrt((s.true_err**2).mean()):.3e}")
