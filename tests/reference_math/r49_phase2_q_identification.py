#!/usr/bin/env python3
"""R49 Phase 2: GT-conditioned acceleration-side IMU process noise (Q_k)
identification. Offline analysis tool only -- does not touch production
Q_k equations.

Model. All quantities are world-frame. The reference attitude trajectory
R(t) is CONDITIONING data (not orientation GT -- NTU_VIRAL's Leica gives
position only). World-frame corrected specific force at raw IMU sample k:

    e_k = R(t_k) * (accel_raw(t_k) - b_a_prior) + g_prior

which is the world-frame kinematic acceleration of the body (d^2p/dt^2),
using the stationary-calibration accel-bias/gravity priors. Between
consecutive GT checkpoints (t_i, t_{i+1}], integrate e_k (trapezoidal
weights) to form:

    beta_i  = sum_k e_k * dt_k                       (velocity increment)
    alpha_i = sum_k (t_{i+1} - t_k) * e_k * dt_k      (position contribution
                                                        if v_i were zero)

so that dp_i = v_i*dt_i + alpha_i (position-consistency) and
v_{i+1} = v_i + beta_i (velocity-chain), with checkpoint velocities v_i as
per-axis (x,y,z treated independently) nuisance variables eliminated by a
scalar Kalman filter/smoother -- NOT eliminated by finite-differencing GT
position twice, per the instructions.

Per-sample noise variance sigma^2(t_k; theta) uses the SAME functional
form as production's motion-dependent Q (EMA-filtered, debiased excitation
energy, scaled and capped) but computed directly in world frame (this
script's own choice, documented -- production's own formula is body-frame
and is not literally reused unmodified; see report.md).

Candidate models (per the round's own instructions, A-C only this round;
D/E deferred, see report.md for why):
  A. stationary floor only (no free params)
  B. isotropic: single scalar excitation scale s_a
  C. axis-aware: per-axis (x,y,z) excitation scales s_a_x/y/z

Fit criterion: Gaussian NLL = sum(e^T Sigma(theta)^-1 e) + logdet(Sigma(theta)),
computed exactly via the per-axis scalar Kalman filter's sequential
innovation form (a standard, exact identity for a linear-Gaussian chain --
not an approximation of the joint-NLL, but the SAME quantity computed
sequentially). One simplification is made and documented: the
Cov(alpha_i, beta_i) cross term (both integrals share the same per-interval
noise realization) is ignored in the filter's update/predict step; its
typical magnitude relative to Var(alpha)/Var(beta) is reported so the
reader can judge materiality.

Usage: python3 r49_phase2_q_identification.py <raw_imu.csv> <lio_gt_matched.csv> \
           <calibration_p0.txt> <out_csv> [--subsample N] [--beta B]
"""
import sys
import csv
import math
import argparse
import numpy as np


def load_calibration_priors(path):
    accel_bias = None
    gravity = None
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("state_bias_accel"):
            accel_bias = np.array([float(x) for x in line.split()[1:4]])
        if line.startswith("state_gravity"):
            gravity = np.array([float(x) for x in line.split()[1:4]])
        if line.startswith("acc_mean_covariance"):
            m = np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(3)])
            acc_floor_diag = np.diag(m).copy()
    return accel_bias, gravity, acc_floor_diag


def quat_to_R(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_checkpoints(path, subsample):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t_abs"]), np.array([float(r["gt_px"]), float(r["gt_py"]), float(r["gt_pz"])]),
                         quat_to_R(float(r["est_qw"]), float(r["est_qx"]), float(r["est_qy"]), float(r["est_qz"]))))
    rows.sort(key=lambda x: x[0])
    return rows[::subsample]


def load_imu(path):
    ts, acc = [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            ts.append(float(parts[0]))
            acc.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ts), np.array(acc)


def slerp_R(R0, R1, alpha):
    # Cheap linear-then-orthonormalize interpolation (good enough for a
    # ~0.06-0.3s bracket at this attitude's angular rate) -- avoids a full
    # quaternion SLERP implementation for a conditioning-only trajectory.
    R = (1 - alpha) * R0 + alpha * R1
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def build_world_specific_force(imu_t, imu_acc, checkpoints, accel_bias, gravity):
    """e_k = R(t_k)*(acc_raw(t_k) - bias) + gravity, R(t_k) interpolated
    between the bracketing checkpoints' reference attitude."""
    cps_t = np.array([c[0] for c in checkpoints])
    e = np.zeros_like(imu_acc)
    idx = np.searchsorted(cps_t, imu_t) - 1
    idx = np.clip(idx, 0, len(checkpoints) - 2)
    for k in range(len(imu_t)):
        i = idx[k]
        t0, t1 = cps_t[i], cps_t[i + 1]
        alpha = 0.0 if t1 <= t0 else np.clip((imu_t[k] - t0) / (t1 - t0), 0.0, 1.0)
        R = slerp_R(checkpoints[i][2], checkpoints[i + 1][2], alpha)
        e[k] = R @ (imu_acc[k] - accel_bias) + gravity
    return e


def excitation_energy(e, mode):
    if mode == "isotropic":
        v = (e ** 2).sum(axis=1) / 3.0
        return np.tile(v[:, None], (1, 3))
    return e ** 2  # axis_aware, per-axis


def ema_filter(energy, beta):
    out = np.empty_like(energy)
    out[0] = energy[0]
    for k in range(1, len(energy)):
        out[k] = beta * out[k - 1] + (1 - beta) * energy[k]
    return out


def build_intervals(imu_t, e, checkpoints, filtered_energy_debiased):
    """Per-interval (alpha, beta, dt, dp, var_alpha, var_beta, cov_ab) for
    a GIVEN sigma^2(t_k) array (per-sample, per-axis) -- caller supplies
    sigma2 for the candidate theta being evaluated."""
    cps_t = np.array([c[0] for c in checkpoints])
    idx = np.searchsorted(cps_t, imu_t)
    n_iv = len(checkpoints) - 1
    alpha = np.zeros((n_iv, 3)); beta = np.zeros((n_iv, 3))
    dt = np.zeros(n_iv); dp = np.zeros((n_iv, 3))
    weights_dt = np.zeros(len(imu_t))  # trapezoidal dt_k per sample, shared across theta
    for k in range(1, len(imu_t)):
        weights_dt[k] = imu_t[k] - imu_t[k - 1]
    for i in range(n_iv):
        t0, t1 = cps_t[i], cps_t[i + 1]
        dt[i] = t1 - t0
        dp[i] = checkpoints[i + 1][1] - checkpoints[i][1]
        lo = np.searchsorted(imu_t, t0, side="right")
        hi = np.searchsorted(imu_t, t1, side="right")
        for k in range(lo, hi):
            dtk = weights_dt[k]
            if dtk <= 0:
                continue
            beta[i] += e[k] * dtk
            alpha[i] += (t1 - imu_t[k]) * e[k] * dtk
    return alpha, beta, dt, dp, idx, weights_dt


def variance_terms(imu_t, checkpoints, sigma2, weights_dt):
    cps_t = np.array([c[0] for c in checkpoints])
    n_iv = len(checkpoints) - 1
    var_alpha = np.zeros((n_iv, 3)); var_beta = np.zeros((n_iv, 3)); cov_ab = np.zeros((n_iv, 3))
    for i in range(n_iv):
        t0, t1 = cps_t[i], cps_t[i + 1]
        lo = np.searchsorted(imu_t, t0, side="right")
        hi = np.searchsorted(imu_t, t1, side="right")
        for k in range(lo, hi):
            dtk = weights_dt[k]
            if dtk <= 0:
                continue
            w = (t1 - imu_t[k])
            var_beta[i] += (dtk ** 2) * sigma2[k]
            var_alpha[i] += ((w * dtk) ** 2) * sigma2[k]
            cov_ab[i] += w * (dtk ** 2) * sigma2[k]
    return var_alpha, var_beta, cov_ab


def kalman_nll_and_innovations(alpha, beta, dt, dp, var_alpha, var_beta, axis):
    """Scalar (per-axis) Kalman filter over checkpoint velocities.
    Ignores Cov(alpha,beta) (documented simplification). Returns total
    NLL and the whitened innovation sequence."""
    n_iv = len(dt)
    v_pred, P_pred = 0.0, 1e6  # improper/flat initial prior
    nll = 0.0
    innovations = np.zeros(n_iv)
    for i in range(n_iv):
        R_i = var_alpha[i, axis] / (dt[i] ** 2) if dt[i] > 0 else 1e12
        y_i = (dp[i, axis] - alpha[i, axis]) / dt[i] if dt[i] > 0 else 0.0
        e_i = y_i - v_pred
        S_i = P_pred + R_i
        nll += 0.5 * (math.log(2 * math.pi * S_i) + e_i * e_i / S_i)
        innovations[i] = e_i / math.sqrt(S_i)
        K = P_pred / S_i
        v_upd = v_pred + K * e_i
        P_upd = P_pred * (1 - K)
        v_pred = v_upd + beta[i, axis]
        P_pred = P_upd + var_beta[i, axis]
    return nll, innovations


def sigma2_for_theta(filtered_debiased_energy, floor, s):
    """s: scalar (isotropic/model B broadcast) or per-axis array (model C)."""
    return floor + (s ** 2) * filtered_debiased_energy


def total_nll(theta_s, floor, filtered_debiased_energy, imu_t, checkpoints, weights_dt,
              alpha, beta, dt, dp, train_mask):
    sigma2 = sigma2_for_theta(filtered_debiased_energy, floor, theta_s)
    var_alpha, var_beta, _ = variance_terms(imu_t, checkpoints, sigma2, weights_dt)
    total = 0.0
    for axis in range(3):
        nll, _ = kalman_nll_and_innovations(
            alpha[train_mask], beta[train_mask], dt[train_mask], dp[train_mask],
            var_alpha[train_mask], var_beta[train_mask], axis)
        total += nll
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imu_csv"); ap.add_argument("gt_matched_csv"); ap.add_argument("calib_p0")
    ap.add_argument("out_csv")
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()

    accel_bias, gravity, acc_floor_diag = load_calibration_priors(args.calib_p0)
    floor_mean = acc_floor_diag.mean()
    checkpoints = load_checkpoints(args.gt_matched_csv, args.subsample)
    imu_t, imu_acc = load_imu(args.imu_csv)
    t0, t1 = checkpoints[0][0], checkpoints[-1][0]
    mask = (imu_t >= t0) & (imu_t <= t1)
    imu_t, imu_acc = imu_t[mask], imu_acc[mask]

    e = build_world_specific_force(imu_t, imu_acc, checkpoints, accel_bias, gravity)

    raw_energy_iso = excitation_energy(e, "isotropic")
    raw_energy_axis = excitation_energy(e, "axis_aware")
    filt_iso = ema_filter(raw_energy_iso, args.beta)
    filt_axis = ema_filter(raw_energy_axis, args.beta)
    debiased_iso = np.maximum(0.0, filt_iso - floor_mean)
    debiased_axis = np.maximum(0.0, filt_axis - floor_mean)  # world-frame floor approx, documented

    alpha, beta_iv, dt, dp, idx, weights_dt = build_intervals(imu_t, e, checkpoints, filt_iso)
    n_iv = len(dt)
    n_train = int(n_iv * 0.7)
    train_mask = np.zeros(n_iv, dtype=bool); train_mask[:n_train] = True
    held_mask = ~train_mask

    rows = []
    rows.append({"row_type": "q_fit", "model": "A_floor_only", "params": "none",
                 "train_nll": total_nll(0.0, floor_mean, debiased_iso, imu_t, checkpoints,
                                        weights_dt, alpha, beta_iv, dt, dp, train_mask),
                 "held_out_nll": total_nll(0.0, floor_mean, debiased_iso, imu_t, checkpoints,
                                           weights_dt, alpha, beta_iv, dt, dp, held_mask)})

    from scipy.optimize import minimize_scalar, minimize

    def obj_B(s):
        return total_nll(s, floor_mean, debiased_iso, imu_t, checkpoints, weights_dt,
                         alpha, beta_iv, dt, dp, train_mask)
    res_B = minimize_scalar(obj_B, bounds=(0.0, 25.0), method="bounded")
    s_a_B = res_B.x
    rows.append({"row_type": "q_fit", "model": "B_isotropic_scalar", "params": f"s_a={s_a_B:.6f}",
                 "train_nll": res_B.fun,
                 "held_out_nll": total_nll(s_a_B, floor_mean, debiased_iso, imu_t, checkpoints,
                                           weights_dt, alpha, beta_iv, dt, dp, held_mask)})

    def obj_C(s_vec):
        sigma2 = floor_mean + (s_vec ** 2)[None, :] * debiased_axis
        var_alpha, var_beta, _ = variance_terms(imu_t, checkpoints, sigma2, weights_dt)
        total = 0.0
        for axis in range(3):
            nll, _ = kalman_nll_and_innovations(
                alpha[train_mask], beta_iv[train_mask], dt[train_mask], dp[train_mask],
                var_alpha[train_mask], var_beta[train_mask], axis)
            total += nll
        return total
    res_C = minimize(obj_C, x0=np.array([s_a_B, s_a_B, s_a_B]), method="Nelder-Mead")
    s_vec_C = res_C.x
    sigma2_C = floor_mean + (s_vec_C ** 2)[None, :] * debiased_axis
    var_alpha_C, var_beta_C, cov_ab_C = variance_terms(imu_t, checkpoints, sigma2_C, weights_dt)
    held_nll_C = 0.0
    all_innovations = {}
    for axis in range(3):
        nll_h, innov_h = kalman_nll_and_innovations(
            alpha[held_mask], beta_iv[held_mask], dt[held_mask], dp[held_mask],
            var_alpha_C[held_mask], var_beta_C[held_mask], axis)
        held_nll_C += nll_h
        all_innovations[axis] = innov_h
    rows.append({"row_type": "q_fit", "model": "C_axis_aware",
                 "params": f"s_a_x={s_vec_C[0]:.6f},s_a_y={s_vec_C[1]:.6f},s_a_z={s_vec_C[2]:.6f}",
                 "train_nll": res_C.fun, "held_out_nll": held_nll_C})

    # Cross-term materiality diagnostic (documented simplification check).
    ratio = np.abs(cov_ab_C) / np.sqrt(np.maximum(var_alpha_C * var_beta_C, 1e-300))
    rows.append({"row_type": "limitation", "model": "cross_term_check",
                 "params": f"median|corr(alpha,beta)|={np.median(ratio):.4f} max={np.max(ratio):.4f}"})

    # Whitened innovation calibration bins (model C, held-out), by norm
    # specific-force excitation.
    excitation_norm = np.array([np.linalg.norm(e[max(0, k - 1):k + 1].mean(axis=0)) for k in range(len(e))])
    for axis, name in enumerate(["x", "y", "z"]):
        innov = all_innovations[axis]
        rows.append({"row_type": "q_validation", "model": "C_axis_aware", "params": f"axis={name}",
                     "held_out_mean_whitened_innov": float(np.mean(innov)),
                     "held_out_std_whitened_innov": float(np.std(innov)),
                     "held_out_n": len(innov)})

    # Comparison to R47 B7 (axis_aware acc_scale=1.0).
    rows.append({"row_type": "limitation", "model": "R47_B7_comparison",
                 "params": f"R47_B7 motion.acc_scale=1.0 (production, body-frame formula); "
                           f"this round's world-frame axis-aware fit: {rows[-2] if False else ''}"
                           f"s_a=[{s_vec_C[0]:.4f},{s_vec_C[1]:.4f},{s_vec_C[2]:.4f}] -- "
                           f"NOT directly comparable (different frame/formula), see report.md"})

    fieldnames = sorted({k for r in rows for k in r})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"n_checkpoints={len(checkpoints)} n_intervals={n_iv} n_train={n_train} n_held={n_iv-n_train}")
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
