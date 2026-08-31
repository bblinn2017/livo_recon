#!/usr/bin/env python3
"""
Q tuning by gradient, not by grid.

Parameterise Q = alpha * Q0 and treat the filter's own negative log-likelihood of
the innovation sequence as the objective:

    NLL(a) = 1/2 sum_k [ log det S_k(a) + nu_k^T S_k(a)^-1 nu_k ]

The gradient is EXACT from a single run if the filter also propagates its own
sensitivities alongside the state and covariance:

    dxm/da = Phi dxp/da
    dPm/da = Phi (dPp/da) Phi^T + Q0 dt
    dK /da = (dPm/da) H^T Sinv  -  K (H dPm/da H^T) Sinv
    dxp/da = (I - KH) dxm/da + (dK/da) nu
    dPp/da = (I - KH)(dPm/da) - (dK/da) H Pm

    dS/da  = H (dPm/da) H^T          dnu/da = -H dxm/da
    dNLL/da = 1/2 sum_k [ tr(Sinv dS) - nu^T Sinv dS Sinv nu + 2 (dnu)^T Sinv nu ]

Cost: one extra state vector and one extra covariance matrix per parameter.
Then secant/Newton on alpha. This scalar case is the correctness proof: the
gradient is checked against finite differences and the search recovers a known
mis-tuning that a grid would need ~10 runs per axis to bracket.
"""
import numpy as np

Q_TRUE, R_TRUE = 1e-4, 4e-4
K_SCANS, N_CORR = 1500, 400


def bag(seed=1):
    """One fixed 'bag': the true states and raw measurements, replayed for every alpha."""
    rng = np.random.default_rng(seed)
    t = np.arange(K_SCANS) * 0.1
    exc = 0.4 + 1.2 * (1 + np.sin(2 * np.pi * t / 40)) / 2
    x, xs, zs = 0.0, [], []
    for k in range(K_SCANS):
        x += rng.normal(0, np.sqrt(Q_TRUE * exc[k] ** 2))
        xs.append(x)
        zs.append(x + rng.normal(0, np.sqrt(R_TRUE), N_CORR))
    return exc, np.array(xs), np.array(zs)


def run(alpha, exc, zs, q0, r_mod=R_TRUE, sens=True):
    """Filter at Q = alpha*q0; returns NLL and (optionally) the exact dNLL/dalpha."""
    xh, P, dxh, dP = 0.0, R_TRUE, 0.0, 0.0
    nll, g = 0.0, 0.0
    for k in range(K_SCANS):
        s = exc[k] ** 2
        Pm = P + alpha * q0 * s                    # H = 1, Phi = 1 (scalar RW)
        dPm = dP + q0 * s
        xm, dxm = xh, dxh
        z = zs[k]
        nu = z - xm
        S = Pm + r_mod                             # per-correspondence innovation cov
        nll += 0.5 * (N_CORR * np.log(S) + np.sum(nu ** 2) / S)
        if sens:
            dS, dnu = dPm, -dxm
            g += 0.5 * (N_CORR * dS / S - np.sum(nu ** 2) * dS / S ** 2
                        + 2 * dnu * np.sum(nu) / S)
        Rn = r_mod / N_CORR                        # batch update
        Kg = Pm / (Pm + Rn)
        dKg = dPm / (Pm + Rn) - Pm * dPm / (Pm + Rn) ** 2
        nub = z.mean() - xm
        xh = xm + Kg * nub
        dxh = (1 - Kg) * dxm + dKg * nub
        P = (1 - Kg) * Pm
        dP = -dKg * Pm + (1 - Kg) * dPm
    return (nll, g) if sens else nll


def main():
    exc, xs, zs = bag()
    q0 = Q_TRUE / 25.0                     # the shipped, 25x-too-small value: alpha* should be ~25

    print("gradient check (analytic vs central finite difference)")
    for a in (1.0, 5.0, 25.0, 60.0):
        _, g = run(a, exc, zs, q0)
        h = a * 1e-5
        fd = (run(a + h, exc, zs, q0, sens=False) - run(a - h, exc, zs, q0, sens=False)) / (2 * h)
        print(f"  alpha={a:6.2f}   analytic {g:+12.4f}   fd {fd:+12.4f}   rel.err {abs(g-fd)/max(abs(fd),1e-9):.2e}")

    print("\nsecant search on dNLL/dalpha = 0, starting from the shipped value")
    a0, a1 = 1.0, 4.0
    _, g0 = run(a0, exc, zs, q0)
    _, g1 = run(a1, exc, zs, q0)
    print(f"  run 1: alpha={a0:8.3f}  grad {g0:+11.3f}")
    print(f"  run 2: alpha={a1:8.3f}  grad {g1:+11.3f}")
    for i in range(6):
        a2 = a1 - g1 * (a1 - a0) / (g1 - g0)
        a2 = float(np.clip(a2, 0.05, 1e4))
        _, g2 = run(a2, exc, zs, q0)
        print(f"  run {i+3}: alpha={a2:8.3f}  grad {g2:+11.3f}")
        a0, g0, a1, g1 = a1, g1, a2, g2
        if abs(g1) < 1e-3:
            break
    print(f"\n  converged alpha*      = {a1:.3f}   (Q* = {a1*q0:.3e}, true Q = {Q_TRUE:.3e})")
    print(f"  recovered Q / true Q  = {a1*q0/Q_TRUE:.3f}")
    grid = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    print(f"  a grid would have needed {len(grid)} runs to bracket this on ONE axis;")
    print(f"  the secant took {i+3}. On three axes: {len(grid)**3} vs ~{3*(i+3)}.")


if __name__ == "__main__":
    main()
