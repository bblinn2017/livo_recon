#pragma once
#include "livo_recon/utils/state/state.h"

namespace livo_recon {

struct EkfUpdate
{
    // 2026-08-24: reverted to fixed 6x6/6x1 (R,P) -- was Eigen::MatrixXd/
    // VectorXd dynamically sized to also support a 12-dim (R,P,V,W) "wide"
    // mode, which existed solely for the now-removed iterative-deskew
    // mechanism's wide_jacobian_vw option (confirmed regressed, never
    // usable). See docs/removed_livo_recon_spline_deskew_2026aug24.md.
    Eigen::Matrix<double, 6, 6> HtH = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> Htz = Eigen::Matrix<double, 6, 1>::Zero();
    int n_meas = 0;

    V3D dtheta = V3D::Zero();
    V3D dt     = V3D::Zero();

    void reset() {
        HtH.setZero();
        Htz.setZero();
        n_meas = 0;
        dtheta = dt = V3D::Zero();
    }

    // Standard iterated-ESKF per-iteration MEAN update. NOT the previous
    // version of this function (applyUpdate), which re-read/rewrote
    // state->covMut() every iteration of a frame's inner loop: iteration 2
    // would treat iteration 1's already-shrunk POSTERIOR as a brand-new
    // prior, absorbing the same (or barely-relinearized) measurement's
    // information again and again -- unboundedly shrinking P the longer the
    // inner loop ran (confirmed via a real ATE regression on the sibling
    // FAST-LIVO2/livo_vio port of this exact file: 50 iterations gave WORSE
    // trajectory accuracy than 20). `prior_cov`/`propagat` are now the SAME
    // fixed frame-entry snapshot for every iteration of one frame's inner
    // loop -- only H/Htz get re-linearized each iteration at the current
    // (already-partially-corrected) state. This does NOT touch
    // state->cov()/covMut() at all; call applyCovarianceUpdate() exactly
    // once, after the whole inner loop finishes, to actually write the
    // posterior.
    //
    // `vec` (state->boxminusFromPropagat(propagat)) is the manifold-
    // consistent (SO(3)-aware) deviation of the fixed prior mean from
    // wherever the current iterate has already drifted to. Absent any
    // measurement evidence (K1@Htz -> 0), `solution -> vec`, i.e. the state
    // reverts exactly to the original prior -- the restoring force this
    // function previously lacked entirely (it only ever applied
    // -A^{-1}*b, with no term at all pulling drifted iterates back toward
    // the true prior mean). See FAST-LIVO2's voxel_map.cpp::StateEstimation
    // (`vec - G*vec`) for the original reference derivation this mirrors.
    void applyMeanUpdate(const StateGroupPtr& state,
                          const Eigen::MatrixXd& prior_cov,
                          const StateGroup& propagat)
    {
        const int dim = state->dimState();
        const int n = static_cast<int>(HtH.rows());  // 6 (R,P,V)
        last_H_full_ = Eigen::MatrixXd::Zero(dim, dim);
        // R,P,V are always CONTIGUOUS starting at idxR()=0, so this single
        // block assignment covers the full n=6 HtH block directly.
        last_H_full_.block(StateGroup::idxR(), StateGroup::idxR(), n, n) = HtH;

        Eigen::MatrixXd A = last_H_full_ + prior_cov.inverse();
        ldlt_.compute(A);
        last_K1_ = ldlt_.solve(Eigen::MatrixXd::Identity(dim, dim));

        // G's only nonzero columns are the first n (H_full is zero outside
        // its top-left nxn block), so G.cols(0,n) = K1.cols(0,n) * HtH
        // exactly.
        const Eigen::MatrixXd K1_cols = last_K1_.block(0, StateGroup::idxR(), dim, n);
        const Eigen::MatrixXd G_cols = K1_cols * HtH;

        const Eigen::VectorXd vec = state->boxminusFromPropagat(propagat);

        // NOTE the leading minus sign on the measurement term -- this
        // class's Htz uses the "+H^T*W*r" convention (see solveSystem()'s
        // own accumulation comment), the opposite of FAST-LIVO2's
        // voxel_map.cpp (whose HTz is built from meas_vec = -dis_to_plane_,
        // i.e. -(H^T*W*r)) that this formula's structure was templated
        // from. Confirmed via a real divergence bug: without this negation
        // the correction becomes positive feedback (median |dt| roughly
        // doubling every iteration, diverging to meters within ~15
        // iterations) instead of converging. G/vec are unaffected (G comes
        // only from HtH, never from Htz/r's sign).
        const Eigen::VectorXd solution = -K1_cols * Htz + vec - G_cols * vec.segment(StateGroup::idxR(), n);

        state->applyDelta(solution);

        dtheta = solution.segment<3>(StateGroup::idxR());
        dt     = solution.segment<3>(StateGroup::idxP());
    }

    // Writes the posterior covariance exactly ONCE, using the SAME fixed
    // prior (prior_cov) applyMeanUpdate() used, blended with the FINAL
    // iteration's H_full (cached by the last applyMeanUpdate() call). Call
    // this once after a frame's whole inner iteration loop finishes
    // (whatever the reason it stopped), not per-iteration.
    void applyCovarianceUpdate(const StateGroupPtr& state, const Eigen::MatrixXd& prior_cov)
    {
        Eigen::MatrixXd A = last_H_full_ + prior_cov.inverse();
        ldlt_.compute(A);
        // Compute P_new = (H + P^{-1})^{-1} = A^{-1} directly, rather than via
        // P -= G*P (G = A^{-1}*H). The latter suffers catastrophic
        // cancellation and yields non-SPD covariances when G approaches
        // identity (a confident measurement) -- see the identical fix/comment
        // in lio_processing.cpp's CPU solveSystem() (now itself just calling
        // through to applyMeanUpdate()/applyCovarianceUpdate()).
        Eigen::MatrixXd P_new = ldlt_.solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
        P_new = 0.5 * (P_new + P_new.transpose());
        state->covMut() = P_new;
    }

private:
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
    Eigen::MatrixXd last_H_full_;
    Eigen::MatrixXd last_K1_;
};

}  // namespace livo_recon
