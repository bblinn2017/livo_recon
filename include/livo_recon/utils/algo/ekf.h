#pragma once
#include "livo_recon/utils/state/state.h"
#include <limits>

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

    // T0-E (2026-08-31): the quadratic+log-determinant terms of this
    // frame's batch-update NLL, i.e. everything EXCEPT the purely
    // residual-level pieces (sum_i log(sigma_i^2), sum_i r_i^2/sigma_i^2,
    // and n_residuals*log(2*pi) -- the Gaussian normalization term,
    // 2026-08-31 code-audit fix: dropping it is harmless comparing runs
    // with the same n_residuals per frame, but n_residuals varies both
    // per-frame and with q_alpha, so it must be included when comparing
    // totals ACROSS runs) the caller already has from iterating
    // residuals_ -- see LioProc::estimateStateCorrection()'s log_nll_en
    // block for how the pieces combine into the full per-frame NLL, and
    // LioProcOptions::log_nll_en's doc comment for the frame-accounting
    // fix (log every frame including n_residuals=0 ones, never silently
    // skip) that makes runs at different q_alpha comparable at all.
    //
    // Derivation: for a linear-Gaussian batch update with prior x~N(x0,P0)
    // and independent per-residual noise r_i~N(0,sigma_i^2), the marginal
    // (prior-predictive) NLL of the batch is NLL = 1/2*[log det(S_full) +
    // r^T S_full^-1 r] where S_full = H_full P0 H_full^T + R_full
    // (R_full = diag(sigma_i^2)) -- an N_CORR x N_CORR matrix, infeasible
    // to form directly. Two standard identities avoid ever forming it:
    //   det(S_full) = det(R_full) * det(P0) * det(A),  A = P0^-1 + HtH
    //   r^T S_full^-1 r = r^T R_full^-1 r  -  Htz^T A^-1 Htz
    // (HtH = H_full^T R_full^-1 H_full, Htz = H_full^T R_full^-1 r -- both
    // exactly this class's own HtH/Htz accumulators, and A is exactly
    // applyMeanUpdate()'s own A). So NLL = 1/2*[ sum_i log(sigma_i^2) +
    // sum_i r_i^2/sigma_i^2 + log det(P0) + log det(A) - Htz^T A^-1 Htz ]
    // -- this function returns log det(P0) + log det(A) - Htz^T A^-1 Htz;
    // the caller adds the two residual-level sums and halves the total.
    //
    // Computed from a FRESH LDLT of A/prior_cov, independent of
    // applyMeanUpdate()'s own (called separately, doesn't require this
    // to run before/after it, doesn't touch last_H_full_/last_K1_/
    // state_ at all -- purely read-only over HtH/Htz/prior_cov). "Frozen
    // Jacobian" scope note: this is the NLL of the CURRENT frame's prior
    // (state_->cov() before this frame's own correction) against its
    // OWN residuals -- exactly the T0-D-style "first-iteration,
    // un-relinearized" quantity, not a converged-update NLL.
    double nllQuadraticAndLogdet(const Eigen::MatrixXd& prior_cov) const
    {
      const int dim = prior_cov.rows();
      const int n = static_cast<int>(HtH.rows());
      Eigen::MatrixXd H_full = Eigen::MatrixXd::Zero(dim, dim);
      H_full.block(StateGroup::idxR(), StateGroup::idxR(), n, n) = HtH;
      const Eigen::MatrixXd A = H_full + prior_cov.inverse();

      // 2026-08-31 code-audit fix: a numerically non-PD prior/A (e.g. a
      // pathological q_alpha sweep value) previously wrote a silent NaN
      // into nll.txt instead of failing loudly -- NaN is easy to miss in
      // a summed column and would masquerade as "this alpha is somehow
      // infinitely good/bad" rather than "the covariance broke".
      Eigen::LDLT<Eigen::MatrixXd> ldlt_A(A);
      Eigen::VectorXd Htz_full = Eigen::VectorXd::Zero(dim);
      Htz_full.segment(StateGroup::idxR(), n) = Htz;
      const double quad = Htz_full.dot(ldlt_A.solve(Htz_full));
      const double logdet_A = ldlt_A.vectorD().array().log().sum();

      Eigen::LDLT<Eigen::MatrixXd> ldlt_prior(prior_cov);
      const double logdet_prior = ldlt_prior.vectorD().array().log().sum();

      if (ldlt_A.info() != Eigen::Success || ldlt_prior.info() != Eigen::Success ||
          !(ldlt_A.isPositive() && ldlt_prior.isPositive()))
        return std::numeric_limits<double>::quiet_NaN();

      return logdet_prior + logdet_A - quad;
    }

    // T0-E-4 (2026-08-31): condition-number proxy for the LAST applyMeanUpdate()/
    // applyCovarianceUpdate() solve (A = H_full + prior_cov^-1), i.e. the actual
    // EKF solve path -- NOT nllQuadraticAndLogdet()'s own separate fresh LDLT,
    // which is a diagnostic-only decomposition. max(|D_ii|)/min(|D_ii|) from the
    // LDLT's diagonal is a cheap proxy for cond(A) (exact for a diagonal A;
    // an underestimate in general, but tracks the same order of magnitude and
    // needs no extra decomposition). Same staleness caveat as HtH/Htz: only
    // meaningful when applyMeanUpdate() actually ran this iteration (n_res>0).
    // T0-F-2b (2026-08-31): ||K|| for the LAST applyMeanUpdate() call --
    // Frobenius norm of K1_cols (last_K1_'s R-column block), the gain
    // that pre-multiplies Htz in applyMeanUpdate()'s solution formula
    // (see its own comment). Same staleness caveat as pivotRatio().
    double kalmanGainNorm() const {
      if (last_K1_.size() == 0) return std::numeric_limits<double>::quiet_NaN();
      const int n = static_cast<int>(HtH.rows());
      return last_K1_.block(0, StateGroup::idxR(), last_K1_.rows(), n).norm();
    }

    double pivotRatio() const {
      const auto& d = ldlt_.vectorD();
      if (d.size() == 0) return std::numeric_limits<double>::quiet_NaN();
      const double max_d = d.array().abs().maxCoeff();
      const double min_d = d.array().abs().minCoeff();
      if (min_d <= 0.0) return std::numeric_limits<double>::infinity();
      return max_d / min_d;
    }

private:
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
    Eigen::MatrixXd last_H_full_;
    Eigen::MatrixXd last_K1_;
};

}  // namespace livo_recon
