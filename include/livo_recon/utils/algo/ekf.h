#pragma once
#include "livo_recon/utils/state/state.h"
#include <limits>

namespace livo_recon {

// CQ-57 item 4: ONE COVARIANCE UPDATE, NOT TWO COPIES. The shared "solve
// A^-1 directly, then symmetrize" core both EkfUpdate::applyCovarianceUpdate()
// (decoupled) and LioProcCoupled's own posterior18 construction (coupled)
// need -- computing the inverse directly rather than P -= G*P avoids the
// catastrophic cancellation that yields non-SPD covariances when G
// approaches identity (EkfUpdate::applyCovarianceUpdate()'s own original
// comment on why). When M is non-null, projects A^-1 through it first
// (M*A^-1*M^T is PSD for any M given a PSD A^-1 -- coupled's own use case,
// projecting its larger [delta_s,c] joint solve down to the physical
// state). Callers are responsible for their OWN rank-deficiency guard
// before calling this (see LioProcCoupled's own vectorD().minCoeff() check
// -- NOT added here, since adding it to this shared path would also change
// decoupled's existing behaviour, and item 4's own 4/4 decoupled-md5
// requirement means decoupled's output must stay byte-identical).
inline Eigen::MatrixXd solveCovarianceFromA(const Eigen::MatrixXd& A,
                                             const Eigen::MatrixXd* M = nullptr)
{
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    const Eigen::MatrixXd coeff_cov = ldlt.solve(Eigen::MatrixXd::Identity(A.rows(), A.rows()));
    Eigen::MatrixXd P_new = M ? Eigen::MatrixXd((*M) * coeff_cov * M->transpose()) : coeff_cov;
    P_new = 0.5 * (P_new + P_new.transpose());
    return P_new;
}

struct EkfUpdate
{
    // History (9-13): see docs/livo_recon_changelog.md#include-livo_recon-utils-algo-ekf.h-9
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
        const int n = static_cast<int>(HtH.rows());  // 6 (R,P)
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
    //
    // CQ-70: H_full_discount (default 1.0, the identity) divides ONLY this
    // call's OWN copy of last_H_full_ before it's folded into A -- it never
    // touches last_H_full_ itself (applyMeanUpdate() already ran and is
    // unaffected regardless of call order) or Htz (this update doesn't read
    // Htz at all). A caller that never passes this argument gets EXACTLY
    // the pre-CQ-70 formula, guarded to skip the division entirely at
    // discount==1.0 rather than relying on x/1.0==x -- see
    // solveCovarianceFromA()'s own doc comment for why this shared path's
    // byte-for-byte behavior at the default is load-bearing (decoupled-md5).
    void applyCovarianceUpdate(const StateGroupPtr& state, const Eigen::MatrixXd& prior_cov,
                                double H_full_discount = 1.0)
    {
        Eigen::MatrixXd A = (H_full_discount == 1.0)
            ? Eigen::MatrixXd(last_H_full_ + prior_cov.inverse())
            : Eigen::MatrixXd(last_H_full_ / H_full_discount + prior_cov.inverse());
        // CQ-57 item 4: shared with LioProcCoupled's own posterior18
        // construction -- see solveCovarianceFromA()'s own doc comment.
        // M=nullptr here reproduces the EXACT prior behaviour (P_new = A^-1
        // directly, then symmetrize) -- this class's own ldlt_ member is no
        // longer used for this step (solveCovarianceFromA builds its own
        // local LDLT), which is numerically identical, just no longer
        // caching the factorization on the instance.
        state->covMut() = solveCovarianceFromA(A);
    }

    // History (114-150): see docs/livo_recon_changelog.md#include-livo_recon-utils-algo-ekf.h-114
    double nllQuadraticAndLogdet(const Eigen::MatrixXd& prior_cov) const
    {
      const int dim = prior_cov.rows();
      const int n = static_cast<int>(HtH.rows());
      Eigen::MatrixXd H_full = Eigen::MatrixXd::Zero(dim, dim);
      H_full.block(StateGroup::idxR(), StateGroup::idxR(), n, n) = HtH;
      const Eigen::MatrixXd A = H_full + prior_cov.inverse();

      // History (159-163): see docs/livo_recon_changelog.md#include-livo_recon-utils-algo-ekf.h-159
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

    // History (180-191): see docs/livo_recon_changelog.md#include-livo_recon-utils-algo-ekf.h-180
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
