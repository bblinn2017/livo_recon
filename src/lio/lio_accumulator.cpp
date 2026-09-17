#include "livo_recon/lio/lio_accumulator.h"

namespace livo_recon
{

void accumulateLioResiduals(const std::vector<Residual>& residuals, EkfUpdate& out)
{
  const int n = static_cast<int>(residuals.size());

  out.reset();
  out.n_meas = n;

  // Fully serial, single-pass, single-accumulator reduction in residual-
  // index order -- see VioAccumulator::accumulate()'s matching fix
  // (vio_accumulator.cpp) for the rationale: the result must not depend on
  // OMP thread count (a downstream iterative solve can amplify LSB-level
  // rounding differences from different summation groupings into
  // materially different results for byte-identical input).
  //
  // Two OMP-parallel versions of this function were tried and abandoned
  // 2026-09-17, in order:
  //  (1) a per-RESIDUAL `static thread_local` buffer pair, reused across
  //      calls via resize() to amortize the O(n) allocation -- REVERTED:
  //      reproducibly SIGSEGV'd at node startup, well before this function
  //      was ever first called (bisected to that one declaration; likely a
  //      TLS allocation issue given this binary links both OpenMP and CUDA
  //      translation units -- root cause not chased further).
  //  (2) a per-RESIDUAL (not static/thread_local) O(n) buffer pair, freshly
  //      allocated each call but skipping the Zero() fill (every index
  //      written exactly once via `=`, not accumulated via `+=`) -- worked,
  //      but a fair load-matched A/B (same idle host, binary-only diff)
  //      showed only a ~2% delta on state_estimation/frame -- not worth the
  //      allocation, the two-pass structure, or the O(n) storage at all.
  //
  // This version has neither problem: no static/thread_local (no TLS risk),
  // and no O(n) storage of any kind (not even per-thread) -- each term is
  // computed and immediately added into out.HtH/out.Htz's own blocks in one
  // step, in index order. This is PROVABLY bit-identical to the two-pass
  // parallel-compute-then-serial-sum design above: `acc += term` is the
  // same floating-point operation whether `term` was just computed or was
  // read back from a place it was stored earlier, and the accumulation
  // order/grouping (strict index order, one add per term, no batching) is
  // unchanged -- unlike a per-thread partial-sum design (rejected for the
  // same reason as (1)/(2) above plus this one: grouping by thread chunk,
  // even with a FIXED thread count, changes the rounding versus a flat
  // sequential sum, since floating-point addition is not associative). The
  // tradeoff is no OMP parallelism for computing the n per-residual terms;
  // each term is a handful of 3x3 matrix products (cheap), and this
  // function's own share of frame time (~5ms/frame, per profiling) makes
  // that an acceptable trade for correctness that doesn't need re-verifying
  // every time this function changes.
  for (int i = 0; i < n; ++i) {
    const auto& res = residuals[i];
    const V3D& hr = res.point_cross_normal;
    const V3D hp = res.normal;
    const double w  = 1.0 / res.sigma_squared;
    const double wr = w * res.r;

    out.HtH.block<3,3>(0, 0).noalias() += w  * hr * hr.transpose();
    out.HtH.block<3,3>(0, 3).noalias() += w  * hr * hp.transpose();
    out.HtH.block<3,3>(3, 0).noalias() += w  * hp * hr.transpose();
    out.HtH.block<3,3>(3, 3).noalias() += w  * hp * hp.transpose();
    out.Htz.segment<3>(0).noalias()    += wr * hr;
    out.Htz.segment<3>(3).noalias()    += wr * hp;
  }
}

void accumulateLioResidualsCuda(const std::vector<Residual>& residuals, EkfUpdate& out,
                                 LioCudaBuffers& buf)
{
  const int n = static_cast<int>(residuals.size());
  buf.ensure(n);

  for (int i = 0; i < n; ++i) {
    const auto& r = residuals[i];
    buf.h_normals[i] = make_float3(
      static_cast<float>(r.normal.x()),
      static_cast<float>(r.normal.y()),
      static_cast<float>(r.normal.z()));
    buf.h_crosses[i] = make_float3(
      static_cast<float>(r.point_cross_normal.x()),
      static_cast<float>(r.point_cross_normal.y()),
      static_cast<float>(r.point_cross_normal.z()));
    buf.h_sigma_squared[i] = static_cast<float>(r.sigma_squared);
    buf.h_rs[i]            = static_cast<float>(r.r);
  }

  double HtH_out[36] = {};
  double Htz_out[6]  = {};
  runLioAccumulation(buf, n, HtH_out, Htz_out);

  out.reset();
  out.n_meas = n;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      out.HtH(i, j) = HtH_out[i * 6 + j];
  for (int i = 0; i < 6; ++i)
    out.Htz(i) = Htz_out[i];
}

}  // namespace livo_recon
