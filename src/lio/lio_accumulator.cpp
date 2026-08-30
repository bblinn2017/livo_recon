#include "livo_recon/lio/lio_accumulator.h"
#include "livo_recon/utils/algo/omp_utils.h"

namespace livo_recon
{

void accumulateLioResiduals(const std::vector<Residual>& residuals, EkfUpdate& out)
{
  const int n = static_cast<int>(residuals.size());
  const int nthreads = cappedOmpThreads();

  out.reset();
  out.n_meas = n;

  using M66 = Eigen::Matrix<double, 6, 6>;
  using V6  = Eigen::Matrix<double, 6, 1>;

  // Per-RESIDUAL (not per-thread) accumulators -- see VioAccumulator::
  // accumulate()'s matching fix (vio_accumulator.cpp) for the full
  // rationale: summing per-thread partial sums makes the floating-point
  // result depend on how many OMP threads ran this call (different
  // groupings round differently at the LSB level), which a downstream
  // iterative solve can amplify into materially different results for
  // byte-identical input. Reducing in fixed residual-index order instead
  // makes the result independent of thread count entirely.
  std::vector<M66> rHtH(n, M66::Zero());
  std::vector<V6>  rHtz(n, V6::Zero());

  #pragma omp parallel for schedule(static) num_threads(nthreads)
  for (int i = 0; i < n; ++i) {
    const auto& res = residuals[i];
    const V3D& hr = res.point_cross_normal;
    const V3D hp = res.normal;
    const double w  = 1.0 / res.sigma_squared;
    const double wr = w * res.r;

    rHtH[i].block<3,3>(0, 0).noalias() += w  * hr * hr.transpose();
    rHtH[i].block<3,3>(0, 3).noalias() += w  * hr * hp.transpose();
    rHtH[i].block<3,3>(3, 0).noalias() += w  * hp * hr.transpose();
    rHtH[i].block<3,3>(3, 3).noalias() += w  * hp * hp.transpose();
    rHtz[i].segment<3>(0).noalias()    += wr * hr;
    rHtz[i].segment<3>(3).noalias()    += wr * hp;
  }

  for (int i = 0; i < n; ++i) {
    out.HtH += rHtH[i];
    out.Htz += rHtz[i];
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
