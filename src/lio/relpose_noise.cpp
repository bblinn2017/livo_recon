#include "livo_recon/lio/relpose_noise.h"

#include <algorithm>

namespace livo_recon
{

namespace
{
// Per R2: the per-step contribution to Sigma_p, (t_end-t_k)^2*Q_v_k + Q_p_k + (t_end-t_k)*(Q_pv_k+Q_pv_k^T).
M3D sigmaPTerm(const RelposeStepNoise& s, double t_end)
{
  const double dte = t_end - s.t;
  return dte * dte * s.Q_v + s.Q_p + dte * (s.Q_pv + s.Q_pv.transpose());
}
}  // namespace

void relposeProcessNoiseBruteForce(
    const std::vector<RelposeStepNoise>& steps, double t_point, double t_end,
    M3D& sigma_theta, M3D& sigma_p)
{
  sigma_theta.setZero();
  sigma_p.setZero();
  for (const RelposeStepNoise& s : steps)
  {
    const double s_end = s.t + s.dt;
    if (s_end <= t_point)
      continue;   // entirely before the point: no contribution
    double f = 1.0;
    if (s.t < t_point)   // straddles t_point
      f = (s.dt > 1e-12) ? (s_end - t_point) / s.dt : 0.0;
    sigma_theta.noalias() += f * s.Q_theta;
    sigma_p.noalias() += f * sigmaPTerm(s, t_end);
  }
}

RelposeSuffixTables buildRelposeSuffixTables(
    const std::vector<RelposeStepNoise>& steps, double t_end)
{
  RelposeSuffixTables tabs;
  const size_t K = steps.size();
  tabs.step_start.resize(K);
  tabs.step_end.resize(K);
  tabs.term_theta.resize(K);
  tabs.term_p.resize(K);
  tabs.suffix_theta.assign(K + 1, M3D::Zero());
  tabs.suffix_p.assign(K + 1, M3D::Zero());

  for (size_t k = 0; k < K; ++k)
  {
    tabs.step_start[k] = steps[k].t;
    tabs.step_end[k] = steps[k].t + steps[k].dt;
    tabs.term_theta[k] = steps[k].Q_theta;
    tabs.term_p[k] = sigmaPTerm(steps[k], t_end);
  }
  for (size_t k = K; k-- > 0;)
  {
    tabs.suffix_theta[k] = tabs.suffix_theta[k + 1] + tabs.term_theta[k];
    tabs.suffix_p[k] = tabs.suffix_p[k + 1] + tabs.term_p[k];
  }
  return tabs;
}

void relposeProcessNoiseFromSuffix(
    const RelposeSuffixTables& tabs, double t_point, M3D& sigma_theta, M3D& sigma_p)
{
  const size_t K = tabs.step_start.size();
  sigma_theta.setZero();
  sigma_p.setZero();
  if (K == 0)
    return;

  // Find the straddling step: the last step whose start time is <= t_point. Steps are ascending
  // in t (poses are built in forward time order), so an upper_bound on step_start gives the first
  // step STARTING after t_point; the straddling step is the one just before it.
  const auto it = std::upper_bound(tabs.step_start.begin(), tabs.step_start.end(), t_point);
  size_t ks = (it == tabs.step_start.begin()) ? 0 : static_cast<size_t>(it - tabs.step_start.begin()) - 1;

  if (tabs.step_end[ks] <= t_point)
  {
    // t_point is at or after the end of every step covering it (e.g. exactly at scan end, or
    // past the last step's interval) -- no steps remain with t_k >= t_point.
    if (ks + 1 >= K) return;
    ++ks;
  }

  double f = 1.0;
  const double dt = tabs.step_end[ks] - tabs.step_start[ks];
  if (tabs.step_start[ks] < t_point)
    f = (dt > 1e-12) ? (tabs.step_end[ks] - t_point) / dt : 0.0;

  sigma_theta = f * tabs.term_theta[ks] + tabs.suffix_theta[ks + 1];
  sigma_p = f * tabs.term_p[ks] + tabs.suffix_p[ks + 1];
}

}  // namespace livo_recon
