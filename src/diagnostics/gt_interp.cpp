#include "livo_recon/diagnostics/gt_interp.h"

#include <algorithm>
#include <cmath>

namespace livo_recon
{

GtInterpResult interpolateGtToQueryTime(const std::vector<GtSample3>& samples, double t_query, double max_gap)
{
  GtInterpResult out;
  if (samples.empty()) return out;
  if (t_query < samples.front().t || t_query > samples.back().t) return out;   // never extrapolate

  // Binary search: first index whose t is >= t_query.
  auto it = std::lower_bound(samples.begin(), samples.end(), t_query,
                              [](const GtSample3& s, double t) { return s.t < t; });
  const size_t hi_idx = static_cast<size_t>(it - samples.begin());

  // Exact match (including landing on the very first sample, where there is
  // no preceding bracket at all): report it directly, fraction 0, no
  // division, self-consistent regardless of position in the array. A small
  // relative tolerance absorbs floating-point representation differences
  // between an accumulated timestamp (e.g. i*dt) and a literal query of the
  // "same" value.
  const double eq_tol = 1e-9 * std::max(1.0, std::abs(t_query));
  if (hi_idx < samples.size() && std::abs(samples[hi_idx].t - t_query) <= eq_tol) {
    out.available = true;
    out.pos = samples[hi_idx].pos;
    out.t_lo = out.t_hi = t_query;
    out.fraction = 0.0;
    out.gap = 0.0;
    return out;
  }

  const size_t lo_idx = hi_idx - 1;   // safe here: exact-match case above already handled hi_idx==0
  const GtSample3& lo = samples[lo_idx];
  const GtSample3& hi = samples[hi_idx];
  out.t_lo = lo.t;
  out.t_hi = hi.t;
  out.gap = hi.t - lo.t;
  if (out.gap > max_gap) { out.available = false; return out; }

  out.fraction = (t_query - lo.t) / out.gap;
  out.pos = lo.pos + out.fraction * (hi.pos - lo.pos);
  out.available = true;
  return out;
}

}  // namespace livo_recon
