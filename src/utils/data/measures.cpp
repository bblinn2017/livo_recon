#include "livo_recon/utils/data/measures.h"

namespace livo_recon
{

bool Measures::isValid(double t) const
{
  return !calib_done.get() || t >= curr_time.get();
}

void Measures::pushMeasureGroup(MeasureGroup&& mg)
{
  std::lock_guard<std::mutex> lock(measure_mutex);
  measure_groups.emplace_back(std::move(mg));
}

bool Measures::popMeasureGroup(MeasureGroup& mg)
{
  std::lock_guard<std::mutex> lock(measure_mutex);
  if (measure_groups.empty()) return false;
  mg = std::move(measure_groups.front());
  measure_groups.pop_front();
  return true;
}

}  // namespace livo_recon
