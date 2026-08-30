#include "livo_recon/map/map_backend_factory.h"

#include <stdexcept>

#include "livo_recon/map/voxelmap.h"
#include "livo_recon/map/akf_map.h"

namespace livo_recon
{

MapBackendPtr createMapBackend(const std::string& backend_name, StateGroupPtr state,
                               ProfilerPtr profiler, DataQueuesPtr data_queues)
{
  if (backend_name == "voxel") {
    return std::make_shared<VoxelMap>(state, profiler, data_queues);
  }
  if (backend_name == "akf") {
    return std::make_shared<AkfMap>(state, profiler, data_queues);
  }
  // Fail loudly rather than silently falling back to "voxel", matching
  // this codebase's "fail loud on unexpected config" convention (see
  // run_job.sh's CACHE_FILE check for the same philosophy).
  throw std::runtime_error("createMapBackend: unknown voxel_map/backend '" + backend_name +
                            "' (expected 'voxel' or 'akf')");
}

}  // namespace livo_recon
