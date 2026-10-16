#include "livo_recon/map/map_backend_factory.h"

#include "livo_recon/map/voxelmap.h"

namespace livo_recon
{

// R97A: voxel_map/backend is removed; VoxelMap (the binned map, R88) is the only MapBackend.
MapBackendPtr createMapBackend(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues)
{
  return std::make_shared<VoxelMap>(state, profiler, data_queues);
}

}  // namespace livo_recon
