#pragma once

#include <memory>
#include <string>

#include "livo_recon/map/map_backend.h"

namespace livo_recon
{

// Forward-declares only, mirroring node_context.h's own rationale (avoid
// pulling in state.h/profiler.h/data_queues.h just for this factory's
// signature).
struct DataQueues;
class  StateGroup;
class  FrameProfiler;
using DataQueuesPtr = std::shared_ptr<DataQueues>;
using StateGroupPtr = std::shared_ptr<StateGroup>;
using ProfilerPtr   = std::shared_ptr<FrameProfiler>;

// Constructs the one concrete MapBackend (VoxelMap, the binned map, R88).
// R97A: voxel_map/backend is removed; there is no longer a choice to make.
MapBackendPtr createMapBackend(StateGroupPtr state, ProfilerPtr profiler, DataQueuesPtr data_queues);

}  // namespace livo_recon
