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

// Constructs the concrete MapBackend named by backend_name -- "voxel"
// (default, today's VoxelMap) or "akf" (Phase 3's AkfMap). Called once
// from NodeContext's construction, before the rest of that backend's own
// loadParameters() runs (the caller reads voxel_map/backend off pnh
// itself, ahead of this call, to know which concrete class to build).
MapBackendPtr createMapBackend(const std::string& backend_name, StateGroupPtr state,
                               ProfilerPtr profiler, DataQueuesPtr data_queues);

}  // namespace livo_recon
