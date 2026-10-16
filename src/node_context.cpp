#include "livo_recon/node_context.h"

#include "livo_recon/utils/data/data_queues.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/utils/log/printer.h"
#include "livo_recon/map/map_backend_factory.h"
#include "livo_recon/vio/tracker.h"

namespace livo_recon
{

NodeContext::NodeContext(ros::NodeHandle& nh_, ros::NodeHandle& pnh_,
                          image_transport::ImageTransport& it_)
  : nh(nh_), pnh(pnh_), it(it_),
    data_queues(std::make_shared<DataQueues>()),
    measures(std::make_shared<Measures>()),
    state(std::make_shared<StateGroup>()),
    profiler(std::make_shared<FrameProfiler>()),
    printer(std::make_shared<Printer>()),
    // R97A: voxel_map/backend is removed; createMapBackend() always builds the one map (VoxelMap, the binned map).
    voxel_map(createMapBackend(state, profiler, data_queues)),
    tracker(std::make_shared<Tracker>(state, profiler))
{}

}  // namespace livo_recon
