#include "livo_recon/node_context.h"

#include "livo_recon/utils/data/data_queues.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/utils/log/printer.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/map/map_backend_factory.h"
#include "livo_recon/vio/tracker.h"

namespace livo_recon
{

namespace
{
// Read once, ahead of the rest of param loading, since the factory needs
// to know which concrete MapBackend to construct before that backend's
// own loadParameters() can run. Defaults to "voxel" (today's VoxelMap) so
// every existing config's behavior is unchanged unless a job explicitly
// overrides voxel_map/backend to "akf".
std::string mapBackendName(ros::NodeHandle& pnh)
{
  std::string backend_name = "voxel";
  paramWarn<std::string>(pnh, "voxel_map/backend", backend_name, "voxel");
  return backend_name;
}
}  // namespace

NodeContext::NodeContext(ros::NodeHandle& nh_, ros::NodeHandle& pnh_,
                          image_transport::ImageTransport& it_)
  : nh(nh_), pnh(pnh_), it(it_),
    data_queues(std::make_shared<DataQueues>()),
    measures(std::make_shared<Measures>()),
    state(std::make_shared<StateGroup>()),
    profiler(std::make_shared<FrameProfiler>()),
    printer(std::make_shared<Printer>()),
    voxel_map(createMapBackend(mapBackendName(pnh_), state, profiler, data_queues)),
    tracker(std::make_shared<Tracker>(state, profiler))
{}

}  // namespace livo_recon
