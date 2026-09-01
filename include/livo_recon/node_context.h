#pragma once

#include <memory>

#include <ros/ros.h>
#include <image_transport/image_transport.h>

// History (8-20): see docs/livo_recon_changelog.md#include-livo_recon-node_context.h-8
namespace livo_recon
{

struct DataQueues;
struct Measures;
class  StateGroup;
class  FrameProfiler;
class  Printer;
class  MapBackend;
class  Tracker;

// Re-declaring these aliases here (identical redeclaration of the same
// type as their canonical definitions in data_queues.h/measures.h/state.h/
// profiler.h/printer.h/map_backend.h/tracker.h -- legal in C++, and
// keeps `DataQueuesPtr` etc. usable by any file that only includes this
// header) rather than requiring each of those headers just for their one
// alias.
using DataQueuesPtr  = std::shared_ptr<DataQueues>;
using MeasuresPtr    = std::shared_ptr<Measures>;
using StateGroupPtr  = std::shared_ptr<StateGroup>;
using ProfilerPtr    = std::shared_ptr<FrameProfiler>;
using PrinterPtr     = std::shared_ptr<Printer>;
using MapBackendPtr  = std::shared_ptr<MapBackend>;
using TrackerPtr     = std::shared_ptr<Tracker>;

struct NodeContext
{
  ros::NodeHandle&                 nh;
  ros::NodeHandle&                 pnh;
  image_transport::ImageTransport& it;

  DataQueuesPtr  data_queues;
  MeasuresPtr    measures;
  StateGroupPtr  state;
  ProfilerPtr    profiler;
  PrinterPtr     printer;
  MapBackendPtr  voxel_map;  // "voxel" (VoxelMap, default) or "akf" (AkfMap) -- see voxel_map/backend
  TrackerPtr     tracker;

  NodeContext(ros::NodeHandle& nh_, ros::NodeHandle& pnh_,
              image_transport::ImageTransport& it_);
};

}  // namespace livo_recon
