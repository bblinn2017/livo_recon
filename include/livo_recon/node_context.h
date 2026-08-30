#pragma once

#include <memory>

#include <ros/ros.h>
#include <image_transport/image_transport.h>

// Forward-declares only -- NO full type definitions here. This is the
// single highest-leverage fix for livo_recon's rebuild-cascade problem
// (2026-08-14): every processing/*.h includes this file, so it used to
// transitively pull in common_lib.h (ROS/OpenCV/PCL/Eigen plus ALL of
// livo_recon's own utility headers: ekf.h, state.h, measures.h, etc.) plus
// the full voxelmap.h/tracker.h -- meaning a change to e.g. ekf.h alone
// forced a rebuild of the whole package. Since NodeContext only ever
// stores these as shared_ptr (never by value), the pointee types don't
// need to be complete here -- shared_ptr<Incomplete> is fine as a class
// member (its destructor doesn't need T complete, since make_shared bakes
// a type-erased deleter into the control block at construction time).
// The constructor body (the only place that actually needs the complete
// types, for the make_shared<T>() calls) moved to node_context.cpp.
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
