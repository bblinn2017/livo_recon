#pragma once

// Legacy "everything" aggregator. As of 2026-08-14, NodeContext (and hence
// every processing/*.h) no longer includes this transitively -- see
// node_context.h's doc comment. This header still exists for the handful
// of files that genuinely want the full breadth (ROS/CV/PCL message types
// AND all of livo_recon's own internal utility types together): map/
// voxelmap.h, vio/tracker.h, tools/cache_tracker_output.cpp. New code
// should prefer including exactly the specific internal headers it needs
// (state.h, ekf.h, measures.h, etc.) plus common_ros.h only if it also
// needs ROS/CV/PCL message types -- not this file -- so an unrelated
// header change doesn't force it to rebuild.

#include "livo_recon/common_ros.h"

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/utils/log/printer.h"
#include "livo_recon/utils/data/data_queues.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/map/voxelmap_utils.h"

namespace livo_recon
{
}  // namespace livo_recon
