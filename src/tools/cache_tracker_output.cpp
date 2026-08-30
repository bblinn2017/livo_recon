// Standalone cache-generation tool for livo_recon's tracker (CoTracker/
// TAPNext/Track-On, see vio/tracker.h).
//
// Motivation: Tracker::stepAndReseedOnce() (src/vio/tracker.cpp) never
// reads pose/EKF state anywhere -- confirmed in tracker.cpp, state_ is
// only read for FIXED camera intrinsics/distortion (undistortPixel()) and
// in recordFramePose() (called externally by LivoReconNode::estimateState(),
// outside step()/reseed()'s own call path). Its output is a pure function of (bag,
// grid_rows, grid_cols, backend, tracker_format). Every ablation sweep
// that only varies downstream (state-dependent) LIO/VIO params (residual
// weighting/gating, acc_cov/gyr_cov, ...) re-runs the identical, GPU-
// expensive tracker pass over the identical bag every time. This tool
// runs the tracker ONCE per (bag, grid_rows, grid_cols, backend,
// tracker_format) combination and serializes its raw per-frame output
// (TrackedFrame, see include/livo_recon/vio/tracked_frame.h) to a cache
// file that Tracker's own replay mode (TrackerOptions::track_cache_path)
// reads back, skipping the GPU pass entirely.
//
// Directly ported from FAST-LIVO2's test/cache_tracker_output.cpp -- see
// that file's header comment for the design this mirrors. Two differences
// from FAST-LIVO2's version: (1) livo_recon's TrackedFrame carries no
// bearing_z/rad_per_px per-point fields (it DOES carry a per-point 2x2
// uv_jacobian now, same as FAST-LIVO2 -- see AnchorPoint::uv_jacobian_seed/
// TrackedFrame::uv_curr_uv_jacobian's doc comments, feeding
// VioProcOptions::distortion_weight_on's rigorous weight), so the format
// is close but not identical; (2) this tool drives the REAL Tracker class
// directly via its public async API (feedImage/waitReadyFor/consumeResult)
// rather than reimplementing step()/reseed() bookkeeping standalone, since
// livo_recon's Tracker (unlike FAST-LIVO2's) already fully owns that logic
// behind a clean interface with no other node-level dependencies.
//
// Reads bag/topic/tracker parameters from the ROS parameter server under
// this node's own private namespace, using the SAME param names/paths
// StateGroup::loadParameters() and Tracker::loadParameters() already read
// (topics/image, tracker/backend, tracker/tracker_format, tracker/
// grid_selection/*, camera/*) -- so this tool can be launched with the
// exact same per-sequence launch/config a live run uses, just pointed at
// a different node name, plus two additional params:
//   common/offline_bag_path      -- REQUIRED, the bag to read images from
//   tracker/cache_out_path       -- REQUIRED, where to write the .rlivtrackcache
//
// ─────────────────────────────────────────────────────────────────────────
//  .rlivtrackcache binary format (little-endian, native double/float/int
//  widths -- this is an internal cache written and read on the same
//  architecture, no cross-platform portability attempted)
// ─────────────────────────────────────────────────────────────────────────
//  Header:
//    char[4]   magic            = "RLTC"
//    uint32_t  version           = 3
//    int32_t   grid_rows
//    int32_t   grid_cols
//    uint32_t  tracker_format_len, then that many bytes (backend + "/" +
//                                    tracker_format, no NUL terminator)
//    uint32_t  bag_basename_len,   then that many bytes
//    uint32_t  n_frames           -- number of TrackedFrame records below
//
//  Then n_frames records, each:
//    double    timestamp
//    double    prev_timestamp     -- (v3+) timestamp of the frame whose
//                                     reseed produced this frame's anchors/
//                                     indices -- see TrackedFrame's doc
//                                     comment. NOT rot_anchor/pos_anchor --
//                                     those are resolved fresh at replay
//                                     time from the REPLAYING run's own
//                                     live trajectory (Tracker::
//                                     resolveAnchorPose()), not cached.
//    uint8_t   ok
//    uint8_t   reseeded
//    uint32_t  n_uv               -- length of uv_curr/uv_curr_uv_jacobian/indices (parallel, same length)
//    n_uv *    { float x, float y }                       -- uv_curr
//    n_uv *    { double m00,m01,m10,m11 }  (row-major)     -- uv_curr_uv_jacobian
//    n_uv *    int32_t                                     -- indices
//    uint32_t  n_anchors
//    n_anchors * {
//      float x, float y,                                   -- anchors[i].uv_seed
//      double m00,m01,m10,m11  (row-major)                 -- anchors[i].uv_jacobian_seed
//    }
//
// Usage (roslaunch, same param namespace as the real node):
//   rosrun livo_recon cache_tracker_output_livo_recon
//       _common/offline_bag_path:=/path/to/bag.bag
//       _tracker/cache_out_path:=/path/to/out.rlivtrackcache
//       [any other tracker/*, topics/image, camera/* overrides the sequence needs]

#include <cstring>
#include "livo_recon/utils/log/param_warn.h"
#include <fstream>
#include <sstream>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <cv_bridge/cv_bridge.h>

#include "livo_recon/common_lib.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/profiler.h"
#include "livo_recon/utils/algo/omp_utils.h"
#include "livo_recon/vio/tracker.h"

using namespace livo_recon;

namespace
{

template <typename T>
void writePod(std::ofstream& f, const T& v)
{
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

void writeStr(std::ofstream& f, const std::string& s)
{
  const uint32_t len = static_cast<uint32_t>(s.size());
  writePod(f, len);
  f.write(s.data(), len);
}

void writeUvJacobian(std::ofstream& f, const Eigen::Matrix2d& J)
{
  writePod(f, J(0, 0));
  writePod(f, J(0, 1));
  writePod(f, J(1, 0));
  writePod(f, J(1, 1));
}

// Mirrors TrackerCacheReader::load()'s readTrackedFrame() field-for-field
// (see tracker_cache.cpp).
void writeTrackedFrame(std::ofstream& f, const TrackedFrame& tf)
{
  writePod(f, tf.timestamp);
  writePod(f, tf.prev_timestamp);
  writePod(f, static_cast<uint8_t>(tf.ok ? 1 : 0));
  writePod(f, static_cast<uint8_t>(tf.reseeded ? 1 : 0));

  const uint32_t n_uv = static_cast<uint32_t>(tf.uv_curr.size());
  writePod(f, n_uv);
  for (const auto& p : tf.uv_curr)
  {
    writePod(f, static_cast<float>(p.x));
    writePod(f, static_cast<float>(p.y));
  }
  for (const auto& J : tf.uv_curr_uv_jacobian) writeUvJacobian(f, J);
  for (int idx : tf.indices) writePod(f, static_cast<int32_t>(idx));

  const uint32_t n_anchors = static_cast<uint32_t>(tf.anchors.size());
  writePod(f, n_anchors);
  for (const auto& a : tf.anchors)
  {
    writePod(f, static_cast<float>(a.uv_seed.x));
    writePod(f, static_cast<float>(a.uv_seed.y));
    writeUvJacobian(f, a.uv_jacobian_seed);
  }
}

std::string basenameOf(const std::string& path)
{
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace

int main(int argc, char** argv)
{
  // Must be the very first statement -- see pinOmpThreadsForDeterminism()'s
  // doc comment (fixes a confirmed run-to-run OpenMP non-determinism, see
  // main.cpp). This tool has no EKF accumulation of its own, but links the
  // same LibTorch/AOTInductor backend, so pin here too for consistency and
  // to keep generated caches (and any other parallel region this tool
  // touches) reproducible.
  livo_recon::pinOmpThreadsForDeterminism();

  ros::init(argc, argv, "cache_tracker_output");
  ros::NodeHandle pnh("~");

  std::string bag_path, out_path, image_topic;
  // Private namespace ("~", i.e. pnh) throughout, matching every launch
  // file's convention of loading <rosparam> INSIDE the <node> tag (see
  // livo_recon_slam_2022.launch) -- a plain nh.param() here would never
  // see a value set via a launch file's config/defaults/common.yaml load
  // or CLI-passed _common/offline_bag_path:=... remap.
  paramWarn<std::string>(pnh, "common/offline_bag_path", bag_path, "");
  paramWarn<std::string>(pnh, "tracker/cache_out_path", out_path, "");
  paramWarn<std::string>(pnh, "topics/image", image_topic, "/camera/image_raw");

  // MUST mirror CbkProc::imageCallback()'s own decimation exactly (see
  // that function's doc comment) -- a real run only ever calls
  // Tracker::feedImage() for every image_subsample_n-th image (HILTI:
  // 4, to match its rotating-lidar full-scan cadence against a faster
  // camera; NTU_VIRAL: 1, no decimation). Feeding EVERY image here
  // regardless (as this tool originally did) produces a cache whose
  // per-frame tracking reflects 1-image-apart displacement/reseed
  // dynamics instead of the actual image_subsample_n-image-apart dynamics
  // a real run needs -- silently wrong data, not a lookup failure, since
  // every real-run timestamp WOULD still be found in the (superset) cache,
  // just pointing at the wrong tracking result for it.
  int image_subsample_n = 1;
  paramWarn<int>(pnh, "preprocess/image_subsample_n", image_subsample_n, 1);
  ROS_INFO_STREAM("[cache_tracker_output] preprocess/image_subsample_n=" << image_subsample_n);

  if (bag_path.empty() || out_path.empty())
  {
    ROS_ERROR("cache_tracker_output: both common/offline_bag_path and "
               "tracker/cache_out_path are required");
    return 1;
  }

  auto state = std::make_shared<StateGroup>();
  ROS_INFO_STREAM(state->loadParameters(pnh));
  auto profiler = std::make_shared<FrameProfiler>();

  Tracker tracker(state, profiler);
  ROS_INFO_STREAM(tracker.loadParameters(pnh));
  if (!tracker.loaded() || tracker.replayMode())
  {
    ROS_ERROR("cache_tracker_output: tracker backend failed to load (or "
               "tracker/track_cache_path was set -- this tool must run "
               "LIVE, not in replay mode)");
    return 1;
  }

  // Same param names Tracker::loadParameters() itself just read, used here
  // only to stamp the cache header for TrackerCacheReader's mismatch check
  // (Tracker has no public opts() getter -- deliberately not adding one
  // just for this, duplicating two already-read param reads is cheaper
  // than growing Tracker's public surface for a single tool).
  std::string backend, tracker_format;
  int grid_rows = 0, grid_cols = 0;
  paramWarn<std::string>(pnh, "tracker/backend", backend, "cotracker");
  paramWarn<std::string>(pnh, "tracker/tracker_format", tracker_format, "jit");
  paramWarn<int>(pnh, "tracker/grid_selection/grid_rows", grid_rows, 10);
  paramWarn<int>(pnh, "tracker/grid_selection/grid_cols", grid_cols, 10);

  ROS_INFO_STREAM("[cache_tracker_output] reading " << bag_path << " (topic " << image_topic << ")");

  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);
  rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>{image_topic}));

  tracker.startAsyncTracking();

  uint32_t n_fed = 0;
  int frame_counter = 0;  // mirrors CbkProc::imageCallback()'s own static counter exactly
  for (const rosbag::MessageInstance& m : view)
  {
    sensor_msgs::Image::ConstPtr msg = m.instantiate<sensor_msgs::Image>();
    if (!msg) continue;

    if (image_subsample_n > 1 && (++frame_counter % image_subsample_n != 0)) continue;

    const cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    if (cv_ptr->image.empty()) continue;
    tracker.feedImage(cv_ptr->image.clone(), msg->header.stamp.toSec());
    ++n_fed;
  }
  bag.close();
  ROS_INFO_STREAM("[cache_tracker_output] fed " << n_fed << " images, draining results...");

  std::vector<TrackedFrame> frames;
  frames.reserve(n_fed);
  // See Tracker's class doc comment -- feed and consume are always 1:1, in
  // strict arrival order, so waiting for exactly n_fed results (with a
  // generous per-frame timeout treated as "genuinely stuck, stop waiting")
  // is sufficient; no separate "async queue drained" signal exists (or is
  // needed) beyond that count.
  while (frames.size() < n_fed)
  {
    if (!tracker.waitReadyFor(std::chrono::seconds(120)))
    {
      ROS_ERROR_STREAM("[cache_tracker_output] FATAL: stalled waiting for tracker result "
          << frames.size() << "/" << n_fed << " after 120s -- aborting");
      return 1;
    }
    TrackedFrame tf;
    while (tracker.consumeResult(tf)) frames.push_back(std::move(tf));
  }
  tracker.stopAsyncTracking();

  std::ofstream f(out_path, std::ios::binary);
  if (!f.is_open())
  {
    ROS_ERROR_STREAM("[cache_tracker_output] FATAL: failed to open output path " << out_path);
    return 1;
  }
  f.write("RLTC", 4);
  writePod(f, static_cast<uint32_t>(3));
  writePod(f, static_cast<int32_t>(grid_rows));
  writePod(f, static_cast<int32_t>(grid_cols));
  writeStr(f, backend + "/" + tracker_format);
  writeStr(f, basenameOf(bag_path));
  writePod(f, static_cast<uint32_t>(frames.size()));
  for (const auto& tf : frames) writeTrackedFrame(f, tf);
  f.close();

  ROS_INFO_STREAM("[cache_tracker_output] wrote " << frames.size() << " frames to " << out_path);
  return 0;
}
