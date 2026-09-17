#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "livo_recon/utils/data/data_wrappers.h"

namespace livo_recon
{

template <typename T>
class LockedValue
{
  struct State
  {
    std::mutex mutex;
    T value{};
  };

public:
  LockedValue()
    : state_(std::make_shared<State>())
  {}

  explicit LockedValue(const T& v)
    : state_(std::make_shared<State>())
  {
    state_->value = v;
  }

  T get() const
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->value;
  }

  void set(const T& v)
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->value = v;
  }

  LockedValue(const LockedValue&) = default;
  LockedValue& operator=(const LockedValue&) = default;
  LockedValue(LockedValue&&) = default;
  LockedValue& operator=(LockedValue&&) = default;

private:
  std::shared_ptr<State> state_;
};

struct DataQueues
{
  std::deque<ImageData> image_queue;
  std::deque<std::vector<PointXYZT>> lidar_queue;
  // History (60-69): see docs/livo_recon_changelog.md#include-livo_recon-utils-data-data_queues.h-60
  std::deque<std::vector<PointXYZT>> dry_run_lidar_queue;
  std::deque<ImuSample> imu_queue;
  // Raw ground-truth pose samples (evo/gt_source=="topic") -- ingestion
  // lives in CbkProc::gtCallback() alongside image/lidar/imu, same as
  // every other raw-message queue here; EvoProc drains this each
  // processEvo() tick via popGt() rather than owning its own ROS
  // subscriber. Unbounded (unlike the others, no natural "max_time" pop
  // cursor exists for it) -- EvoProc enforces its own bounded buffer size
  // after draining, mirroring gtCallback()'s old queue_size eviction.
  std::deque<GtPoseSample> gt_queue;

  std::mutex image_mutex;
  std::mutex lidar_mutex;
  std::mutex dry_run_lidar_mutex;
  std::mutex imu_mutex;
  std::mutex gt_mutex;

  double latest_image_time = 0.0;
  double latest_lidar_time = 0.0;
  double latest_dry_run_lidar_time = 0.0;
  double latest_imu_time   = 0.0;

  double start_time = 0.;

  // Wall-clock (steady_clock) nanosecond timestamp of the most recent
  // arrival on the lidar/imu topics specifically (not image -- ready()
  // only ever needs to know how settled *those* two streams are relative
  // to the front image). Updated by pushLidar/pushImu regardless of
  // whether the message is actually kept.
  std::atomic<int64_t> last_lidar_arrival_ns{0};
  std::atomic<int64_t> last_imu_arrival_ns{0};

  // CQ-37: bag-time analog of last_{lidar,imu}_arrival_ns, for offline
  // mode's quiet check -- see ready()'s own doc comment and streamSettled()
  // (data_queues.cpp) for why offline mode needs a DIFFERENT clock here
  // than live mode's wall-clock one. Updated by push{Imu,Lidar,Image} to
  // the max message timestamp seen so far across ALL THREE streams (a CAS
  // loop, same pattern as voxelplane.cpp's g_max_plane_var_trace, since
  // live mode's callbacks can race here just as they already do on
  // last_{lidar,imu}_arrival_ns).
  std::atomic<double> latest_fed_time{-1e18};

  // Set once, at startup, by runOffline() -- see ready()'s own doc comment.
  // false (default) preserves live mode's existing wall-clock quiet check
  // exactly; live playback still has the real cross-topic ROS-transport-
  // delivery race streamSettled()'s wall-clock branch exists for, which a
  // bag-time check cannot substitute for (message timestamp order is not
  // the same thing as ROS transport delivery order under live playback).
  bool offline_mode = false;
  void setOfflineMode(bool v) { offline_mode = v; }
  // Internal helper for push{Imu,Lidar,Image} -- see latest_fed_time's doc
  // comment above.
  void updateLatestFedTime(double t);

  // ready()'s per-stream (lidar, imu) acceptance test, checked against the
  // front image's timestamp `t`, is:
  //   (latest_X_time >= t + lookahead_margin_s)                     -- (A)
  //   || (latest_X_time >= t && quiet for >= quiet_margin_s)         -- (B)
  // (A): that stream already has a full comfortable margin of data past
  // this image -- trust it outright, no need to wait further.
  // (B): the stream has at least reached this image's own timestamp, but
  // not the full margin -- only trust it once that stream itself has gone
  // quiet (no new arrivals) for quiet_margin_s, i.e. we've likely caught
  // up to whatever's actually available (e.g. near the tail end of a bag)
  // rather than being in the middle of a live cross-topic delivery race.
  //
  // Needed because image/lidar/imu are three independent ROS subscriptions;
  // their relative delivery order under live playback isn't strictly
  // guaranteed to match publish-time order. Confirmed necessary: two
  // identical calibration runs accumulated 1027 vs. 1003 IMU samples before
  // this margin was added -- a ~0.06s discrepancy in which image got chosen
  // as the start-time reference, which cascaded into large downstream
  // ATE-comparison instability (every measure-group timestamp shifts by
  // that same amount).
  //
  // CQ-37: (B)'s "quiet for quiet_margin_s" was measured in REAL wall-clock
  // time (std::chrono::steady_clock) unconditionally -- correct for live
  // playback, where wall-clock quiet genuinely means "the ROS transport
  // layer has settled" (the cross-topic delivery race this whole mechanism
  // exists for is itself a real-time phenomenon: message A published
  // before B can still be RECEIVED after B due to transport/subscriber
  // jitter, independent of either message's own timestamp). But
  // runOffline() reads directly from rosbag::View, which delivers messages
  // in STRICT TIMESTAMP ORDER by construction -- there IS no cross-topic
  // delivery race to wait out in offline mode, so wall-clock quiet is the
  // wrong clock entirely there. Confirmed as the root cause of a
  // reproducible, nondeterministic "IMU calibration stalled" hang this
  // session: runOffline()'s CALIB loop feeds bag messages in an untimed
  // busy loop (no rate limit), so its 300-iteration stall budget
  // (CalibProcOptions::stall_calls_max) can exhaust in far less than
  // quiet_margin_s's 50ms of real time whenever the CPU is fast/idle --
  // condition (B) could then never fire before the call budget ran out,
  // and whether it happened to fire first instead depended on incidental
  // scheduling jitter, not on the bag or the config. When offline_mode is
  // set, streamSettled() (data_queues.cpp) uses BAG-TIME quiet instead:
  // latest_fed_time (the furthest-advanced message timestamp seen across
  // ALL THREE streams, i.e. "how far into the bag has been read") minus
  // this stream's own latest_stream_time, compared against quiet_margin_s
  // -- same semantic ("has the feed moved past this stream without a new
  // arrival"), measured on the clock offline reading actually advances on,
  // so it is exact and CPU-speed-independent instead of racing an
  // unrelated iteration budget.
  double lookahead_margin_s = 0.1;
  double quiet_margin_s     = 0.05;

  void setStartTime(double timestamp);

  void pushImu(const ImuSample& msg);
  void pushLidar(std::vector<PointXYZT>&& msg);
  // See dry_run_lidar_queue's doc comment above.
  void pushDryRunLidar(std::vector<PointXYZT>&& msg);
  void pushImage(const ImageData& msg);
  void pushGt(const GtPoseSample& msg);

  bool ready();
  ImageData popImage();
  bool popLidar(std::vector<PointXYZT>& out, double max_time);
  bool popDryRunLidar(std::vector<PointXYZT>& out, double max_time);
  bool popImu(std::deque<ImuSample>& out, double max_time);
  // Drains gt_queue entirely (not time-bounded like the pop* above -- GT
  // samples arrive independently of any measure-group cadence, so there's
  // no natural max_time to pop up to). Empty vector if nothing's arrived.
  std::vector<GtPoseSample> popGt();
};
using DataQueuesPtr = std::shared_ptr<DataQueues>;

}  // namespace livo_recon
