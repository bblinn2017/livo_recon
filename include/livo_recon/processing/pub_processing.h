#pragma once

#include "livo_recon/node_context.h"
#include "livo_recon/common_ros.h"
#include "livo_recon/utils/algo/hashing.h"
#include "livo_recon/utils/data/measures.h"
#include "livo_recon/utils/log/profiler.h"

#include <jsk_rviz_plugins/OverlayText.h>

#include <condition_variable>
#include <queue>
#include <thread>

namespace livo_recon
{

struct PubProcOptions
{
  // Master visualization switch: when false, publishResults() skips
  // publishGeometry/publishImage/publishTimestamp/publishOdometry's ROS
  // publish calls entirely -- not just the .publish() call but the
  // (often expensive: point-cloud building, image conversion, uv-tracking
  // draw) computation feeding it too, since no one's listening. Explicit
  // file exports (outputs/odom/export, outputs/pcd/export, outputs/
  // image/export) are unaffected -- those are actual test outputs, not
  // visualization. Prints periodic progress instead (see publishResults()).
  bool        viz_enable            = true;
  double      vis_ds_leaf_size      = 0.0;
  bool        viz_grid_map          = false;
  bool        viz_plane_map         = false;
  std::string output_path           = "";
  bool        export_pcd            = false;
  bool        export_images         = false;
  int         export_image_interval = 1;
  bool        export_odom           = false;
};

struct WriteTask
{
  std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB>> pcd_points;
  bool        write_image = false;
  cv::Mat     image;
  std::string img_name;
  Eigen::Quaterniond qcw;
  V3D         tcw;
};

class PubProc
{
public:
  explicit PubProc(NodeContext& ctx);

  std::string loadParameters(ros::NodeHandle& pnh);

  void publishResults(const MeasureGroup& mg);
  void exportColmap();

private:
  StateGroupPtr     state_;
  ProfilerPtr       profiler_;
  PrinterPtr        printer_;
  MapBackendPtr     voxel_map_;
  TrackerPtr        tracker_;
  DataQueuesPtr     data_queues_;
  ros::NodeHandle&  pnh_;

  PubProcOptions opts_;

  ros::Publisher pub_geom_;      // PointCloud2 or MarkerArray depending on viz/plane_map
  ros::Publisher pub_odom_;
  ros::Publisher pub_path_;
  ros::Publisher pub_image_;     // raw RGB or grid map depending on viz/grid_map
  ros::Publisher pub_timestamp_; // TEXT_VIEW_FACING marker with relative image.t
  ros::Publisher pub_evo_stats_; // jsk_rviz_plugins/OverlayText table of mg.evo_stats (n/ATE/RTE/ROE), when non-empty -- screen-space overlay, not a 3D marker, so it stays fixed/readable regardless of camera view

  tf::TransformBroadcaster tf_br_;
  nav_msgs::Path           path_;

  robin_hood::unordered_flat_set<VoxelKey, VoxelKeyHash> vis_ds_seen_;

  std::ofstream odom_file_;
  // P8.  One extra file alongside odom_file_, opened/closed the same way,
  // gated on the same outputs/odom/export flag -- the existing .tum stays
  // byte-identical for every consumer that already parses it.
  std::ofstream pose_pair_file_;
  std::ofstream points3d_file_;
  uint64_t      points3d_count_     = 0;
  std::ofstream images_file_;
  uint64_t      images_count_       = 0;
  int           frame_count_        = 0;
  int           export_image_count_ = 0;

  std::queue<WriteTask>   write_queue_;
  std::mutex              write_mutex_;
  std::condition_variable write_cv_;
  std::thread             write_thread_;
  bool                    write_stop_ = false;

  void writeWorker();
  void publishOdometry (const MeasureGroup& mg);
  void publishGeometry (const MeasureGroup& mg);
  void publishImage    (const MeasureGroup& mg);
  void publishTimestamp(const MeasureGroup& mg);
  void publishEvoStats (const MeasureGroup& mg);
  void enqueueExport   (const MeasureGroup& mg);
};

}  // namespace livo_recon
