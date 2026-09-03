#include "livo_recon/processing/pub_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/state/state.h"
#include "livo_recon/utils/log/printer.h"
#include "livo_recon/map/voxelmap.h"
#include "livo_recon/vio/tracker.h"

namespace livo_recon
{

PubProc::PubProc(NodeContext& ctx)
  : state_(ctx.state), profiler_(ctx.profiler),
    printer_(ctx.printer), voxel_map_(ctx.voxel_map),
    tracker_(ctx.tracker),
    data_queues_(ctx.data_queues), pnh_(ctx.pnh)
{}

std::string PubProc::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<bool>(pnh, "viz/enable",          opts_.viz_enable,      true);
  paramWarn<double>(pnh, "viz/ds/ds_leaf_size", opts_.vis_ds_leaf_size, 0.0);
  paramWarn<bool>(pnh, "viz/grid_map",        opts_.viz_grid_map,    false);
  paramWarn<bool>(pnh, "viz/plane_map",       opts_.viz_plane_map,   false);

  paramWarn<std::string>(pnh, "outputs/path",               opts_.output_path,           "");
  paramWarn<bool>(pnh, "outputs/pcd/export",         opts_.export_pcd,            false);
  paramWarn<bool>(pnh, "outputs/image/export",       opts_.export_images,         false);
  paramWarn<int>(pnh, "outputs/image/interval",     opts_.export_image_interval, 1);
  opts_.export_image_interval = std::max(1, opts_.export_image_interval);

  paramWarn<bool>(pnh, "outputs/odom/export", opts_.export_odom, false);
  if (opts_.export_odom && !opts_.output_path.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(opts_.output_path, ec);
    const std::string odom_path = opts_.output_path + "/odometry.txt";
    odom_file_.open(odom_path, std::ios::out | std::ios::trunc);
    if (!odom_file_.is_open())
      ROS_ERROR_STREAM("[output] failed to open odom export path '"
        << odom_path << "' — disabling odom export");
    // P8.  Same gate, same directory, a NEW file -- odometry.txt itself is
    // untouched.
    const std::string pose_pair_path = opts_.output_path + "/pose_pair.csv";
    pose_pair_file_.open(pose_pair_path, std::ios::out | std::ios::trunc);
    if (pose_pair_file_.is_open())
      pose_pair_file_ << "t,px,py,pz,qx,qy,qz,qw,"
                         "ppx,ppy,ppz,pqx,pqy,pqz,pqw,"
                         "vx,vy,vz,pvx,pvy,pvz\n";
    else
      ROS_ERROR_STREAM("[output] failed to open pose_pair export path '"
        << pose_pair_path << "' — disabling pose_pair export");
  }

  if (!opts_.output_path.empty() && (opts_.export_pcd || opts_.export_images)) {
    std::error_code ec;
    const auto try_mkdir = [&](const std::string& p) {
      std::filesystem::create_directories(p, ec);
      if (ec) return false;
      const std::string probe = p + "/.write_probe";
      std::ofstream f(probe);
      if (!f.is_open()) { ec = std::make_error_code(std::errc::permission_denied); return false; }
      f.close();
      std::filesystem::remove(probe, ec);
      return true;
    };

    bool ok = true;
    if (opts_.export_images) {
      const std::string img_dir = opts_.output_path + "/images";
      std::filesystem::remove_all(img_dir, ec);
      ok &= try_mkdir(img_dir);
    }
    if (ok) ok &= try_mkdir(opts_.output_path + "/sparse/0");

    if (!ok) {
      ROS_ERROR_STREAM("[output] failed to create or write to output path '"
        << opts_.output_path << "': " << ec.message() << " — disabling all export");
      opts_.output_path.clear();
      opts_.export_pcd    = false;
      opts_.export_images = false;
    }
  }

  if (opts_.export_pcd && !opts_.output_path.empty()) {
    const std::string pts_path = opts_.output_path + "/sparse/0/points3D.bin";
    points3d_file_.open(pts_path, std::ios::binary);
    if (!points3d_file_.is_open()) {
      ROS_ERROR_STREAM("[output] failed to open " << pts_path << " — disabling pcd export");
      opts_.export_pcd = false;
    } else {
      const uint64_t placeholder = 0;
      points3d_file_.write(reinterpret_cast<const char*>(&placeholder), 8);
    }
  }

  if (opts_.export_images && !opts_.output_path.empty()) {
    // cameras.bin — written once; intrinsics must be loaded before this call
    {
      const std::string cam_path = opts_.output_path + "/sparse/0/cameras.bin";
      std::ofstream f(cam_path, std::ios::binary);
      if (!f.is_open()) {
        ROS_ERROR_STREAM("[output] failed to open " << cam_path << " — disabling image export");
        opts_.export_images = false;
      } else {
        const uint64_t n_cams = 1;
        f.write(reinterpret_cast<const char*>(&n_cams), 8);
        const uint32_t cam_id = 1;
        const int32_t  model  = 1;  // PINHOLE
        f.write(reinterpret_cast<const char*>(&cam_id), 4);
        f.write(reinterpret_cast<const char*>(&model),  4);
        const uint64_t w = static_cast<uint64_t>(state_->cameraWidth());
        const uint64_t h = static_cast<uint64_t>(state_->cameraHeight());
        f.write(reinterpret_cast<const char*>(&w), 8);
        f.write(reinterpret_cast<const char*>(&h), 8);
        const double params[4] = { state_->cameraFx(), state_->cameraFy(),
                                   state_->cameraCx(), state_->cameraCy() };
        f.write(reinterpret_cast<const char*>(params), 32);
      }
    }
    // images.bin — open for streaming; placeholder count patched on shutdown
    if (opts_.export_images) {
      const std::string img_path = opts_.output_path + "/sparse/0/images.bin";
      images_file_.open(img_path, std::ios::binary);
      if (!images_file_.is_open()) {
        ROS_ERROR_STREAM("[output] failed to open " << img_path << " — disabling image export");
        opts_.export_images = false;
      } else {
        const uint64_t placeholder = 0;
        images_file_.write(reinterpret_cast<const char*>(&placeholder), 8);
      }
    }
  }

  if (opts_.viz_plane_map)
    pub_geom_ = pnh_.advertise<visualization_msgs::MarkerArray>("plane_markers", 1);
  else
    pub_geom_ = pnh_.advertise<sensor_msgs::PointCloud2>("cloud_registered", 10);
  pub_odom_      = pnh_.advertise<nav_msgs::Odometry>("odometry", 10);
  pub_path_      = pnh_.advertise<nav_msgs::Path>("path", 10);
  pub_image_     = pnh_.advertise<sensor_msgs::Image>("rgb_img", 1);
  pub_timestamp_ = pnh_.advertise<visualization_msgs::Marker>("timestamp_marker", 1);
  pub_evo_stats_ = pnh_.advertise<jsk_rviz_plugins::OverlayText>("evo_stats_overlay", 1);

  path_.header.frame_id = "camera_init";

  if (opts_.export_pcd || opts_.export_images)
    write_thread_ = std::thread(&PubProc::writeWorker, this);

  std::ostringstream oss;
  oss << "[params/pub]"
      << "\n  viz/enable:          " << (opts_.viz_enable       ? "true" : "false")
      << "\n  viz/ds/ds_leaf_size: " << opts_.vis_ds_leaf_size
      << "\n  viz/grid_map:        " << (opts_.viz_grid_map    ? "true" : "false")
      << "\n  viz/plane_map:       " << (opts_.viz_plane_map   ? "true" : "false")
      << "\n  outputs/path:        " << opts_.output_path
      << "\n  outputs/pcd/export:  " << (opts_.export_pcd    ? "true" : "false")
      << "\n  outputs/image/export:" << (opts_.export_images ? "true" : "false")
      << "\n  outputs/image/interval:" << opts_.export_image_interval
      << "\n  outputs/odom/export: " << (opts_.export_odom && odom_file_.is_open() ? "true" : "false");
  return oss.str();
}

void PubProc::writeWorker()
{
  while (true) {
    WriteTask task;
    {
      std::unique_lock<std::mutex> lock(write_mutex_);
      write_cv_.wait(lock, [this]{ return !write_queue_.empty() || write_stop_; });
      if (write_stop_ && write_queue_.empty()) break;
      task = std::move(write_queue_.front());
      write_queue_.pop();
    }

    if (opts_.export_pcd && points3d_file_.is_open()) {
      for (const auto& pt : task.pcd_points) {
        const uint64_t pt_id = ++points3d_count_;
        points3d_file_.write(reinterpret_cast<const char*>(&pt_id), 8);
        const double xyz[3] = { pt.x, pt.y, pt.z };
        points3d_file_.write(reinterpret_cast<const char*>(xyz), 24);
        const uint8_t rgb[3] = { pt.r, pt.g, pt.b };
        points3d_file_.write(reinterpret_cast<const char*>(rgb), 3);
        const double error = 0.0;
        points3d_file_.write(reinterpret_cast<const char*>(&error), 8);
        const uint64_t track_len = 0;
        points3d_file_.write(reinterpret_cast<const char*>(&track_len), 8);
      }
    }

    if (task.write_image && images_file_.is_open()) {
      cv::imwrite(opts_.output_path + "/images/" + task.img_name, task.image);
      const uint32_t img_id = static_cast<uint32_t>(++images_count_);
      images_file_.write(reinterpret_cast<const char*>(&img_id), 4);
      const double qvec[4] = { task.qcw.w(), task.qcw.x(), task.qcw.y(), task.qcw.z() };
      images_file_.write(reinterpret_cast<const char*>(qvec), 32);
      const double tvec[3] = { task.tcw.x(), task.tcw.y(), task.tcw.z() };
      images_file_.write(reinterpret_cast<const char*>(tvec), 24);
      const uint32_t cam_id = 1;
      images_file_.write(reinterpret_cast<const char*>(&cam_id), 4);
      images_file_.write(task.img_name.c_str(), task.img_name.size() + 1);
      const uint64_t n_pts2d = 0;
      images_file_.write(reinterpret_cast<const char*>(&n_pts2d), 8);
    }
  }
}

void PubProc::publishResults(const MeasureGroup& mg)
{
  printer_->print(PrintCategory::PROFILER, profiler_->report());
  printer_->print(PrintCategory::STATE, state_->toString());
  printer_->print(PrintCategory::VOXEL, voxel_map_->statsString());

  publishOdometry(mg);
  enqueueExport(mg);

  if (opts_.viz_enable)
  {
    publishGeometry(mg);
    publishImage(mg);
    publishTimestamp(mg);
    publishEvoStats(mg);
  }
  else if (frame_count_ % 100 == 0)
  {
    // Nothing to watch in rviz with viz disabled (e.g. headless test-script
    // runs) -- print periodic progress instead so a run's liveness/pace is
    // still visible.
    std::ostringstream oss;
    oss << "[progress]  t=" << std::fixed << std::setprecision(1) << mg.image.t
        << "s  frame=" << frame_count_;
    printer_->print(PrintCategory::PARAMS, oss.str());
  }

  ++frame_count_;
}

void PubProc::publishOdometry(const MeasureGroup& mg)
{
  const ros::Time stamp(mg.image.t + data_queues_->start_time);
  const Eigen::Quaterniond q(state_->rot());

  if (odom_file_.is_open()) {
    odom_file_ << std::fixed << std::setprecision(9)
               << stamp.toSec() << " "
               << state_->pos().x() << " " << state_->pos().y() << " " << state_->pos().z() << " "
               << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    odom_file_.flush();
  }

  // P8.  Posterior columns mirror odom_file_'s own row exactly (same
  // source, same precision) -- see P8's own verify text: they must equal
  // the .tum to full precision, which is the cross-check that this file and
  // odometry.txt are looking at the same scan's same state.
  if (pose_pair_file_.is_open()) {
    const Eigen::Quaterniond pq(mg.prior_rot);
    pose_pair_file_ << std::fixed << std::setprecision(9)
        << stamp.toSec() << ","
        << state_->pos().x() << "," << state_->pos().y() << "," << state_->pos().z() << ","
        << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << ","
        << mg.prior_pos.x() << "," << mg.prior_pos.y() << "," << mg.prior_pos.z() << ","
        << pq.x() << "," << pq.y() << "," << pq.z() << "," << pq.w() << ","
        << state_->vel().x() << "," << state_->vel().y() << "," << state_->vel().z() << ","
        << mg.prior_vel.x() << "," << mg.prior_vel.y() << "," << mg.prior_vel.z() << "\n";
    pose_pair_file_.flush();
  }

  if (!opts_.viz_enable) return;

  tf::Transform tf_transform;
  tf_transform.setOrigin(tf::Vector3(state_->pos().x(), state_->pos().y(), state_->pos().z()));
  tf_transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
  tf_br_.sendTransform(tf::StampedTransform(tf_transform, stamp, "camera_init", "body"));

  nav_msgs::Odometry odom;
  odom.header.stamp    = stamp;
  odom.header.frame_id = "camera_init";
  odom.child_frame_id  = "body";
  odom.pose.pose.position.x    = state_->pos().x();
  odom.pose.pose.position.y    = state_->pos().y();
  odom.pose.pose.position.z    = state_->pos().z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  pub_odom_.publish(odom);

  geometry_msgs::PoseStamped ps;
  ps.header = odom.header;
  ps.pose   = odom.pose.pose;
  path_.header.stamp = stamp;
  path_.poses.push_back(ps);
  pub_path_.publish(path_);
}

void PubProc::publishGeometry(const MeasureGroup& mg)
{
  const ros::Time stamp(mg.image.t + data_queues_->start_time);

  if (opts_.viz_plane_map) {
    if (voxel_map_->vizDirty()) {
      TimedScope ts(profiler_, "publish/plane_markers");
      pub_geom_.publish(voxel_map_->buildVizMarkers("camera_init", stamp));
    }
    return;
  }

  TimedScope ts(profiler_, "publish/cloud");
  const int  img_rows = mg.image.image.rows;
  const int  img_cols = mg.image.image.cols;
  const bool do_ds    = opts_.vis_ds_leaf_size > 0.0;

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cloud.points.reserve(mg.points.size());

  for (const auto& src : mg.points) {
    const V3D p_world = state_->bodyToWorld(src.point);
    const V3D p_cam   = state_->imuToCamera(src.point);
    if (p_cam.z() <= 0) continue;
    const int u = static_cast<int>(std::round(state_->cameraFx() * p_cam.x() / p_cam.z() + state_->cameraCx()));
    const int v = static_cast<int>(std::round(state_->cameraFy() * p_cam.y() / p_cam.z() + state_->cameraCy()));
    if (u < 0 || v < 0 || u >= img_cols || v >= img_rows) continue;
    if (do_ds && !vis_ds_seen_.insert(worldToKeyFn(p_world, opts_.vis_ds_leaf_size)).second) continue;
    const cv::Vec3b& px = mg.image.image.at<cv::Vec3b>(v, u);
    pcl::PointXYZRGB pt;
    pt.x = static_cast<float>(p_world.x());
    pt.y = static_cast<float>(p_world.y());
    pt.z = static_cast<float>(p_world.z());
    pt.r = px[2]; pt.g = px[1]; pt.b = px[0];
    cloud.points.push_back(pt);
  }

  cloud.width    = static_cast<uint32_t>(cloud.points.size());
  cloud.height   = 1;
  cloud.is_dense = false;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  cloud_msg.header.stamp    = stamp;
  cloud_msg.header.frame_id = "camera_init";
  pub_geom_.publish(cloud_msg);
}


void PubProc::publishImage(const MeasureGroup& mg)
{
  const ros::Time stamp(mg.image.t + data_queues_->start_time);
  cv_bridge::CvImage cv_img;
  cv_img.header.stamp    = stamp;
  cv_img.header.frame_id = "camera";
  cv_img.encoding        = "bgr8";
  const cv::Mat uv_vis = tracker_->maybeDrawUvTracking(mg.image.image);
  if (!uv_vis.empty())
    cv_img.image = uv_vis;
  else if (opts_.viz_grid_map && !mg.grid_vis_image.empty())
    cv_img.image = mg.grid_vis_image;
  else
    cv_img.image = mg.image.image;
  pub_image_.publish(*cv_img.toImageMsg());
}

void PubProc::publishTimestamp(const MeasureGroup& mg)
{
  const ros::Time stamp(mg.image.t + data_queues_->start_time);

  std::ostringstream ss;
  ss << std::fixed << std::setprecision(3) << mg.image.t << " s";

  visualization_msgs::Marker marker;
  marker.header.stamp    = stamp;
  marker.header.frame_id = "camera_init";
  marker.ns              = "timestamp";
  marker.id              = 0;
  marker.type            = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action          = visualization_msgs::Marker::ADD;
  marker.pose.position.x = state_->pos().x();
  marker.pose.position.y = state_->pos().y();
  marker.pose.position.z = state_->pos().z() + 1.5;
  marker.pose.orientation.w = 1.0;
  marker.scale.z         = 0.4;
  marker.color.r         = 1.0;
  marker.color.g         = 1.0;
  marker.color.b         = 1.0;
  marker.color.a         = 1.0;
  marker.text            = ss.str();
  pub_timestamp_.publish(marker);
}

void PubProc::publishEvoStats(const MeasureGroup& mg)
{
  // Empty until evo/enable is on AND enough poses have been matched to
  // align (see EvoProc::processEvo) -- e.g. slam_2022/slam_2023 (no ground
  // truth topic) never populate this, so this overlay simply never appears
  // for those datasets rather than showing stale/zeroed numbers.
  if (mg.evo_stats.empty())
    return;

  // jsk_rviz_plugins/OverlayText: a fixed screen-space (not 3D-world)
  // text box -- add the "jsk_rviz_plugins/OverlayText" display in rviz,
  // pointed at this topic, to see it as a stable on-screen table rather
  // than a 3D marker that shrinks/rotates with the camera.
  jsk_rviz_plugins::OverlayText overlay;
  overlay.action     = jsk_rviz_plugins::OverlayText::ADD;
  // 4 metrics laid out in columns (header row + value row, see
  // EvoProc::processEvo) rather than one per row -- wide and short instead
  // of narrow and tall, to fit comfortably.
  overlay.width       = 700;
  overlay.height      = 70;
  overlay.left        = 10;
  overlay.top         = 10;
  overlay.text_size   = 14;
  overlay.font        = "DejaVu Sans Mono";
  overlay.bg_color.r  = 0.0f; overlay.bg_color.g = 0.0f; overlay.bg_color.b = 0.0f; overlay.bg_color.a = 0.6f;
  overlay.fg_color.r  = 1.0f; overlay.fg_color.g = 0.7f; overlay.fg_color.b = 0.0f; overlay.fg_color.a = 1.0f;
  overlay.text        = mg.evo_stats;
  pub_evo_stats_.publish(overlay);
}

void PubProc::enqueueExport(const MeasureGroup& mg)
{
  const bool do_pcd   = opts_.export_pcd   && points3d_file_.is_open();
  const bool do_image = opts_.export_images && images_file_.is_open()
                        && frame_count_ % opts_.export_image_interval == 0;
  if (!do_pcd && !do_image) return;

  WriteTask task;

  if (do_pcd) {
    const int img_rows = mg.image.image.rows;
    const int img_cols = mg.image.image.cols;
    for (const auto& src : mg.points) {
      const V3D p_world = state_->bodyToWorld(src.point);
      const V3D p_cam   = state_->imuToCamera(src.point);
      if (p_cam.z() <= 0) continue;
      const int u = static_cast<int>(std::round(state_->cameraFx() * p_cam.x() / p_cam.z() + state_->cameraCx()));
      const int v = static_cast<int>(std::round(state_->cameraFy() * p_cam.y() / p_cam.z() + state_->cameraCy()));
      if (u < 0 || v < 0 || u >= img_cols || v >= img_rows) continue;
      const cv::Vec3b& px = mg.image.image.at<cv::Vec3b>(v, u);
      pcl::PointXYZRGB pt;
      pt.x = static_cast<float>(p_world.x());
      pt.y = static_cast<float>(p_world.y());
      pt.z = static_cast<float>(p_world.z());
      pt.r = px[2]; pt.g = px[1]; pt.b = px[0];
      task.pcd_points.push_back(pt);
    }
  }

  if (do_image) {
    std::ostringstream img_ss;
    img_ss << std::setw(5) << std::setfill('0') << export_image_count_ << ".png";
    task.img_name    = img_ss.str();
    task.write_image = true;
    task.image       = mg.image.image.clone();
    task.tcw         = state_->tcw();
    task.qcw         = Eigen::Quaterniond(state_->Rcw());
    ++export_image_count_;
  }

  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    write_queue_.push(std::move(task));
  }
  write_cv_.notify_one();
}

void PubProc::exportColmap()
{
  // Drain the write queue and join the worker before touching files or counters
  if (write_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(write_mutex_);
      write_stop_ = true;
    }
    write_cv_.notify_one();
    write_thread_.join();
  }

  if (odom_file_.is_open()) odom_file_.close();
  if (pose_pair_file_.is_open()) pose_pair_file_.close();

  if (!opts_.export_pcd && !opts_.export_images) return;
  if (opts_.output_path.empty()) {
    printer_->print(PrintCategory::PARAMS, "[colmap] output/path not set; skipping export");
    return;
  }

  // images.bin — patch count header, then close
  if (opts_.export_images && images_file_.is_open()) {
    images_file_.seekp(0);
    images_file_.write(reinterpret_cast<const char*>(&images_count_), 8);
    images_file_.close();
  }

  // points3D.bin — patch count header, then close
  if (opts_.export_pcd && points3d_file_.is_open()) {
    points3d_file_.seekp(0);
    points3d_file_.write(reinterpret_cast<const char*>(&points3d_count_), 8);
    points3d_file_.close();
  } else if (opts_.export_images) {
    // No pcd export — write an empty points3D.bin so COLMAP can still load the dataset
    namespace fs = std::filesystem;
    const fs::path sparse_dir = fs::path(opts_.output_path) / "sparse" / "0";
    std::ofstream f(sparse_dir / "points3D.bin", std::ios::binary);
    const uint64_t n_pts = 0;
    f.write(reinterpret_cast<const char*>(&n_pts), 8);
  }

  std::ostringstream oss;
  oss << "[colmap] export: " << opts_.output_path
      << "  points=" << points3d_count_
      << "  images=" << images_count_;
  printer_->print(PrintCategory::PARAMS, oss.str());
}

}  // namespace livo_recon
