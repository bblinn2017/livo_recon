#pragma once

#include <opencv2/opencv.hpp>
#include "livo_recon/utils/algo/math.h"

namespace livo_recon
{

// Wrapper for image with timestamp
struct ImageData
{
  cv::Mat image;
  double t;
};

// Wrapper for imu sample with timestamp
struct ImuSample
{
  V3D acc = V3D::Zero();
  V3D gyro = V3D::Zero();
  double t = 0.;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuSample() = default;

  ImuSample(const V3D& a, const V3D& g, double t_)
    : acc(a), gyro(g), t(t_) {}
};

// Wrapper for point with timestamp
struct PointXYZT
{
  V3D p;   // x, y, z
  double t;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointXYZT() = default;

  PointXYZT(const V3D& p_, double t_)
    : p(p_), t(t_) {}
};

// Wrapper for point with covariance
struct PointXYZCov {
  V3D point;

  // PURE sensor+deskew measurement-noise covariance (independent per
  // point) -- rotated body->world by StateGroup::toWorld()/toCamera().
  // NEVER has pose uncertainty folded in, and never mutated downstream --
  // see `pos_cov` below for the (always separate, always explicit) other
  // component. A consumer needing the isotropic (trace/3) proxy computes
  // it inline from this (no separate stored scalar -- one source of truth).
  M3D sensor_cov;

  // The current state's pose (rotation+position) uncertainty's
  // contribution to this point's world-frame position (StateGroup::
  // poseCovAt()), captured DIRECTLY at the same point sensor_cov is set
  // (VoxelMap::updateMap()) -- never derived by subtracting one component
  // from a combined total (numerically fragile, and unnecessary now that
  // both are captured independently at their natural source). SHARED
  // across every point in one frame's shared pose estimate, unlike
  // sensor_cov (independent per point) -- still varies point-to-point
  // itself via poseCovAt()'s own lever-arm dependence on each point's
  // body-frame offset. M3D::Zero() default until VoxelMap::updateMap()
  // sets it (e.g. calib's skipCalibration() path never does). A consumer
  // needing the combined sensor+pose covariance (e.g. the PCA plane fit's
  // residual-weighting propagation) computes `sensor_cov + pos_cov`
  // explicitly at the point of use -- mirrors WorldPointCov's existing
  // sensor_cov/pose_cov/total() pattern, which already got this right.
  M3D pos_cov = M3D::Zero();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointXYZCov() = default;

  PointXYZCov(const V3D& p, const M3D& c)
    : point(p), sensor_cov(c) {}
};

struct PointWithNormal
{
  V3D point_w;
  V3D normal;
  M3D cov_w = M3D::Zero();
  V2D pixel = V2D::Zero();
};

// Wrapper for a single ground-truth pose sample (evo/gt_source=="topic",
// e.g. NTU_VIRAL's live Leica stream) -- raw ingestion lives in CbkProc
// (same as ImageData/ImuSample/PointXYZT), t already start_time-relative
// (see CbkProc::gtCallback()'s doc comment), consumed by EvoProc via
// DataQueues::popGt().
struct GtPoseSample
{
  V3D    pos = V3D::Zero();
  M3D    rot = M3D::Identity();  // identity if the topic doesn't provide a meaningful orientation
  double t   = 0.;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GtPoseSample() = default;

  GtPoseSample(const V3D& p, const M3D& r, double t_)
    : pos(p), rot(r), t(t_) {}
};

struct Pose6D
{
  double t;
  V3D    acc_head;  // world-frame acc at head time  (= R_head * a_body + g)
  V3D    acc_tail;  // world-frame acc at tail time  (= R_tail * a_body + g)
  V3D    gyr;
  V3D    vel;
  V3D    pos;
  M3D    rot;
  double dt;        // interval duration [s] — used for acc lerp in undistortion

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Pose6D() = default;

  Pose6D(double t_,
         const V3D& acc_head_,
         const V3D& acc_tail_,
         const V3D& gyr_,
         const V3D& vel_,
         const V3D& pos_,
         const M3D& rot_,
         double dt_)
    : t(t_), acc_head(acc_head_), acc_tail(acc_tail_), gyr(gyr_),
      vel(vel_), pos(pos_), rot(rot_), dt(dt_) {}
};

}
