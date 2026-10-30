#include "livo_recon/calib/trajectory_refine.h"
#include "livo_recon/vio/tracker_cache.h"

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>

namespace livo_recon
{
namespace calib
{

namespace
{

// --- cumulative cubic B-spline (uniform), w-first quaternions -- ported
// verbatim from LiDAR_IMU_Init's ct_refine.cpp (see trajectory_refine.h's
// header comment). Pure math, no dependency on the point source or the
// association mechanism, so nothing here needed to change. -------------
template <typename T>
void quatConj(const T* q, T* c)
{
  c[0] = q[0];
  c[1] = -q[1];
  c[2] = -q[2];
  c[3] = -q[3];
}

template <typename T>
void splineEval(const T* const* q, const T* const* p, double u, T* qo, T* po)
{
  const double u2 = u * u, u3 = u2 * u;
  const double b[3] = {(5 + 3 * u - 3 * u2 + u3) / 6.0, (1 + 3 * u + 3 * u2 - 2 * u3) / 6.0, u3 / 6.0};
  for (int i = 0; i < 3; i++) po[i] = p[0][i];
  for (int j = 1; j <= 3; j++)
    for (int i = 0; i < 3; i++) po[i] += b[j - 1] * (p[j][i] - p[j - 1][i]);
  T qa[4] = {q[0][0], q[0][1], q[0][2], q[0][3]};
  for (int j = 1; j <= 3; j++) {
    T c[4], d[4], aa[3], e[4], tmp[4];
    quatConj(q[j - 1], c);
    ceres::QuaternionProduct(c, q[j], d);
    ceres::QuaternionToAngleAxis(d, aa);
    for (int i = 0; i < 3; i++) aa[i] *= T(b[j - 1]);
    ceres::AngleAxisToQuaternion(aa, e);
    ceres::QuaternionProduct(qa, e, tmp);
    for (int i = 0; i < 4; i++) qa[i] = tmp[i];
  }
  for (int i = 0; i < 4; i++) qo[i] = qa[i];
}

// Point-to-plane residual against a FIXED {normal, d} (one outer
// iteration's association, held constant while the spline is optimized --
// see trajectory_refine.h's header comment for why this is safe/standard
// for this class of problem). Plane convention matches VoxelPlane::gate()'s
// own: n . y + d = 0, i.e. r = n . y + d (NOT li_init's n.y - d -- see
// refineTrajectory()'s derivation of d from a Residual's own r/normal).
struct PlaneCost
{
  double x[3], n[3], d, u, w;   // point (body frame, own capture time), plane n.y+d=0, segment time, weight
  template <typename T>
  bool operator()(const T* q0, const T* q1, const T* q2, const T* q3, const T* p0, const T* p1, const T* p2,
                   const T* p3, T* r) const
  {
    const T* q[4] = {q0, q1, q2, q3};
    const T* p[4] = {p0, p1, p2, p3};
    T qo[4], po[3], xw[3];
    splineEval(q, p, u, qo, po);
    const T xl[3] = {T(x[0]), T(x[1]), T(x[2])};
    ceres::UnitQuaternionRotatePoint(qo, xl, xw);
    r[0] = T(w) * (T(n[0]) * (xw[0] + po[0]) + T(n[1]) * (xw[1] + po[1]) + T(n[2]) * (xw[2] + po[2]) + T(d));
    return true;
  }
};

struct KnotPrior
{
  double q0[4], p0[3], wr, wp;
  template <typename T>
  bool operator()(const T* q, const T* p, T* r) const
  {
    const T c[4] = {T(q0[0]), T(-q0[1]), T(-q0[2]), T(-q0[3])};
    T d[4], aa[3];
    ceres::QuaternionProduct(c, q, d);
    ceres::QuaternionToAngleAxis(d, aa);
    for (int i = 0; i < 3; i++) {
      r[i] = T(wr) * aa[i];
      r[3 + i] = T(wp) * (p[i] - T(p0[i]));
    }
    return true;
  }
};

M3D quatToR(const std::array<double, 4>& q)
{
  return Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized().toRotationMatrix();
}

struct Assoc
{
  size_t i;     // index into `points`
  V3D n;
  double d;     // plane n.y + d = 0
  double w;     // 1/sqrt(sigma_squared), from MapBackend::findPlaneResidual()
  int32_t node_id = -1;   // Residual::plane_node_id -- stable per-voxel diagnostic identity,
                          // for grouping dumped points by which voxel/plane they matched (see
                          // opt.debug_dump_path's doc comment)
  double plane_var_term = 0.0;   // Residual::plane_var_term -- this plane's own fit-uncertainty
                                  // contribution, isolated from sigma_squared (which also folds in
                                  // floor_term/sigma_diag_squared), to check its calibration
                                  // against the REAL observed per-voxel residual variance
  double sigma_diag_squared = 0.0;   // Residual::sigma_diag_squared -- the per-point sensor-noise
                                      // term (n^T * getBodyCov() * n, this point's raw/tangential
                                      // range covariance PROJECTED onto the matched plane's normal)
                                      // -- this already varies with incidence angle purely from the
                                      // projection geometry, even with no explicit incidence term,
                                      // since getBodyCov() is anisotropic (radial vs tangential)
  double floor_term = 0.0;   // Residual::floor_term -- weightFloor()'s own contribution (sensor_range
                              // mode: a flat range_err^2 floor, independent of incidence/geometry)
};

// --- visual (epipolar) correction, ported from LiDAR_IMU_Init's
// ct_refine.cpp CameraEpipolarCost/prepareCameraRaysFromTrackCache (which
// was itself derived from livo_recon's own VioAccumulator epipolar
// geometry, see trajectory_refine.h's TrajectoryRefineOptions::Camera doc
// comment) -- rewritten here as a Ceres-AutoDiff residual over two FREE
// spline poses, reading matches straight from livo_recon's OWN
// TrackerCacheReader/.rlivtrackcache instead of a separate parser. -------

struct CameraRay
{
  double t_a, t_b;   // anchor (seed) time, tracked time
  V3D xa, xb;         // normalized bearing rays (z=1), already undistorted
};

// livo_recon's TrackedFrame stores uv_seed/uv_curr already distortion-
// corrected (Tracker::undistortPixel(), see cache_tracker_output.cpp's
// format doc comment) -- so building a normalized bearing ray here is just
// the pinhole subtract-cx/divide-fx step, no distortion iteration needed.
std::vector<CameraRay> loadCameraRays(const TrajectoryRefineOptions::Camera& cam, std::string& err)
{
  std::vector<CameraRay> out;
  TrackerCacheReader reader;
  if (!reader.load(cam.track_cache_path, err)) return out;

  std::map<std::pair<double, double>, std::vector<CameraRay>> by_pair;
  for (size_t i = 0; i < reader.size(); i++) {
    const TrackedFrame& f = reader.at(i);
    if (!f.ok || f.prev_timestamp < 0) continue;
    for (size_t k = 0; k < f.indices.size(); k++) {
      const AnchorPoint& a = f.anchors[f.indices[k]];
      CameraRay ray;
      ray.t_a = f.prev_timestamp;
      ray.t_b = f.timestamp;
      ray.xa = V3D((a.uv_seed.x - cam.cx) / cam.fx, (a.uv_seed.y - cam.cy) / cam.fy, 1.0);
      ray.xb = V3D((f.uv_curr[k].x - cam.cx) / cam.fx, (f.uv_curr[k].y - cam.cy) / cam.fy, 1.0);
      by_pair[{ray.t_a, ray.t_b}].push_back(ray);
    }
  }
  std::mt19937 rng(3);
  size_t n_ransac_in = 0, n_ransac_total = 0, n_pairs_ransaced = 0;
  const double pixel_scale = 0.5 * (cam.fx + cam.fy);
  for (auto& kv : by_pair) {
    std::vector<CameraRay>& v = kv.second;

    // (1) Essential-matrix RANSAC per pair -- see TrajectoryRefineOptions::
    // Camera::ransac_threshold_px's doc comment. Points are already
    // normalized bearings (z=1), so findEssentialMat uses an identity
    // camera matrix and the threshold is converted from pixels into that
    // same normalized scale via pixel_scale.
    if (cam.ransac_threshold_px > 0 && static_cast<int>(v.size()) >= cam.ransac_min_matches) {
      cv::Mat p1(static_cast<int>(v.size()), 2, CV_64F), p2(static_cast<int>(v.size()), 2, CV_64F);
      for (size_t i = 0; i < v.size(); i++) {
        p1.at<double>(static_cast<int>(i), 0) = v[i].xa(0);
        p1.at<double>(static_cast<int>(i), 1) = v[i].xa(1);
        p2.at<double>(static_cast<int>(i), 0) = v[i].xb(0);
        p2.at<double>(static_cast<int>(i), 1) = v[i].xb(1);
      }
      cv::Mat mask;
      const cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
      cv::Mat E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, cam.ransac_threshold_px / pixel_scale, mask);
      n_ransac_total += v.size();
      if (!E.empty() && mask.rows == static_cast<int>(v.size())) {
        std::vector<CameraRay> inliers;
        for (size_t i = 0; i < v.size(); i++)
          if (mask.at<uchar>(static_cast<int>(i))) inliers.push_back(v[i]);
        n_ransac_in += inliers.size();
        v = inliers;
        n_pairs_ransaced++;
      }
    }

    if (static_cast<int>(v.size()) > cam.max_matches_per_pair) {
      std::shuffle(v.begin(), v.end(), rng);
      v.resize(cam.max_matches_per_pair);
    }
    for (const CameraRay& r : v) out.push_back(r);
  }
  if (n_ransac_total > 0)
    std::fprintf(stderr, "  camera residual RANSAC: %zu/%zu matches kept as inliers across %zu pairs (threshold %.2fpx)\n",
                 n_ransac_in, n_ransac_total, n_pairs_ransaced, cam.ransac_threshold_px);
  std::sort(out.begin(), out.end(), [](const CameraRay& a, const CameraRay& b) { return a.t_a < b.t_a; });
  return out;
}

// qo/po (LiDAR-frame spline pose) -> camera world pose, via the fixed
// R_l2c/t_l2c extrinsic. q_l2c_conj is the (constant) conjugate quaternion
// of R_l2c, precomputed once by the caller (cheap, not part of the
// AutoDiff chain).
template <typename T>
void lidarPoseToCameraPose(const T* qo, const T* po, const double* q_l2c_conj, const T* t_l2c_T, T* q_cam,
                           T* p_cam)
{
  const T qc_T[4] = {T(q_l2c_conj[0]), T(q_l2c_conj[1]), T(q_l2c_conj[2]), T(q_l2c_conj[3])};
  ceres::QuaternionProduct(qo, qc_T, q_cam);   // q_cam = qo (x) conj(q_l2c)  [R_cam = R_lidar * R_l2c^T]
  T Rt[3];
  ceres::UnitQuaternionRotatePoint(q_cam, t_l2c_T, Rt);   // R_cam * t_l2c
  for (int i = 0; i < 3; i++) p_cam[i] = po[i] - Rt[i];
}

// Epipolar residual between the spline-interpolated camera poses at t_a,
// t_b -- see trajectory_refine.h's TrajectoryRefineOptions::Camera doc
// comment for the derivation. Dynamic parameter count: parameters[0..m-1]
// are the m UNIQUE knot quaternion blocks this residual's two (overlapping
// or not) 4-control-point windows touch, parameters[m..2m-1] the matching
// position blocks, in the same order -- la[]/lb[] index into 0..m-1 for
// frame A's / frame B's own 4 control points.
struct CameraEpipolarCost
{
  double u_a, u_b;
  int la[4], lb[4];
  int m;
  V3D xa, xb;
  double q_l2c_conj[4];
  V3D t_l2c;
  double weight;   // weight_px * pixel_scale, folded into one constant
  template <typename T>
  bool operator()(T const* const* params, T* r) const
  {
    const T *qa[4], *pa[4], *qb[4], *pb[4];
    for (int i = 0; i < 4; i++) { qa[i] = params[la[i]]; pa[i] = params[m + la[i]]; }
    for (int i = 0; i < 4; i++) { qb[i] = params[lb[i]]; pb[i] = params[m + lb[i]]; }
    T qo_a[4], po_a[3], qo_b[4], po_b[3];
    splineEval(qa, pa, u_a, qo_a, po_a);
    splineEval(qb, pb, u_b, qo_b, po_b);

    const T tl2c[3] = {T(t_l2c(0)), T(t_l2c(1)), T(t_l2c(2))};
    T q_cam_a[4], p_cam_a[3], q_cam_b[4], p_cam_b[3];
    lidarPoseToCameraPose(qo_a, po_a, q_l2c_conj, tl2c, q_cam_a, p_cam_a);
    lidarPoseToCameraPose(qo_b, po_b, q_l2c_conj, tl2c, q_cam_b, p_cam_b);

    // R_cc = R_cam_b^T * R_cam_a  (as quaternion: conj(q_cam_b) (x) q_cam_a)
    T qb_conj[4] = {q_cam_b[0], -q_cam_b[1], -q_cam_b[2], -q_cam_b[3]};
    T q_cc[4];
    ceres::QuaternionProduct(qb_conj, q_cam_a, q_cc);
    // t_cc = R_cam_b^T * (p_cam_a - p_cam_b)
    T dp[3] = {p_cam_a[0] - p_cam_b[0], p_cam_a[1] - p_cam_b[1], p_cam_a[2] - p_cam_b[2]};
    T t_cc[3];
    ceres::UnitQuaternionRotatePoint(qb_conj, dp, t_cc);

    const T x0[3] = {T(xa(0)), T(xa(1)), T(xa(2))};
    const T x1[3] = {T(xb(0)), T(xb(1)), T(xb(2))};
    T y[3];
    ceres::UnitQuaternionRotatePoint(q_cc, x0, y);
    // l = t_cc x y
    const T l0 = t_cc[1] * y[2] - t_cc[2] * y[1];
    const T l1 = t_cc[2] * y[0] - t_cc[0] * y[2];
    const T l2 = t_cc[0] * y[1] - t_cc[1] * y[0];
    const T norm_l = ceres::sqrt(l0 * l0 + l1 * l1 + T(1e-12));
    const T num = x1[0] * l0 + x1[1] * l1 + x1[2] * l2;
    r[0] = T(weight) * num / norm_l;
    return true;
  }
};

struct CameraResidualStats
{
  int n = 0;
  double mean = 0, median = 0, p90 = 0, max = 0;
};

// Plain-double epipolar residual (absolute pixels), same formula as
// CameraEpipolarCost::operator() but evaluated directly against the
// trajectory's CURRENT poses (no AutoDiff) and converted back to actual
// pixels (dividing out cam_pixel_scale) -- so this number is independent of
// weight_px and tells you how tight the visual correction actually is, not
// how hard it was told to try. Returns < 0 if either endpoint falls outside
// the trajectory's valid range (caller should skip/keep accordingly).
double cameraRayResidualPx(const SplineTrajectory& traj, const CameraRay& ray,
                           const TrajectoryRefineOptions::Camera& cam)
{
  if (!traj.valid(ray.t_a) || !traj.valid(ray.t_b)) return -1.0;
  M3D Ra, Rb;
  V3D pa, pb;
  traj.eval(ray.t_a, Ra, pa);
  traj.eval(ray.t_b, Rb, pb);
  const M3D Rcam_a = Ra * cam.R_l2c.transpose();
  const V3D pcam_a = pa - Rcam_a * cam.t_l2c;
  const M3D Rcam_b = Rb * cam.R_l2c.transpose();
  const V3D pcam_b = pb - Rcam_b * cam.t_l2c;
  const M3D Rcc = Rcam_b.transpose() * Rcam_a;
  const V3D tcc = Rcam_b.transpose() * (pcam_a - pcam_b);
  const V3D y = Rcc * ray.xa;
  const V3D l = tcc.cross(y);
  const double norm_l = std::sqrt(l(0) * l(0) + l(1) * l(1) + 1e-12);
  const double num = ray.xb.dot(l);
  const double pixel_scale = 0.5 * (cam.fx + cam.fy);
  return std::fabs(num / norm_l) * pixel_scale;
}

// Ported from LiDAR_IMU_Init's ct_refine.cpp post-solve diagnostic (same name/fields).
CameraResidualStats cameraResidualStats(const SplineTrajectory& traj, const std::vector<CameraRay>& cam_rays,
                                        const TrajectoryRefineOptions::Camera& cam)
{
  CameraResidualStats out;
  std::vector<double> px_err;
  px_err.reserve(cam_rays.size());
  for (const CameraRay& ray : cam_rays) {
    const double px = cameraRayResidualPx(traj, ray, cam);
    if (px >= 0) px_err.push_back(px);
  }
  if (px_err.empty()) return out;
  std::sort(px_err.begin(), px_err.end());
  auto pct = [&](double q) { return px_err[std::min(px_err.size() - 1, static_cast<size_t>(q * px_err.size()))]; };
  double mean = 0;
  for (double v : px_err) mean += v;
  mean /= px_err.size();
  out.n = static_cast<int>(px_err.size());
  out.mean = mean;
  out.median = pct(0.5);
  out.p90 = pct(0.9);
  out.max = px_err.back();
  return out;
}

}  // namespace

void SplineTrajectory::init(const std::vector<TrajPose>& P, double t_start, double t_end, double dt)
{
  t0_ = t_start;
  dt_ = dt;
  const size_t K = static_cast<size_t>(std::ceil((t_end - t_start) / dt)) + 3;
  t1_ = t0_ + (K - 3) * dt_;
  q_.resize(K);
  p_.resize(K);
  size_t j = 0;
  Eigen::Quaterniond prev(1, 0, 0, 0);
  for (size_t k = 0; k < K; k++) {
    const double tk = t0_ + (static_cast<double>(k) - 1.0) * dt_;
    while (j + 2 < P.size() && P[j + 1].t <= tk) j++;
    double a = (tk - P[j].t) / (P[j + 1].t - P[j].t);
    a = std::min(1.0, std::max(0.0, a));
    Eigen::Quaterniond qa(P[j].rot), qb(P[j + 1].rot);
    Eigen::Quaterniond q = qa.slerp(a, qb).normalized();
    if (q.dot(prev) < 0) q.coeffs() *= -1;
    prev = q;
    V3D p = (1 - a) * P[j].pos + a * P[j + 1].pos;
    q_[k] = {q.w(), q.x(), q.y(), q.z()};
    p_[k] = {p(0), p(1), p(2)};
  }
}

void SplineTrajectory::locate(double t, int& c, double& u) const
{
  const double s = (t - t0_) / dt_;
  c = static_cast<int>(std::floor(s));
  u = s - c;
}

bool SplineTrajectory::valid(double t) const
{
  int c;
  double u;
  locate(t, c, u);
  return c >= 0 && c + 3 < static_cast<int>(q_.size());
}

void SplineTrajectory::eval(double t, M3D& R, V3D& p) const
{
  int c;
  double u;
  locate(t, c, u);
  c = std::min(std::max(c, 0), static_cast<int>(q_.size()) - 4);
  const double* q[4] = {q_[c].data(), q_[c + 1].data(), q_[c + 2].data(), q_[c + 3].data()};
  const double* pp[4] = {p_[c].data(), p_[c + 1].data(), p_[c + 2].data(), p_[c + 3].data()};
  double qo[4], po[3];
  splineEval(q, pp, u, qo, po);
  R = Eigen::Quaterniond(qo[0], qo[1], qo[2], qo[3]).normalized().toRotationMatrix();
  p = V3D(po[0], po[1], po[2]);
}

std::vector<TrajPose> sampleTrajectory(const SplineTrajectory& traj, double rate, const std::vector<double>* times)
{
  std::vector<double> ts;
  if (times) {
    for (double t : *times)
      if (traj.valid(t)) ts.push_back(t);
  } else {
    for (double t = traj.tStart(); traj.valid(t); t += 1.0 / rate) ts.push_back(t);
  }
  std::vector<TrajPose> out;
  out.reserve(ts.size());
  for (double t : ts) {
    TrajPose pose;
    pose.t = t;
    traj.eval(t, pose.rot, pose.pos);
    out.push_back(pose);
  }
  return out;
}

TrajectoryRefineReport refineTrajectory(const std::vector<TrajPoint>& points, const std::vector<TrajPose>& init,
                                         MapBackend& map, SplineTrajectory& traj, const TrajectoryRefineOptions& opt)
{
  (void)init;   // traj is already init()'d by the caller; kept in the signature for API
                // symmetry with ct_refine.cpp's refineTrajectory() and in case a future
                // rigid_still auto-detect fallback (ct_refine's longestStill()) is added.
  auto t_begin = std::chrono::steady_clock::now();
  TrajectoryRefineReport rep;
  const size_t K = traj.size();

  // Known-still knots: held at one shared pose (the Lie-algebra mean of
  // their own initial values) for every outer iteration -- ported from
  // ct_refine.cpp's CtOptions::rigid_still, see trajectory_refine.h's doc
  // comment on TrajectoryRefineOptions::t0_still/t1_still for why this
  // matters.
  std::vector<char> still_knot(K, 0);
  if (opt.t0_still >= 0 && opt.t1_still > opt.t0_still) {
    std::vector<int> idx;
    for (size_t k = 0; k < K; k++) {
      const double tk = traj.tStart() + (static_cast<double>(k) - 1.0) * traj.dt();
      if (tk >= opt.t0_still && tk <= opt.t1_still) idx.push_back(static_cast<int>(k));
    }
    if (idx.size() >= 2) {
      const M3D R0 = quatToR(traj.q_[idx[0]]);
      V3D log_sum = V3D::Zero(), p_sum = V3D::Zero();
      for (int k : idx) {
        const M3D Rk = quatToR(traj.q_[k]);
        const Eigen::AngleAxisd aa(R0.transpose() * Rk);
        log_sum += aa.axis() * aa.angle();
        p_sum += V3D(traj.p_[k][0], traj.p_[k][1], traj.p_[k][2]);
      }
      const V3D log_mean = log_sum / idx.size();
      const M3D R_ref = R0 * Eigen::AngleAxisd(log_mean.norm(), log_mean.normalized()).toRotationMatrix();
      const V3D p_ref = p_sum / idx.size();
      const Eigen::Quaterniond q_ref(R_ref);
      for (int k : idx) {
        traj.q_[k] = {q_ref.w(), q_ref.x(), q_ref.y(), q_ref.z()};
        traj.p_[k] = {p_ref(0), p_ref(1), p_ref(2)};
        still_knot[k] = 1;
      }
      traj.still_t0_ = opt.t0_still;
      traj.still_t1_ = opt.t1_still;
      if (opt.verbose)
        std::fprintf(stderr, "  rigid_still: knots %d..%d (t=%.2f..%.2fs) fixed to one shared pose\n", idx.front(),
                     idx.back(), opt.t0_still, opt.t1_still);
    } else if (opt.verbose) {
      std::fprintf(stderr, "  rigid_still: window given but < 2 knots fall inside it, skipping\n");
    }
  }

  const std::vector<std::array<double, 4>> q_init = traj.q_;
  const std::vector<std::array<double, 3>> p_init = traj.p_;

  // Visual (epipolar) correction setup -- loaded once, reused across every
  // outer iteration (the matches themselves don't change, only which
  // spline poses they're evaluated against). See trajectory_refine.h's
  // TrajectoryRefineOptions::Camera doc comment.
  std::vector<CameraRay> cam_rays;
  double q_l2c_conj[4] = {1, 0, 0, 0};
  double cam_pixel_scale = 1.0;
  if (opt.camera.enable) {
    std::string cam_err;
    cam_rays = loadCameraRays(opt.camera, cam_err);
    if (cam_rays.empty()) {
      std::fprintf(stderr, "  camera residual: no matches loaded from %s (%s) -- disabling\n",
                   opt.camera.track_cache_path.c_str(), cam_err.c_str());
    } else {
      const Eigen::Quaterniond q_l2c(opt.camera.R_l2c);
      q_l2c_conj[0] = q_l2c.w();
      q_l2c_conj[1] = -q_l2c.x();
      q_l2c_conj[2] = -q_l2c.y();
      q_l2c_conj[3] = -q_l2c.z();
      cam_pixel_scale = 0.5 * (opt.camera.fx + opt.camera.fy);
      if (opt.verbose)
        std::fprintf(stderr, "  camera residual: %zu matches loaded from %s\n", cam_rays.size(),
                     opt.camera.track_cache_path.c_str());
    }
  }
  const bool cam_enabled = opt.camera.enable && !cam_rays.empty();
  if (cam_enabled && opt.verbose) {
    const CameraResidualStats pre = cameraResidualStats(traj, cam_rays, opt.camera);
    std::fprintf(stderr, "  pre-solve camera residual (%d matches): mean %.2fpx median %.2fpx p90 %.2fpx max %.2fpx\n",
                 pre.n, pre.mean, pre.median, pre.p90, pre.max);
  }

  for (int outer = 0; outer < opt.outer; outer++) {
    // --- association: evaluate the CURRENT spline at each point's own
    // capture time, rotate {point, sensor_cov} to world frame (same math
    // as StateGroup::toWorld(), just driven by the spline instead of an
    // EKF state), query the already-built (frozen -- never updated here)
    // map for a plane match + its real per-point sigma_squared. ---------
    const int n = static_cast<int>(points.size());
    std::vector<Assoc> per_point(n);
    std::vector<char> hit(n, 0);
    long n_miss = 0;
    #pragma omp parallel for schedule(static) reduction(+:n_miss)
    for (int i = 0; i < n; i++) {
      const TrajPoint& pt = points[i];
      if (!traj.valid(pt.t)) { n_miss++; continue; }
      M3D R;
      V3D p;
      traj.eval(pt.t, R, p);
      WorldPointCov pt_world;
      pt_world.point = R * pt.raw_body_point + p;
      pt_world.sensor_cov = R * pt.sensor_cov * R.transpose();
      pt_world.pose_cov = M3D::Zero();
      pt_world.include_pose_cov_in_gate = false;
      pt_world.body_point = pt.raw_body_point;
      pt_world.rot_transpose = R.transpose();
      Residual res{};
      const bool admitted = map.findPlaneResidual(pt_world, res);
      if (!admitted) { n_miss++; continue; }
      res.sigma_squared += res.plane_var_term;
      if (!std::isfinite(res.sigma_squared) || res.sigma_squared <= 0.0) { n_miss++; continue; }
      // d_plane is defined so that res.r (the UNWEIGHTED signed distance
      // findPlaneResidual() already computed, before this outer
      // iteration's solve changes anything) equals n.pt_world.point +
      // d_plane exactly -- reused directly below for rms_before instead of
      // a redundant second spline evaluation.
      const double d_plane = res.r - res.normal.dot(pt_world.point);
      const double w = 1.0 / std::sqrt(res.sigma_squared);
      if (!std::isfinite(d_plane) || !std::isfinite(w) || !res.normal.allFinite()) { n_miss++; continue; }
      per_point[i] = Assoc{static_cast<size_t>(i), res.normal, d_plane, w, res.plane_node_id, res.plane_var_term,
                           res.sigma_diag_squared, res.floor_term};
      hit[i] = 1;
    }
    std::vector<Assoc> as;
    as.reserve(n - n_miss);
    double rms_before_acc = 0.0;
    for (int i = 0; i < n; i++) {
      if (!hit[i]) continue;
      as.push_back(per_point[i]);
      M3D R;
      V3D p;
      traj.eval(points[i].t, R, p);
      const double r_unweighted = per_point[i].n.dot(R * points[i].raw_body_point + p) + per_point[i].d;
      rms_before_acc += r_unweighted * r_unweighted;
    }
    const double rms_before = as.empty() ? 0.0 : std::sqrt(rms_before_acc / as.size());

    // Diagnostic dump (opt.debug_dump_path, see trajectory_refine.h) --
    // overwritten every outer iteration, so whatever is on disk once the
    // loop ends reflects the LAST iteration that ran (post-convergence
    // association, not a mid-refinement snapshot).
    if (!opt.debug_dump_path.empty()) {
      std::ofstream dbg(opt.debug_dump_path);
      dbg << "# t range_m azimuth_deg cos_incidence residual_m weight node_id normal_x normal_y normal_z plane_var_term sigma_diag_squared floor_term\n";
      dbg << std::setprecision(17);
      for (int i = 0; i < n; i++) {
        if (!hit[i]) continue;
        const TrajPoint& pt = points[i];
        const double range = pt.raw_body_point.norm();
        const double az = std::atan2(pt.raw_body_point.y(), pt.raw_body_point.x()) * 180.0 / M_PI;
        M3D R;
        V3D p;
        traj.eval(pt.t, R, p);
        const double r_unweighted = per_point[i].n.dot(R * pt.raw_body_point + p) + per_point[i].d;
        // Incidence: angle between the ray (sensor origin -> point, body
        // frame, origin ~ body origin to within cm) and the matched
        // plane's normal (world frame, rotated back to body frame via
        // R^T) -- same convention as the az-incidence heatmap used
        // earlier in this investigation against li_init's residuals.
        const double cos_inc = (range > 1e-6) ? (pt.raw_body_point / range).dot(R.transpose() * per_point[i].n) : 0.0;
        dbg << pt.t << " " << range << " " << az << " " << cos_inc << " " << r_unweighted << " " << per_point[i].w
            << " " << per_point[i].node_id << " " << per_point[i].n(0) << " " << per_point[i].n(1) << " "
            << per_point[i].n(2) << " " << per_point[i].plane_var_term << " " << per_point[i].sigma_diag_squared
            << " " << per_point[i].floor_term << "\n";
      }
    }

    if (opt.verbose)
      std::fprintf(stderr, "outer %d: %zu/%d points matched (%ld missed/rejected)\n", outer + 1, as.size(), n,
                   n_miss);

    // --- windowed ceres solve: knots outside the active window are held
    // constant and the window boundary shifts by half a window on
    // alternate outer iterations (block coordinate descent) -- same
    // structure as ct_refine.cpp's own windowed solve. -------------------
    std::sort(as.begin(), as.end(), [&](const Assoc& x, const Assoc& y) { return points[x.i].t < points[y.i].t; });
    const std::vector<std::array<double, 4>> q_prev = traj.q_;
    const std::vector<std::array<double, 3>> p_prev = traj.p_;
    const int Kw = opt.window_s > 0 ? std::max(8, static_cast<int>(opt.window_s / traj.dt())) : static_cast<int>(K);
    const int shift = (outer % 2) ? Kw / 2 : 0;

    // Per-outer-iteration residual-based outlier rejection (see
    // TrajectoryRefineOptions::Camera::outlier_px_threshold's doc comment):
    // re-evaluate every match against the CURRENT trajectory and drop any
    // whose residual exceeds the threshold FOR THIS ITERATION ONLY -- a
    // match outside the threshold now may re-enter a later iteration once
    // the trajectory improves. Stays sorted by t_a (a stable filter of an
    // already-sorted sequence), so the windowed `ci` indexing below still
    // works unchanged.
    std::vector<CameraRay> cam_rays_active = cam_rays;
    if (cam_enabled && opt.camera.outlier_px_threshold > 0) {
      std::vector<CameraRay> kept;
      kept.reserve(cam_rays.size());
      for (const CameraRay& ray : cam_rays) {
        const double px = cameraRayResidualPx(traj, ray, opt.camera);
        if (px < 0 || px <= opt.camera.outlier_px_threshold) kept.push_back(ray);
      }
      if (opt.verbose)
        std::fprintf(stderr, "  outer %d camera outlier rejection: %zu/%zu matches kept (threshold %.1fpx)\n",
                     outer + 1, kept.size(), cam_rays.size(), opt.camera.outlier_px_threshold);
      cam_rays_active = std::move(kept);
    }

    size_t ai = 0, ci = 0;
    for (int ka = (shift ? -shift : 0); ka < static_cast<int>(K); ka += Kw) {
      const int k_lo = std::max(ka, 0), k_hi = std::min<int>(ka + Kw, static_cast<int>(K));
      ceres::Problem::Options popt;
      popt.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      popt.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
      ceres::Problem problem(popt);
      ceres::CauchyLoss loss(1.0);
      ceres::QuaternionManifold qm;
      std::vector<char> added(K, 0);
      auto addKnot = [&](int k) {
        if (added[k]) return;
        added[k] = 1;
        problem.AddParameterBlock(traj.q_[k].data(), 4, &qm);
        problem.AddParameterBlock(traj.p_[k].data(), 3);
        if (still_knot[k] || k < k_lo || k >= k_hi) {
          problem.SetParameterBlockConstant(traj.q_[k].data());
          problem.SetParameterBlockConstant(traj.p_[k].data());
        }
      };
      while (ai > 0) {
        int c;
        double u;
        traj.locate(points[as[ai - 1].i].t, c, u);
        if (c + 3 < k_lo) break;
        ai--;
      }
      for (; ai < as.size(); ai++) {
        const Assoc& a = as[ai];
        const TrajPoint& pt = points[a.i];
        int c;
        double u;
        traj.locate(pt.t, c, u);
        if (c >= k_hi) break;
        if (c + 3 < k_lo) continue;
        for (int j = 0; j < 4; j++) addKnot(c + j);
        PlaneCost* pc = new PlaneCost{
            {pt.raw_body_point(0), pt.raw_body_point(1), pt.raw_body_point(2)}, {a.n(0), a.n(1), a.n(2)}, a.d, u, a.w};
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<PlaneCost, 1, 4, 4, 4, 4, 3, 3, 3, 3>(pc), &loss,
                                  traj.q_[c].data(), traj.q_[c + 1].data(), traj.q_[c + 2].data(),
                                  traj.q_[c + 3].data(), traj.p_[c].data(), traj.p_[c + 1].data(),
                                  traj.p_[c + 2].data(), traj.p_[c + 3].data());
      }
      // camera residuals: same windowed-touch rule as above, but each one spans TWO segments
      // (t_a's and t_b's), which may or may not overlap -- build the unique control-point set
      // per residual rather than assuming a fixed 8 blocks (see CameraEpipolarCost's doc comment).
      if (cam_enabled) {
        while (ci > 0) {
          int c;
          double u;
          traj.locate(cam_rays_active[ci - 1].t_a, c, u);
          if (c + 3 < k_lo) break;
          ci--;
        }
        for (; ci < cam_rays_active.size(); ci++) {
          const CameraRay& ray = cam_rays_active[ci];
          if (!traj.valid(ray.t_a) || !traj.valid(ray.t_b)) continue;
          int ca, cb;
          double ua, ub;
          traj.locate(ray.t_a, ca, ua);
          traj.locate(ray.t_b, cb, ub);
          ca = std::min(std::max(ca, 0), static_cast<int>(K) - 4);
          cb = std::min(std::max(cb, 0), static_cast<int>(K) - 4);
          if (ca >= k_hi) break;
          if (std::max(ca, cb) + 3 < k_lo) continue;

          std::vector<int> uniq;
          for (int j = 0; j < 4; j++) uniq.push_back(ca + j);
          for (int j = 0; j < 4; j++) uniq.push_back(cb + j);
          std::sort(uniq.begin(), uniq.end());
          uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
          const int m = static_cast<int>(uniq.size());

          auto localIdx = [&](int k) {
            return static_cast<int>(std::lower_bound(uniq.begin(), uniq.end(), k) - uniq.begin());
          };
          CameraEpipolarCost* cc = new CameraEpipolarCost;
          cc->u_a = ua;
          cc->u_b = ub;
          for (int j = 0; j < 4; j++) {
            cc->la[j] = localIdx(ca + j);
            cc->lb[j] = localIdx(cb + j);
          }
          cc->m = m;
          cc->xa = ray.xa;
          cc->xb = ray.xb;
          for (int j = 0; j < 4; j++) cc->q_l2c_conj[j] = q_l2c_conj[j];
          cc->t_l2c = opt.camera.t_l2c;
          cc->weight = opt.camera.weight_px * cam_pixel_scale;

          auto* cost = new ceres::DynamicAutoDiffCostFunction<CameraEpipolarCost>(cc);
          std::vector<double*> blocks(2 * m);
          for (int j = 0; j < m; j++) cost->AddParameterBlock(4);   // quaternion blocks, params[0..m-1]
          for (int j = 0; j < m; j++) cost->AddParameterBlock(3);   // position blocks, params[m..2m-1]
          cost->SetNumResiduals(1);
          for (int j = 0; j < m; j++) {
            addKnot(uniq[j]);
            blocks[j] = traj.q_[uniq[j]].data();
          }
          for (int j = 0; j < m; j++) blocks[m + j] = traj.p_[uniq[j]].data();
          problem.AddResidualBlock(cost, &loss, blocks);
        }
      }
      for (int k = k_lo; k < k_hi; k++) {
        addKnot(k);
        KnotPrior* kp = new KnotPrior;
        for (int i = 0; i < 4; i++) kp->q0[i] = q_init[k][i];
        for (int i = 0; i < 3; i++) kp->p0[i] = p_init[k][i];
        kp->wr = 1.0 / opt.prior_rot;
        kp->wp = 1.0 / opt.prior_pos;
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<KnotPrior, 6, 4, 3>(kp), nullptr, traj.q_[k].data(),
                                  traj.p_[k].data());
      }
      ceres::Solver::Options so;
      // CGNR (iterative, Jacobi-preconditioned), not SPARSE_NORMAL_CHOLESKY/
      // EIGEN_SPARSE: confirmed by direct bisection (shrinking the problem
      // down to ~90 knots / ~10k residuals still reproduced, ruling out a
      // scale/conditioning issue) that EIGEN_SPARSE's symbolic ordering step
      // (Eigen::SimplicialCholeskyBase::ordering(), inside
      // permute_symm_to_fullsymm()) corrupts the heap specifically in THIS
      // executable -- same residual/solve code runs clean under gdb with
      // DENSE_QR instead. ct_refine.cpp uses the identical ceres setup
      // (same Ceres build) without issue in its own, separate binary, which
      // does NOT link Torch/CUDA/livo_cuda -- qscale_refine_livo_recon does
      // (for its LIO/VoxelMap/tracker dependencies), so the leading
      // hypothesis is an allocator conflict from those libraries interfering
      // with Eigen's sparse-matrix allocations, not a logic bug in this
      // file's own residual construction (every input was also verified
      // finite/sane: NaN/Inf-guarded above, weight range checked directly
      // during bisection and found unremarkable). DENSE_QR avoided the
      // crash too but is O(rows*cols^2) in memory -- infeasible at this
      // tool's default ~500-knot/~250k-residual window size. CGNR avoids
      // the crashing code path AND the dense blowup.
      so.linear_solver_type = ceres::CGNR;
      so.preconditioner_type = ceres::JACOBI;
      so.max_num_iterations = 8;
      so.minimizer_progress_to_stdout = false;
      ceres::Solver::Summary sum;
      ceres::Solve(so, &problem, &sum);
      if (opt.verbose) {
        const double t_lo = traj.tStart() + (k_lo - 1.0) * traj.dt();
        const double t_hi = traj.tStart() + (k_hi - 1.0) * traj.dt();
        std::fprintf(stderr,
                     "  window [%.2f,%.2f)s (knots %d-%d): %s, %d iters, cost %.3e -> %.3e, %d residuals\n",
                     t_lo, t_hi, k_lo, k_hi, ceres::TerminationTypeToString(sum.termination_type),
                     static_cast<int>(sum.iterations.size()), sum.initial_cost, sum.final_cost,
                     problem.NumResidualBlocks());
      }
    }

    double rms_after_acc = 0.0;
    for (const Assoc& a : as) {
      M3D R;
      V3D p;
      traj.eval(points[a.i].t, R, p);
      const double r = a.n.dot(R * points[a.i].raw_body_point + p) + a.d;
      rms_after_acc += r * r;
    }
    const double rms_after = as.empty() ? 0.0 : std::sqrt(rms_after_acc / as.size());

    double dr = 0, dp = 0;
    for (size_t k = 0; k < K; k++) {
      dr += std::pow(Eigen::AngleAxisd(quatToR(q_prev[k]).transpose() * quatToR(traj.q_[k])).angle(), 2);
      dp += (V3D(traj.p_[k][0], traj.p_[k][1], traj.p_[k][2]) - V3D(p_prev[k][0], p_prev[k][1], p_prev[k][2]))
                .squaredNorm();
    }
    dr = std::sqrt(dr / K);
    dp = std::sqrt(dp / K);
    rep.n_res.push_back(static_cast<int>(as.size()));
    rep.rms_before.push_back(rms_before);
    rep.rms_after.push_back(rms_after);
    rep.outer_done = outer + 1;
    if (opt.verbose)
      std::fprintf(stderr, "outer %d: rms %.2f -> %.2f mm, knot change %.3f mrad %.2f mm, %.0fs elapsed\n",
                   outer + 1, 1e3 * rms_before, 1e3 * rms_after, 1e3 * dr, 1e3 * dp,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t_begin).count());
    if (cam_enabled && opt.verbose) {
      const CameraResidualStats cs = cameraResidualStats(traj, cam_rays, opt.camera);
      std::fprintf(stderr, "  outer %d camera residual (%d matches): mean %.2fpx median %.2fpx p90 %.2fpx max %.2fpx\n",
                   outer + 1, cs.n, cs.mean, cs.median, cs.p90, cs.max);
    }
    if (dr < 1e-4 && dp < 1e-4) break;
  }
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_begin).count();
  return rep;
}

}  // namespace calib
}  // namespace livo_recon
