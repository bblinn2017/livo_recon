#include "livo_recon/processing/evo_processing.h"
#include "livo_recon/utils/log/param_warn.h"
#include "livo_recon/utils/log/debug_log_dir.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace livo_recon
{

namespace
{

// Debug trace of each pipeline stage's independently-aligned ATE/RTE/ROE/
// ARE, one line per stage per frame -- unlike printer_->print(PrintCategory
// ::EVO, ...) (throttled to outputs/printer/throttle_s, default 5s -- far
// too coarse to see iteration-to-iteration behavior, and only reports the
// VIO stage), this logs every stage on every call. Truncated at the start
// of each process (first call), appended thereafter. Format matches what
// FAST-LIVO2's own converted log uses (see scripts/fastlivo_evo.py) so the
// two can be compared directly.
void debugLogEvo(const std::string& msg)
{
  static bool first_call = true;
  std::ofstream ofs(debugLogPath("evo.txt"), first_call ? std::ios::trunc : std::ios::app);
  first_call = false;
  ofs << msg << "\n";
}

}  // namespace

EvoProc::EvoProc(NodeContext& ctx)
  : state_(ctx.state), profiler_(ctx.profiler), data_queues_(ctx.data_queues)
{}

std::string EvoProc::loadParameters(ros::NodeHandle& pnh)
{
  paramWarn<bool>(pnh, "evo/enable",            opts_.enable,       false);
  paramWarn<std::string>(pnh, "evo/gt_source",  opts_.gt_source,    "topic");
  paramWarn<std::string>(pnh, "evo/gt_topic",   opts_.gt_topic,     "/leica/pose/relative");
  paramWarn<std::string>(pnh, "evo/gt_file",    opts_.gt_file,      "");
  paramWarn<int>(pnh, "evo/queue_size",         opts_.queue_size,   2000);
  paramWarn<double>(pnh, "evo/max_time_diff",   opts_.max_time_diff, 0.05);
  paramWarn<bool>(pnh, "evo/rpe/compute_rte",   opts_.compute_rte,  false);
  paramWarn<bool>(pnh, "evo/rpe/compute_roe",   opts_.compute_roe,  false);
  paramWarn<double>(pnh, "evo/rpe/delta_s",     opts_.rpe_delta_s,  1.0);
  paramWarn<bool>(pnh, "evo/compute_nearest_live", opts_.compute_nearest_live, false);

  {
    std::vector<double> lever_arm{0.0, 0.0, 0.0};
    paramWarn<std::vector<double>>(pnh, "evo/gt_lever_arm", lever_arm, lever_arm);
    if (lever_arm.size() == 3)
      opts_.gt_lever_arm = V3D(lever_arm[0], lever_arm[1], lever_arm[2]);
  }

  const bool file_mode = (opts_.gt_source == "file");

  std::ostringstream oss;
  oss << "[params/evo]"
      << "\n  enable:          " << (opts_.enable ? "true" : "false")
      << "\n  gt_source:       " << opts_.gt_source
      << "\n  gt_topic:        " << opts_.gt_topic
      << "\n  gt_file:         " << opts_.gt_file
      << "\n  queue_size:      " << opts_.queue_size
      << "\n  max_time_diff:   " << opts_.max_time_diff
      << "\n  gt_lever_arm:    [" << opts_.gt_lever_arm.transpose() << "]"
      << "\n  rpe/compute_rte: " << (opts_.compute_rte ? "true" : "false")
      << "\n  rpe/compute_roe: " << (opts_.compute_roe ? "true" : "false")
      << "\n  rpe/delta_s:     " << opts_.rpe_delta_s
      << "\n  compute_nearest_live: " << (opts_.compute_nearest_live ? "true" : "false")
      << (file_mode ? " (ignored -- file mode always computes nearest)" : "");

  // Sparse checkpoints (file_mode) are 15-40s apart in practice (e.g.
  // HILTI's public groundtruth_2022/2023) -- far too widely spaced for
  // RTE/ROE at any reasonable delta_s to mean anything, and HILTI's own
  // evaluation.py is APE/ATE-only for the same reason. Force both off
  // rather than silently returning -1 forever, so a stale yaml setting
  // doesn't look like a live-but-unlucky metric.
  if (file_mode && (opts_.compute_rte || opts_.compute_roe))
  {
    oss << "\n  WARNING: rpe/compute_rte and rpe/compute_roe are forced off "
        << "for gt_source=file (sparse checkpoints too widely spaced for a "
        << "meaningful relative-pose comparison)";
    opts_.compute_rte = false;
    opts_.compute_roe = false;
  }

  if (opts_.enable)
  {
    if (file_mode)
    {
      if (loadGtFile(opts_.gt_file))
        oss << "\n  loaded " << gt_buffer_.size() << " ground-truth checkpoints from " << opts_.gt_file;
      else
        oss << "\n  ERROR: failed to load gt_file " << opts_.gt_file << " -- evo will report nothing";
    }
    else
    {
      // Raw subscription lives in CbkProc now (see drainGtQueue()'s doc
      // comment) -- nothing to do here for topic mode besides the params
      // dump above; CbkProc::loadParameters() reads these same evo/*
      // params independently to decide whether to subscribe.
      oss << "\n  gt ingestion via CbkProc -> DataQueues::popGt()";
    }
  }

  return oss.str();
}

bool EvoProc::loadGtFile(const std::string& path)
{
  std::ifstream ifs(path);
  if (!ifs.is_open())
  {
    ROS_ERROR("[evo] gt_source=file but gt_file could not be opened: %s", path.c_str());
    return false;
  }

  std::string line;
  while (std::getline(ifs, line))
  {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    double t, tx, ty, tz, qx, qy, qz, qw;
    if (!(iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;

    GtSample s;
    s.t = t;  // raw absolute epoch -- shifted once by shiftGtFileTimestampsOnce()
    s.pos = V3D(tx, ty, tz);
    s.rot = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
    gt_buffer_.push_back(s);

    if (!gt_orientation_seen_ && !s.rot.isApprox(M3D::Identity(), 1e-6))
      gt_orientation_seen_ = true;
  }

  std::sort(gt_buffer_.begin(), gt_buffer_.end(),
            [](const GtSample& a, const GtSample& b) { return a.t < b.t; });

  return !gt_buffer_.empty();
}

void EvoProc::shiftGtFileTimestampsOnce()
{
  if (gt_file_ts_shifted_ || opts_.gt_source != "file") return;
  for (auto& s : gt_buffer_) s.t -= data_queues_->start_time;
  gt_file_ts_shifted_ = true;
}

void EvoProc::drainGtQueue()
{
  // Raw ingestion (ROS subscriber in live mode, direct bag dispatch in
  // runOffline()) lives in CbkProc::gtCallback() alongside image/lidar/
  // imu -- see DataQueues::pushGt()/popGt()'s doc comments. This drains
  // whatever's arrived since the last processEvo() tick and folds it into
  // gt_buffer_, replacing the old per-message gtCallback() body that used
  // to run directly off the subscriber; the eviction/cursor-adjustment/
  // orientation-detection logic below is unchanged from that version.
  for (const GtPoseSample& s : data_queues_->popGt())
  {
    gt_buffer_.push_back({s.t, s.pos, s.rot});
    while (static_cast<int>(gt_buffer_.size()) > opts_.queue_size)
    {
      gt_buffer_.pop_front();
      // next_gt_idx_ indexes into gt_buffer_ (see processEvoTopicMode()) --
      // popping the front shifts every remaining element back by one, so
      // the cursor must shift with it or it'll silently point one-past
      // where it should. queue_size (default 2000) at a realistic GT rate
      // is normally far larger than the lag between arrival and
      // processEvoTopicMode() consuming the cursor, so this path is rarely
      // exercised in practice, but must stay correct regardless.
      if (next_gt_idx_ > 0) --next_gt_idx_;
    }

    if (!gt_orientation_seen_ && !s.rot.isApprox(M3D::Identity(), 1e-6))
      gt_orientation_seen_ = true;
  }
}

double EvoProc::computeAte(const std::vector<MatchedPose>& matched, M3D& R_align_out, V3D& t_align_out) const
{
  R_align_out = M3D::Identity();
  t_align_out = V3D::Zero();

  const int n = static_cast<int>(matched.size());
  if (n < 3) return -1.0;  // not enough points to align meaningfully

  V3D centroid_est = V3D::Zero(), centroid_gt = V3D::Zero();
  for (const auto& m : matched) { centroid_est += m.est_pos; centroid_gt += m.gt_pos; }
  centroid_est /= n;
  centroid_gt  /= n;

  // Kabsch/Horn alignment: H = sum (est_i - centroid_est)*(gt_i - centroid_gt)^T,
  // SVD H = U*S*V^T, R = V*U^T (with a reflection fix if det(R) < 0) is the
  // rotation minimizing sum ||R*est_i + t - gt_i||^2 -- i.e. mapping the
  // estimated trajectory onto the ground truth, no scale (this is a
  // metric LIO/VIO system, unlike monocular-only setups where a Sim3
  // alignment with free scale would be needed instead).
  M3D H = M3D::Zero();
  for (const auto& m : matched)
    H += (m.est_pos - centroid_est) * (m.gt_pos - centroid_gt).transpose();

  Eigen::JacobiSVD<M3D> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  M3D V = svd.matrixV();
  M3D R = V * svd.matrixU().transpose();
  if (R.determinant() < 0.0)
  {
    V.col(2) *= -1.0;
    R = V * svd.matrixU().transpose();
  }

  R_align_out = R;
  t_align_out = centroid_gt - R * centroid_est;

  double sq_err_sum = 0.0;
  for (const auto& m : matched)
    sq_err_sum += ((R * m.est_pos + t_align_out) - m.gt_pos).squaredNorm();

  return std::sqrt(sq_err_sum / n);
}

double EvoProc::computeAre(const std::vector<MatchedPose>& matched, const M3D& R_align) const
{
  const int n = static_cast<int>(matched.size());
  if (n < 3) return -1.0;

  double sq_err_sum = 0.0;
  for (const auto& m : matched)
  {
    const M3D err_rot = (R_align * m.est_rot).transpose() * m.gt_rot;
    const double cos_angle = std::clamp((err_rot.trace() - 1.0) / 2.0, -1.0, 1.0);
    const double angle = std::acos(cos_angle);
    sq_err_sum += angle * angle;
  }

  return std::sqrt(sq_err_sum / n) * (180.0 / M_PI);
}

double EvoProc::computeRte(const std::vector<MatchedPose>& matched, int delta_frames) const
{
  const int n = static_cast<int>(matched.size());
  if (n < 2 || delta_frames >= n) return -1.0;

  double sq_err_sum = 0.0;
  int count = 0;
  for (int i = 0; i + delta_frames < n; ++i)
  {
    const int j = i + delta_frames;

    // evo's PoseRelation.point_distance: the difference between the two
    // trajectories' own relative-displacement MAGNITUDES, not a vector
    // difference -- and no alignment needed at all, since a common rigid
    // transform applied to both matched[i]/matched[j] cancels out of this
    // magnitude regardless (rotation preserves norm, translation cancels
    // in the difference), unlike ATE/ARE. Raw est_pos is used directly.
    const double est_dist = (matched[j].est_pos - matched[i].est_pos).norm();
    const double gt_dist  = (matched[j].gt_pos  - matched[i].gt_pos).norm();

    const double err = std::abs(est_dist - gt_dist);
    sq_err_sum += err * err;
    ++count;
  }

  return count > 0 ? std::sqrt(sq_err_sum / count) : -1.0;
}

double EvoProc::computeRoe(const std::vector<MatchedPose>& matched, int delta_frames) const
{
  const int n = static_cast<int>(matched.size());
  if (n < 2 || delta_frames >= n) return -1.0;

  double sq_err_sum = 0.0;
  int count = 0;
  for (int i = 0; i + delta_frames < n; ++i)
  {
    const int j = i + delta_frames;

    // A common rotation applied to both endpoints of a relative-rotation
    // comparison cancels out, so the ATE alignment's rotation isn't needed
    // here.
    const M3D est_rel = matched[i].est_rot.transpose() * matched[j].est_rot;
    const M3D gt_rel  = matched[i].gt_rot.transpose()  * matched[j].gt_rot;
    const M3D err_rot = est_rel.transpose() * gt_rel;

    const double cos_angle = std::clamp((err_rot.trace() - 1.0) / 2.0, -1.0, 1.0);
    const double angle = std::acos(cos_angle);

    sq_err_sum += angle * angle;
    ++count;
  }

  return count > 0 ? std::sqrt(sq_err_sum / count) * (180.0 / M_PI) : -1.0;
}

M3D EvoProc::slerpRot(const M3D& before, const M3D& after, double alpha)
{
  // Mirrors fastlivo_evo.py's quat_slerp() -- spherical linear
  // interpolation between two orientations. Eigen::Quaterniond::slerp()
  // already handles the double-cover case (flipping one side if the dot
  // product is negative) and the near-identical near-linear case
  // internally, same as fastlivo_evo.py's own hand-rolled implementation.
  const Eigen::Quaterniond qa(before);
  const Eigen::Quaterniond qb(after);
  return qa.slerp(alpha, qb).toRotationMatrix();
}

EvoProc::StageMetrics EvoProc::logStage(const char* stage_name, const char* mode_name, double t_abs,
                                        std::vector<MatchedPose>& matched,
                                        const V3D& est_pos, const M3D& est_rot,
                                        const V3D& gt_pos, const M3D& gt_rot) const
{
  // Correct for a ground-truth-sensor lever arm (e.g. NTU VIRAL's Leica
  // prism, offset from the IMU/body frame -- see EvoProcOptions::
  // gt_lever_arm's doc comment) before matching against gt_pos, which
  // tracks that sensor's own location, not the body frame's. Zero by
  // default, so this is a no-op unless a dataset config sets it.
  const V3D est_pos_corrected = est_pos + est_rot * opts_.gt_lever_arm;

  matched.push_back(MatchedPose{t_abs, est_pos_corrected, est_rot, gt_pos, gt_rot});

  StageMetrics m;
  M3D R_align;
  V3D t_align;
  m.ate = computeAte(matched, R_align, t_align);

  std::ostringstream dbg;
  dbg << "[evo/debug]  stage=" << stage_name << "  mode=" << mode_name
      << std::fixed << std::setprecision(4) << "  t_abs=" << t_abs
      << "  n=" << matched.size();

  if (m.ate < 0.0)
  {
    dbg << "  accumulating (need >= 3 to align)";
    debugLogEvo(dbg.str());
    return m;
  }

  // evo converts a desired RPE_DELTA_S into an index delta via the matched
  // trajectory's own median timestamp spacing (evo's id_pairs_from_delta
  // has no working Unit.seconds handler). This function runs on every
  // single commit and is already O(n) per call (computeAte/computeAre/
  // computeRte/computeRoe each walk `matched`), so a real O(n log n)
  // sort-based median every call isn't worth it here -- the cheap MEAN
  // consecutive-gap spacing is used instead, which is a deliberate
  // approximation: for realistically ~evenly-sampled GT/estimate data the
  // mean and median of consecutive gaps are nearly identical.
  const int n = static_cast<int>(matched.size());
  const double mean_dt = (n > 1) ? (matched.back().t - matched.front().t) / (n - 1) : 0.0;
  const int delta_frames = (mean_dt > 0.0)
      ? std::max(1, static_cast<int>(std::lround(opts_.rpe_delta_s / mean_dt)))
      : 1;

  m.rte = opts_.compute_rte ? computeRte(matched, delta_frames) : -1.0;
  // ROE only needs the opt-in flag: it's a RELATIVE-rotation comparison,
  // and reporting it plainly (rather than gating it on gtOrientationMeaningful()
  // the way ARE is) is a deliberate choice for datasets whose ground truth
  // has no real orientation (e.g. NTU VIRAL's Leica prism topic, HILTI's
  // sparse checkpoints) -- against such a source it's a diagnostic
  // curiosity (how much the estimate itself rotated over rpe_delta_s), not
  // a true error against ground truth, but still worth surfacing rather
  // than silently reporting -1 forever. Matches config/ntu_viral/evo.yaml's
  // documented intent, which this decoupling actually satisfies -- ROE was
  // previously bundled under the same gate as ARE below, so it never
  // reported anything despite compute_roe: true.
  m.roe = opts_.compute_roe ? computeRoe(matched, delta_frames) : -1.0;
  // ARE (an ABSOLUTE orientation comparison) has no such reading, even as a
  // curiosity: against an always-identity "ground truth" it's just how far
  // the aligned estimate sits from the identity matrix, constant regardless
  // of any real motion. Stays gated on gtOrientationMeaningful() so it only
  // ever reports something for a ground-truth source that actually measures
  // real orientation.
  m.are = (opts_.compute_roe && gtOrientationMeaningful()) ? computeAre(matched, R_align) : -1.0;

  dbg << "  ATE=" << m.ate << "m";
  if (m.rte >= 0.0) dbg << "  RTE=" << m.rte << "m";
  // Say what the number is measured AGAINST, not just its name. Against a
  // position-only ground truth (NTU VIRAL's laser tracker, HILTI's sparse
  // checkpoints -- both carry an identity quaternion everywhere) ROE is the
  // angular magnitude of the estimate's own rotation over rpe_delta_s, not
  // an error. Every ROE ever quoted in this project's register was read as
  // an error and computed against no reference; the metric name alone is
  // what allowed that, so the reference travels with the value from here on.
  if (m.roe >= 0.0)
    dbg << "  ROE=" << m.roe << "deg"
        << (gtOrientationMeaningful() ? "(vs GT orientation)"
                                      : "(NO GT ORIENTATION -- self-rotation "
                                        "magnitude, not an error)");
  if (m.are >= 0.0) dbg << "  ARE=" << m.are << "deg";

  // This frame's own aligned position error VECTOR (not just its scalar
  // contribution to the running ATE) -- R_align/t_align are this call's
  // whole-trajectory-so-far best fit, so (R_align*est_pos+t_align - gt_pos)
  // is the current pose's residual against ground truth in the GT's own
  // frame. Lets a reader check whether the drift after a disturbance has a
  // consistent direction (systematic bias) or is directionless (noise that
  // just hasn't been walked back down yet).
  const V3D aligned_err = (R_align * est_pos_corrected + t_align) - gt_pos;
  dbg << "  aligned_err=[" << aligned_err.transpose() << "]";

  debugLogEvo(dbg.str());

  return m;
}

std::string EvoProc::formatModeStats(const char* mode_name, const StageMetrics& vio, size_t n,
                                     double rpe_delta_s, bool gt_orientation_meaningful,
                                     std::string& last_stats_slot)
{
  if (vio.ate < 0.0)
  {
    last_stats_slot = std::string("mode=") + mode_name + "  accumulating (need >= 3 matched poses)\n";
    return {};
  }

  std::ostringstream oss;
  oss << "[evo/" << mode_name << "]  n=" << n
      << std::fixed << std::setprecision(4)
      << "  ATE=" << vio.ate << "m";
  if (vio.rte >= 0.0) oss << "  RTE=" << vio.rte << "m@" << rpe_delta_s << "s";
  if (vio.roe >= 0.0) oss << "  ROE=" << vio.roe << "deg@" << rpe_delta_s << "s";
  if (vio.are >= 0.0) oss << "  ARE=" << vio.are << "deg";
  // CQ-13: carry the rotation-reference source into this line too (the "dbg"
  // per-frame stream already does, above) -- APPENDED at the very end, never
  // inserted between existing fields, so every regex/substring consumer of
  // this line (dispatch_queue.sh's ATE=\K[0-9.]+, analyze_hth_collapse.py,
  // this file's own ATE_LINE_RE retry logic) keeps matching unchanged.
  if (vio.roe >= 0.0)
    oss << "  " << (gt_orientation_meaningful ? "(ROE vs GT orientation)"
                                               : "(ROE: NO GT ORIENTATION -- "
                                                 "self-rotation magnitude, not an error)");

  // Wide enough for the largest values actually seen in practice (a
  // diverged run's ATE can reach 6 figures) without a column overflowing
  // its width and shifting everything after it out of alignment with the
  // header row -- setw() only ever pads up to a MINIMUM width, it doesn't
  // truncate a longer value back down to size.
  constexpr int kColW = 14;
  std::ostringstream table;
  table << "mode=" << mode_name << "\n";
  table << std::left
        << std::setw(kColW) << "n"
        << std::setw(kColW) << "ATE(m)"
        << std::setw(kColW) << (vio.rte >= 0.0 ? "RTE(m)" : "")
        << std::setw(kColW) << (vio.roe >= 0.0 ? "ROE(deg)" : "")
        << std::setw(kColW) << (vio.are >= 0.0 ? "ARE(deg)" : "")
        << "\n";
  table << std::fixed << std::setprecision(3)
        << std::setw(kColW) << n
        << std::setw(kColW) << vio.ate;
  if (vio.rte >= 0.0) table << std::setw(kColW) << vio.rte; else table << std::setw(kColW) << "";
  if (vio.roe >= 0.0) table << std::setw(kColW) << vio.roe; else table << std::setw(kColW) << "";
  if (vio.are >= 0.0) table << std::setw(kColW) << vio.are; else table << std::setw(kColW) << "";
  table << "\n";
  last_stats_slot = table.str();

  return oss.str();
}

std::string EvoProc::commitMatch(MeasureGroup* mg, double t_abs,
                                 bool nearest_valid, const EstimateSnapshot& nearest,
                                 bool interp_valid,  const EstimateSnapshot& interp,
                                 const V3D& gt_pos, const M3D& gt_rot)
{
  // Runs one association mode's estimate snapshot through all 3 stages
  // (see logStage()'s doc comment for why each stage is aligned
  // independently) and formats/caches its own console+overlay block --
  // "nearest" and "interp" are two entirely separate trajectories/
  // alignments here, exactly mirroring fastlivo_evo.py's two independent
  // --mode invocations (see processEvo()'s doc comment).
  auto run_mode = [&](const char* mode_name, const EstimateSnapshot& est, std::string& last_stats_slot) -> std::string
  {
    const bool is_interp = (std::string(mode_name) == "interp");
    std::vector<MatchedPose>& imu_vec = is_interp ? matched_imu_.interp : matched_imu_.nearest;
    std::vector<MatchedPose>& lio_vec = is_interp ? matched_lio_.interp : matched_lio_.nearest;
    std::vector<MatchedPose>& vio_vec = is_interp ? matched_vio_.interp : matched_vio_.nearest;

    logStage("imu", mode_name, t_abs, imu_vec, est.imu_pos, est.imu_rot, gt_pos, gt_rot);
    logStage("lio", mode_name, t_abs, lio_vec, est.lio_pos, est.lio_rot, gt_pos, gt_rot);
    const StageMetrics vio = logStage("vio", mode_name, t_abs, vio_vec, est.vio_pos, est.vio_rot, gt_pos, gt_rot);

    // Console/overlay reflect the VIO stage specifically -- the final
    // refinement step, per explicit instruction ("use the vio ate for
    // right now"). imu/lio's own independently-aligned numbers are in
    // /tmp/evo.txt only.
    return formatModeStats(mode_name, vio, vio_vec.size(), opts_.rpe_delta_s,
                           gtOrientationMeaningful(), last_stats_slot);
  };

  std::ostringstream combined;
  if (nearest_valid)
  {
    const std::string line = run_mode("nearest", nearest, last_stats_nearest_);
    if (!line.empty()) combined << line << "\n";
  }
  if (interp_valid)
  {
    const std::string line = run_mode("interp", interp, last_stats_interp_);
    if (!line.empty()) combined << line << "\n";
  }

  // Consumed by PubProc::publishEvoStats (see mg.evo_stats) to show both
  // modes' numbers as an rviz OverlayText -- two stacked tables (one per
  // mode), metrics laid out in COLUMNS within each table (not one metric
  // per row), since the overlay box has much more usable width than
  // height. Carries the latest known table for EACH mode forward
  // independently (see the top of processEvo()), so one mode going stale
  // (e.g. interp skipped this tick because the bracket gap was too wide)
  // doesn't blank out the other mode's still-fresh numbers.
  if (mg != nullptr) mg->evo_stats = last_stats_nearest_ + last_stats_interp_;

  std::string result = combined.str();
  if (!result.empty() && result.back() == '\n') result.pop_back();
  if (result.empty())
    result = "[evo]  accumulating (need >= 3 matched poses to align)";
  return result;
}

std::string EvoProc::processEvoFileMode(MeasureGroup& mg)
{
  std::string result;

  // A single call ordinarily brackets (at most) one checkpoint, but if
  // incoming frames have a big gap spanning several checkpoints (rare),
  // keep matching/committing against bracket_cache_ (the same "before"
  // candidate each time -- which side wins nearest-raw-timestamp differs
  // per checkpoint) and advancing until either the buffer is exhausted or
  // this frame no longer sits at/after the current checkpoint -- see
  // BracketCache's doc comment.
  while (next_gt_idx_ < gt_buffer_.size())
  {
    const GtSample& checkpoint = gt_buffer_[next_gt_idx_];

    if (mg.image.t < checkpoint.t)
    {
      // Still strictly before the checkpoint -- this frame isn't an
      // "after" bracket for it yet, nothing to commit.
      break;
    }

    // mg.image.t >= checkpoint.t: this frame is the first one at/after the
    // checkpoint, i.e. its "after" candidate. Without a cached "before"
    // candidate (the very first estimate frame this run has ever produced
    // already lands at/after this checkpoint's timestamp) there is no pair
    // to pick the nearer of -- skip the checkpoint rather than
    // extrapolating backward (see BracketCache's doc comment).
    if (!bracket_cache_.valid)
    {
      ++next_gt_idx_;
      continue;
    }

    // Nearest-RAW-timestamp matching (evo.core.sync.associate_trajectories'
    // own behavior -- see processEvo()'s doc comment): pick whichever of
    // the "before" (bracket_cache_) or "after" (this frame, mg) side is
    // closer in time to the checkpoint, use that side's RAW pose fields
    // directly (no lerp/slerp), and only commit if that gap is within
    // opts_.max_time_diff -- otherwise this checkpoint is dropped,
    // unmatched, exactly like the "!bracket_cache_.valid -> skip" case
    // above.
    const double gap_before = checkpoint.t - bracket_cache_.t;
    const double gap_after  = mg.image.t - checkpoint.t;
    const bool use_before = gap_before <= gap_after;
    const double gap = use_before ? gap_before : gap_after;

    const bool nearest_valid = (gap <= opts_.max_time_diff);
    if (nearest_valid)
    {
      EstimateSnapshot nearest;
      nearest.imu_pos = use_before ? bracket_cache_.imu_pos : mg.pos_after_imu;
      nearest.imu_rot = use_before ? bracket_cache_.imu_rot : mg.rot_after_imu;
      nearest.lio_pos = use_before ? bracket_cache_.lio_pos : mg.pos_after_lio;
      nearest.lio_rot = use_before ? bracket_cache_.lio_rot : mg.rot_after_lio;
      nearest.vio_pos = use_before ? bracket_cache_.vio_pos : mg.pos_after_vio;
      nearest.vio_rot = use_before ? bracket_cache_.vio_rot : mg.rot_after_vio;

      // checkpoint.t is in the same start_time-relative basis as mg.image.t
      // (see shiftGtFileTimestampsOnce()) -- shift to absolute for
      // commitMatch()/the debug log, same convention processEvoTopicMode()
      // uses.
      const double t_abs = checkpoint.t + data_queues_->start_time;

      // interp mode is never attempted for gt_source == "file" (see
      // gt_source's doc comment) -- pass an unused default-constructed
      // snapshot with interp_valid == false.
      result = commitMatch(&mg, t_abs, nearest_valid, nearest,
                           /*interp_valid=*/false, EstimateSnapshot{},
                           checkpoint.pos, checkpoint.rot);
    }

    ++next_gt_idx_;
  }

  // This frame becomes the "before" bracket for whichever checkpoint comes
  // next (including one this same call already passed, if a gap skipped
  // past it above).
  bracket_cache_.t = mg.image.t;
  bracket_cache_.imu_pos = mg.pos_after_imu; bracket_cache_.imu_rot = mg.rot_after_imu;
  bracket_cache_.lio_pos = mg.pos_after_lio; bracket_cache_.lio_rot = mg.rot_after_lio;
  bracket_cache_.vio_pos = mg.pos_after_vio; bracket_cache_.vio_rot = mg.rot_after_vio;
  bracket_cache_.valid = true;

  return result;
}

std::string EvoProc::processEvoTopicMode(MeasureGroup& mg)
{
  std::string result;

  // Mirror image of processEvoFileMode(): there, GT is sparse and
  // next_gt_idx_ walks a handful of checkpoints, each usually needing
  // several estimate-frame calls to reach its "after" candidate. Here,
  // gt_buffer_ is dense and growing live (via gtCallback(), asynchronous
  // to this function's own calls) -- next_gt_idx_ instead walks however
  // many buffered GT samples now fall at/before this frame's timestamp,
  // which can be zero (estimate updating faster than GT publishes -- the
  // common case) or more than one (GT publishes faster than the estimate
  // updates -- each such sample is matched separately below, against the
  // SAME "before"/"after" candidate pair, so none are silently skipped).
  while (next_gt_idx_ < gt_buffer_.size())
  {
    const GtSample& sample = gt_buffer_[next_gt_idx_];

    if (mg.image.t < sample.t)
    {
      // This GT sample is still strictly ahead of the current estimate
      // frame -- no "after" bracket for it yet, nothing to commit this
      // call.
      break;
    }

    // sample.t <= mg.image.t: this frame is (at least) the "after"
    // candidate for this GT sample. Without a cached "before" candidate
    // (this GT sample's timestamp precedes the very first estimate frame
    // this run has ever produced) there is no pair to pick the nearer of --
    // skip rather than extrapolate backward (same edge case BracketCache
    // handles in file mode).
    if (!bracket_cache_.valid)
    {
      ++next_gt_idx_;
      continue;
    }

    // Two independent association modes, both gated on opts_.max_time_diff
    // but with different thresholds/constructions (see processEvo()'s doc
    // comment) -- mirrors fastlivo_evo.py's lookup_nearest()/lookup_est()
    // exactly:
    //
    // "nearest": pick whichever of the "before" (bracket_cache_) or
    // "after" (this frame, mg) side is closer in time to this GT sample,
    // use that side's RAW pose fields directly (no lerp/slerp), and only
    // commit if that gap is within opts_.max_time_diff.
    const double gap_before = sample.t - bracket_cache_.t;
    const double gap_after  = mg.image.t - sample.t;
    const bool use_before = gap_before <= gap_after;
    const double gap = use_before ? gap_before : gap_after;
    // See EvoProcOptions::compute_nearest_live's doc comment -- "nearest"
    // is fragile live (raw, unblended snapshots can inject a single bad
    // outlier that biases computeAte()'s whole-history batch refit for the
    // rest of the run) and is meant to be scored offline instead
    // (fastlivo_evo.py --mode nearest against the exported odometry.txt),
    // off by default. "interp" is unaffected.
    const bool nearest_valid = opts_.compute_nearest_live && (gap <= opts_.max_time_diff);

    // "interp": lerp(position)/slerp(rotation) the estimate to this GT
    // sample's own exact timestamp, valid only if the two bracketing
    // frames aren't more than 2*opts_.max_time_diff apart -- otherwise the
    // interpolation would span too much real motion to be meaningful
    // (matches fastlivo_evo.py's lookup_est() "b[0]-a[0] > 2*max_time_diff"
    // rejection exactly).
    const double bracket_span = mg.image.t - bracket_cache_.t;
    const bool interp_valid = (bracket_span <= 2.0 * opts_.max_time_diff);
    const double alpha = interp_valid ? (sample.t - bracket_cache_.t) / std::max(bracket_span, 1e-9) : 0.0;

    if (nearest_valid || interp_valid)
    {
      EstimateSnapshot nearest;
      if (nearest_valid)
      {
        nearest.imu_pos = use_before ? bracket_cache_.imu_pos : mg.pos_after_imu;
        nearest.imu_rot = use_before ? bracket_cache_.imu_rot : mg.rot_after_imu;
        nearest.lio_pos = use_before ? bracket_cache_.lio_pos : mg.pos_after_lio;
        nearest.lio_rot = use_before ? bracket_cache_.lio_rot : mg.rot_after_lio;
        nearest.vio_pos = use_before ? bracket_cache_.vio_pos : mg.pos_after_vio;
        nearest.vio_rot = use_before ? bracket_cache_.vio_rot : mg.rot_after_vio;
      }

      EstimateSnapshot interp;
      if (interp_valid)
      {
        interp.imu_pos = bracket_cache_.imu_pos + alpha * (mg.pos_after_imu - bracket_cache_.imu_pos);
        interp.lio_pos = bracket_cache_.lio_pos + alpha * (mg.pos_after_lio - bracket_cache_.lio_pos);
        interp.vio_pos = bracket_cache_.vio_pos + alpha * (mg.pos_after_vio - bracket_cache_.vio_pos);
        interp.imu_rot = slerpRot(bracket_cache_.imu_rot, mg.rot_after_imu, alpha);
        interp.lio_rot = slerpRot(bracket_cache_.lio_rot, mg.rot_after_lio, alpha);
        interp.vio_rot = slerpRot(bracket_cache_.vio_rot, mg.rot_after_vio, alpha);
      }

      // sample.t is already start_time-relative (see gtCallback()) -- shift
      // to absolute for commitMatch()/the debug log, same convention
      // processEvoFileMode() uses. Ground truth is always used exactly as
      // buffered, at its own real timestamp -- only the estimate side ever
      // has a "before"/"after"/interpolated choice to make.
      const double t_abs = sample.t + data_queues_->start_time;

      result = commitMatch(&mg, t_abs, nearest_valid, nearest, interp_valid, interp,
                           sample.pos, sample.rot);
    }

    ++next_gt_idx_;
  }

  // This frame becomes the "before" bracket for whichever GT sample comes
  // next (including one this same call already committed against, if
  // several samples landed in this same inter-frame gap above).
  bracket_cache_.t = mg.image.t;
  bracket_cache_.imu_pos = mg.pos_after_imu; bracket_cache_.imu_rot = mg.rot_after_imu;
  bracket_cache_.lio_pos = mg.pos_after_lio; bracket_cache_.lio_rot = mg.rot_after_lio;
  bracket_cache_.vio_pos = mg.pos_after_vio; bracket_cache_.vio_rot = mg.rot_after_vio;
  bracket_cache_.valid = true;

  return result;
}

std::string EvoProc::finalizePendingFileMatch()
{
  // No-op: with nearest-raw-timestamp matching, a checkpoint commits the
  // instant its "after" candidate is seen (processEvoFileMode(), above), so
  // there's no "pending, window still open" state left to flush at
  // run-end. A checkpoint whose timestamp falls after the very last
  // estimate frame ever produced never gets an "after" candidate and is
  // simply never committed -- see BracketCache's doc comment; there is
  // deliberately no one-sided-extrapolation fallback here. Kept only so the
  // LivoReconNode::run() call site (mirroring its exportColmap()/
  // exportReport() calls) doesn't need to change.
  return {};
}

std::string EvoProc::processEvo(MeasureGroup& mg)
{
  if (!opts_.enable) return {};

  TimedScope ts(profiler_, "evo");

  // Carried forward onto every frame's mg up front, so a frame that falls
  // through one of the early-returns below (no fresh ground-truth match
  // this exact tick, still accumulating, etc.) still gets the most
  // recently computed tables rather than empty ones -- see
  // last_stats_nearest_/last_stats_interp_.
  mg.evo_stats = last_stats_nearest_ + last_stats_interp_;

  shiftGtFileTimestampsOnce();
  if (opts_.gt_source == "topic") drainGtQueue();

  // Both branches share the same bracket-and-commit shape (see
  // processEvoFileMode()/processEvoTopicMode() and BracketCache): match
  // each GT sample against the bracketing RAW estimate frames (before/
  // after), in one or both association modes (nearest and/or interp --
  // see this function's own doc comment), never fabricating a pose on
  // either side beyond what interpolation between two REAL frames
  // provides. They differ only in which side of the GT/estimate pair is
  // sparse vs. dense and growing, and file mode never attempts interp
  // (see gt_source's doc comment).
  if (opts_.gt_source == "file")
    return processEvoFileMode(mg);

  return processEvoTopicMode(mg);
}

}  // namespace livo_recon
