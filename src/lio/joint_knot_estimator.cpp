#include "livo_recon/lio/joint_knot_estimator.h"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace livo_recon
{
namespace
{

JointKnot samplePropagation(const std::vector<Pose6D>& poses, double t)
{
  if (poses.empty()) return {};
  if (t <= poses.front().t)
    return {t, poses.front().rot, poses.front().pos, poses.front().vel};
  if (t >= poses.back().t)
    return {t, poses.back().rot, poses.back().pos, poses.back().vel};

  auto hi = std::lower_bound(poses.begin(), poses.end(), t,
      [](const Pose6D& p, double value) { return p.t < value; });
  const Pose6D& b = *hi;
  const Pose6D& a = *(hi - 1);
  const double u = (t - a.t) / std::max(b.t - a.t, 1e-12);
  JointKnot out;
  out.t = t;
  out.R = a.rot * Exp(u * Log(M3D(a.rot.transpose() * b.rot)));
  out.p = (1.0 - u) * a.pos + u * b.pos;
  out.v = (1.0 - u) * a.vel + u * b.vel;
  return out;
}

Eigen::MatrixXd symmetricInverse(const Eigen::MatrixXd& matrix)
{
  const Eigen::MatrixXd S = 0.5 * (matrix + matrix.transpose());
  Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success || !ldlt.isPositive())
    throw std::runtime_error("joint-knot information matrix is not positive definite");
  return ldlt.solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
}

}  // namespace

bool JointKnotTrajectory::initialize(const std::vector<Pose6D>& poses,
                                     double scan_tail, int knot_count,
                                     const StateGroup& propagated_state)
{
  if (poses.empty() || knot_count < 2 || !(scan_tail > poses.front().t))
    return false;
  if (static_cast<int>(poses.size()) < knot_count - 1) return false;
  knots_.resize(knot_count);
  // Interior knots use actual propagation timestamps.  The approximately
  // uniform index selection is deterministic and keeps every knot boundary
  // aligned with exact production F/Q interval boundaries.
  for (int k = 0; k < knot_count - 1; ++k) {
    const double target = static_cast<double>(k) * poses.size() /
                          static_cast<double>(knot_count - 1);
    const int index = std::min(static_cast<int>(poses.size()) - 1,
                               static_cast<int>(std::llround(target)));
    knots_[k] = samplePropagation(poses, poses[index].t);
  }
  // The propagation record can stop at the beginning of its final interval.
  // The filter state is the authoritative state at the exact scan tail.
  knots_.back().t = scan_tail;
  knots_.back().R = propagated_state.rot();
  knots_.back().p = propagated_state.pos();
  knots_.back().v = propagated_state.vel();
  bg_.assign(knot_count, propagated_state.biasGyr());
  ba_.assign(knot_count, propagated_state.biasAcc());
  g_ = propagated_state.gravity();
  return true;
}

int JointKnotTrajectory::interval(double t, double& u) const
{
  if (knots_.size() < 2) { u = 0.0; return 0; }
  if (t <= knots_.front().t) { u = 0.0; return 0; }
  if (t >= knots_.back().t) { u = 1.0; return knotCount() - 2; }
  auto hi = std::upper_bound(knots_.begin(), knots_.end(), t,
      [](double value, const JointKnot& k) { return value < k.t; });
  const int i = std::max(0, static_cast<int>(hi - knots_.begin()) - 1);
  u = (t - knots_[i].t) / std::max(knots_[i + 1].t - knots_[i].t, 1e-12);
  return std::min(i, knotCount() - 2);
}

JointKnotEvaluation JointKnotTrajectory::evaluate(double t) const
{
  double u = 0.0;
  const int i = interval(t, u);
  const JointKnot& a = knots_[i];
  const JointKnot& b = knots_[i + 1];
  const double dt = b.t - a.t;
  const double u2 = u * u, u3 = u2 * u;
  const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
  const double h10 = u3 - 2.0 * u2 + u;
  const double h01 = -2.0 * u3 + 3.0 * u2;
  const double h11 = u3 - u2;
  const double dh00 = (6.0 * u2 - 6.0 * u) / dt;
  const double dh10 = 3.0 * u2 - 4.0 * u + 1.0;
  const double dh01 = (-6.0 * u2 + 6.0 * u) / dt;
  const double dh11 = 3.0 * u2 - 2.0 * u;

  JointKnotEvaluation out;
  out.R = a.R * Exp(u * Log(M3D(a.R.transpose() * b.R)));
  out.p = h00 * a.p + h10 * dt * a.v + h01 * b.p + h11 * dt * b.v;
  out.v = dh00 * a.p + dh10 * a.v + dh01 * b.p + dh11 * b.v;
  return out;
}

V3D JointKnotTrajectory::worldPoint(double point_time, const V3D& q_body,
                                    JointKnotResidualTime /*residual_time*/) const
{
  // Both residual-time policies describe the same nominal physical point.
  // Tail-time freezes its already-deskewed tail-frame coordinate only for
  // differentiation; it must not pretend the raw return was captured at t1.
  const JointKnotEvaluation x = evaluate(point_time);
  return x.R * q_body + x.p;
}

Eigen::MatrixXd JointKnotTrajectory::worldPointJacobian(
    double point_time, const V3D& q_body,
    JointKnotResidualTime residual_time) const
{
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, dim());
  // Must clear Exp()'s own small-angle threshold (utils/algo/math.h,
  // ang_norm > 0.0000001) with margin: at eps==1e-7 exactly, every rotation
  // perturbation this finite difference applies via applyDelta()->Exp()
  // silently evaluates to the identity, making every theta/rotation column
  // of this Jacobian exactly zero (confirmed directly: perturbed and
  // unperturbed worldPoint() were bit-identical for every rotation column).
  // That discarded all LiDAR attitude sensitivity in Measurement mode (and,
  // via evaluate()'s u=1 endpoint identity a.R*(a.R^T*b.R)=b.R, in Tail mode
  // too) and was confirmed the dominant cause of the catastrophic ATE
  // divergence reported for the coupled_joint_knots_r36_matched campaign.
  constexpr double eps = 1e-6;
  std::vector<int> columns;
  const JointKnotEvaluation tail = evaluate(knots_.back().t);
  const V3D nominal_world = worldPoint(
      point_time, q_body, JointKnotResidualTime::Measurement);
  const V3D q_tail_fixed = tail.R.transpose() * (nominal_world - tail.p);
  if (residual_time == JointKnotResidualTime::Tail) {
    const int o = knotOffset(knotCount() - 1);
    for (int c = 0; c < 9; ++c) columns.push_back(o + c);
  } else {
    double u = 0.0;
    const int i = interval(point_time, u);
    for (const int k : {i, i + 1}) {
      const int o = knotOffset(k);
      if (o < 0) continue;  // immutable head has no optimization columns.
      for (int c = 0; c < 9; ++c) columns.push_back(o + c);
    }
  }
  for (const int c : columns) {
    JointKnotTrajectory plus = *this, minus = *this;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(dim());
    d(c) = eps; plus.applyDelta(d);
    d(c) = -eps; minus.applyDelta(d);
    if (residual_time == JointKnotResidualTime::Tail) {
      const JointKnotEvaluation xp = plus.evaluate(plus.knots().back().t);
      const JointKnotEvaluation xm = minus.evaluate(minus.knots().back().t);
      J.col(c) = ((xp.R * q_tail_fixed + xp.p) -
                  (xm.R * q_tail_fixed + xm.p)) / (2.0 * eps);
    } else {
      J.col(c) = (plus.worldPoint(point_time, q_body, residual_time) -
                  minus.worldPoint(point_time, q_body, residual_time)) / (2.0 * eps);
    }
  }
  // A LiDAR residual has no direct bias/gravity columns.  Their correction
  // comes only through P's cross-covariance with physical knot states.
  return J;
}

Eigen::VectorXd JointKnotTrajectory::boxminus(const JointKnotTrajectory& prior) const
{
  if (prior.knotCount() != knotCount())
    throw std::runtime_error("joint-knot boxminus dimension mismatch");
  Eigen::VectorXd vec = Eigen::VectorXd::Zero(dim());
  for (int k = 1; k < knotCount(); ++k) {
    const int o = knotOffset(k);
    vec.segment<3>(o) = Log(M3D(knots_[k].R.transpose() * prior.knots_[k].R));
    vec.segment<3>(o + 3) = prior.knots_[k].p - knots_[k].p;
    vec.segment<3>(o + 6) = prior.knots_[k].v - knots_[k].v;
  }
  for (int k = 0; k < knotCount(); ++k) {
    const int o = biasOffset(k);
    vec.segment<3>(o) = prior.bg_[k] - bg_[k];
    vec.segment<3>(o + 3) = prior.ba_[k] - ba_[k];
  }
  vec.segment<3>(gravityOffset()) = prior.g_ - g_;
  return vec;
}

void JointKnotTrajectory::applyDelta(const Eigen::VectorXd& dx)
{
  if (dx.size() != dim()) throw std::runtime_error("joint-knot delta dimension mismatch");
  for (int k = 1; k < knotCount(); ++k) {
    const int o = knotOffset(k);
    knots_[k].R = knots_[k].R * Exp(dx.segment<3>(o));
    knots_[k].p += dx.segment<3>(o + 3);
    knots_[k].v += dx.segment<3>(o + 6);
  }
  for (int k = 0; k < knotCount(); ++k) {
    const int o = biasOffset(k);
    bg_[k] += dx.segment<3>(o);
    ba_[k] += dx.segment<3>(o + 3);
  }
  g_ += dx.segment<3>(gravityOffset());
}

JointKnotPrior buildJointKnotImuPrior(
    const JointKnotTrajectory& trajectory, const StateGroup& state,
    const std::vector<Pose6D>& poses,
    const std::vector<Eigen::MatrixXd>& transitions,
    const std::vector<Eigen::MatrixXd>& process_covariances,
    const Eigen::MatrixXd& scan_head_covariance)
{
  const int N = trajectory.knotCount();
  const int D = trajectory.dim();
  constexpr int E = 18;
  if (state.dimState() != E || scan_head_covariance.rows() != E ||
      scan_head_covariance.cols() != E || transitions.size() != poses.size() ||
      process_covariances.size() != poses.size())
    throw std::runtime_error("joint-knot prior requires aligned 18-state production F/Q records");
  std::vector<Eigen::Matrix<double, E, E>> P(N), Phi_from_prev(N);

  // Knot zero is immutable in the MEAN solve, not perfectly known.  Its
  // production covariance must propagate into every future knot.  The solve
  // state is the MARGINAL over mutable knots, knot-specific biases, and the
  // shared gravity variable;
  // conditioning on a zero head error would incorrectly discard the common-
  // mode pose uncertainty and leave only tiny within-scan process noise.
  // That mistake made the first LiDAR correction orders of magnitude too
  // small even though the production filter still carried head uncertainty.
  P[0] = 0.5 * (scan_head_covariance + scan_head_covariance.transpose());
  Eigen::LDLT<Eigen::Matrix<double,E,E>> head_ldlt(P[0]);
  if (head_ldlt.info() != Eigen::Success || !head_ldlt.isPositive())
    throw std::runtime_error("scan-head covariance is not positive definite");

  for (int k = 1; k < N; ++k) {
    const JointKnot& a = trajectory.knots()[k - 1];
    const JointKnot& b = trajectory.knots()[k];
    Eigen::Matrix<double,E,E> F = Eigen::Matrix<double,E,E>::Identity();
    Eigen::Matrix<double,E,E> Q = Eigen::Matrix<double,E,E>::Zero();
    int used = 0;
    for (size_t m = 0; m < poses.size(); ++m) {
      const double interval_start = poses[m].t;
      const double interval_end = poses[m].t + poses[m].dt;
      if (interval_start + 1e-12 < a.t || interval_end > b.t + 1e-12) continue;
      const Eigen::Matrix<double,E,E> Fm = transitions[m];
      const Eigen::Matrix<double,E,E> Qm = process_covariances[m];
      // Keep the complete production process covariance. In particular,
      // Q_bg,bg and Q_ba,ba now create uncertainty in the next knot's bias
      // state instead of being projected out to support a constant-bias
      // approximation.
      Q = Fm * Q * Fm.transpose() + Qm;
      F = Fm * F;
      ++used;
    }
    if (used == 0)
      throw std::runtime_error("no production IMU intervals found between adjacent knots");
    Phi_from_prev[k] = F;
    P[k] = F * P[k - 1] * F.transpose() + Q;
  }

  // Select the retained variables from the full Markov chain. This is a
  // marginal, not a conditioned covariance: immutable head R/P/V uncertainty
  // still appears in every retained block through production propagation.
  struct BlockSelection { int joint; int knot; int production; };
  std::vector<BlockSelection> selected;
  for (int k = 1; k < N; ++k)
    for (int c = 0; c < 3; ++c)
      selected.push_back({trajectory.knotOffset(k) + 3 * c, k, 3 * c});
  for (int k = 0; k < N; ++k) {
    selected.push_back({trajectory.biasOffset(k), k, 9});
    selected.push_back({trajectory.biasOffset(k) + 3, k, 12});
  }
  // Gravity is physically shared and Q_g,g is zero. Retaining the scan-head
  // copy avoids a singular stack of identical gravity variables.
  selected.push_back({trajectory.gravityOffset(), 0, 15});

  const auto crossTimeCovariance = [&](int a, int b)
      -> Eigen::Matrix<double,E,E> {
    Eigen::Matrix<double,E,E> Phi = Eigen::Matrix<double,E,E>::Identity();
    if (a == b) return P[a];
    if (a < b) {
      for (int k = a + 1; k <= b; ++k) Phi = Phi_from_prev[k] * Phi;
      return Eigen::Matrix<double,E,E>(P[a] * Phi.transpose());
    }
    for (int k = b + 1; k <= a; ++k) Phi = Phi_from_prev[k] * Phi;
    return Eigen::Matrix<double,E,E>(Phi * P[b]);
  };

  Eigen::MatrixXd joint = Eigen::MatrixXd::Zero(D, D);
  for (const BlockSelection& row : selected) {
    for (const BlockSelection& col : selected) {
      const Eigen::Matrix<double,E,E> cross =
          crossTimeCovariance(row.knot, col.knot);
      joint.block<3,3>(row.joint, col.joint) =
          cross.block<3,3>(row.production, col.production);
    }
  }
  JointKnotPrior out;
  out.P = 0.5 * (joint + joint.transpose());
  out.P_inverse = symmetricInverse(out.P);
  return out;
}

JointLidarInformation accumulateJointLidarInformation(
    const std::vector<JointLidarRow>& rows, int state_dimension,
    const ResidualRedundancyOptions& options)
{
  if (options.mode != "off" && options.mode != "woodbury")
    throw std::invalid_argument(
        "joint-knot LiDAR information supports only residual_redundancy "
        "modes off and woodbury");
  if (options.rho < 0.0 || options.rho > 1.0 ||
      options.max_discount < 0.0 || options.max_discount > 1.0)
    throw std::invalid_argument(
        "joint-knot residual redundancy requires rho and max_discount in [0,1]");
  JointLidarInformation out;
  out.Gamma = Eigen::MatrixXd::Zero(state_dimension, state_dimension);
  out.b = Eigen::VectorXd::Zero(state_dimension);
  out.mode = options.mode == "off" ? "independent" : "woodbury_plane";

  // Preserve first-observed plane order. Pointer-hash iteration order would
  // make the floating-point summation depend on allocator addresses even for
  // an otherwise deterministic run.
  std::unordered_map<const void*, size_t> group_index;
  std::vector<std::vector<const JointLidarRow*>> groups;
  for (const JointLidarRow& row : rows)
  {
    if (row.H.size() != state_dimension)
      throw std::invalid_argument("joint LiDAR row has the wrong state dimension");
    const double w = 1.0 / std::max(row.sigma_squared, 1e-12);
    out.Gamma.noalias() += w * row.H.transpose() * row.H;
    out.b.noalias() += -w * row.H.transpose() * row.residual;
    if (row.plane_id != nullptr)
    {
      auto inserted = group_index.emplace(row.plane_id, groups.size());
      if (inserted.second) groups.emplace_back();
      groups[inserted.first->second].push_back(&row);
    }
  }

  Eigen::MatrixXd delta_gamma = Eigen::MatrixXd::Zero(
      state_dimension, state_dimension);
  Eigen::VectorXd delta_b = Eigen::VectorXd::Zero(state_dimension);
  double naive_trace_total = 0.0, corrected_trace_total = 0.0;
  for (const auto& group : groups)
  {
    if (group.size() < 2) continue;
    ++out.redundancy_stats.redund_groups_seen;
    const double shared = options.rho * group.front()->plane_var_term;
    for (const JointLidarRow* row : group)
      if (std::abs(row->plane_var_term - group.front()->plane_var_term) >
          1e-9 * std::max(1.0, std::abs(group.front()->plane_var_term)))
        throw std::invalid_argument(
            "rows sharing a plane_id must share plane_var_term");
    if (!(shared > 0.0))
    {
      ++out.redundancy_stats.redund_groups_degenerate_pv;
      continue;
    }
    std::vector<double> independent_weights;
    independent_weights.reserve(group.size());
    bool valid = true;
    for (const JointLidarRow* row : group)
    {
      const double independent_variance = row->sigma_squared - shared;
      if (!(independent_variance > 0.0)) { valid = false; break; }
      independent_weights.push_back(1.0 / independent_variance);
    }
    if (!valid)
    {
      ++out.redundancy_stats.redund_groups_degenerate_var;
      continue;
    }

    Eigen::MatrixXd naive_gamma = Eigen::MatrixXd::Zero(
        state_dimension, state_dimension);
    Eigen::VectorXd naive_b = Eigen::VectorXd::Zero(state_dimension);
    Eigen::MatrixXd corrected_gamma = Eigen::MatrixXd::Zero(
        state_dimension, state_dimension);
    Eigen::VectorXd corrected_b = Eigen::VectorXd::Zero(state_dimension);
    Eigen::VectorXd sum_weighted_h = Eigen::VectorXd::Zero(state_dimension);
    double sum_weighted_r = 0.0, sum_weight = 0.0;
    for (size_t i = 0; i < group.size(); ++i)
    {
      const JointLidarRow& row = *group[i];
      const double naive_weight = 1.0 / std::max(row.sigma_squared, 1e-12);
      const double weight = independent_weights[i];
      naive_gamma.noalias() += naive_weight * row.H.transpose() * row.H;
      naive_b.noalias() += -naive_weight * row.H.transpose() * row.residual;
      corrected_gamma.noalias() += weight * row.H.transpose() * row.H;
      corrected_b.noalias() += -weight * row.H.transpose() * row.residual;
      sum_weighted_h.noalias() += weight * row.H.transpose();
      sum_weighted_r += weight * row.residual;
      sum_weight += weight;
    }
    const double coefficient = shared / (1.0 + shared * sum_weight);
    corrected_gamma.noalias() -= coefficient *
        sum_weighted_h * sum_weighted_h.transpose();
    corrected_b.noalias() += coefficient * sum_weighted_h * sum_weighted_r;

    const double naive_trace = naive_gamma.trace();
    if (naive_trace > 0.0)
    {
      const double discount = 1.0 - corrected_gamma.trace() / naive_trace;
      if (discount > options.max_discount)
      {
        const double scale = options.max_discount / discount;
        corrected_gamma = naive_gamma + scale * (corrected_gamma - naive_gamma);
        corrected_b = naive_b + scale * (corrected_b - naive_b);
        ++out.redundancy_stats.max_discount_bound_groups;
      }
    }
    ++out.redundancy_stats.redund_groups;
    out.redundancy_stats.redund_n_raw += static_cast<int>(group.size());
    const double corrected_trace = corrected_gamma.trace();
    naive_trace_total += naive_trace;
    corrected_trace_total += corrected_trace;
    if (naive_trace > 0.0)
      out.redundancy_stats.redund_n_eff += static_cast<int>(std::ceil(
          group.size() * corrected_trace / naive_trace));
    delta_gamma.noalias() += corrected_gamma - naive_gamma;
    delta_b.noalias() += corrected_b - naive_b;
  }
  out.redundancy_stats.naive_info_gain = naive_trace_total;
  out.redundancy_stats.woodbury_info_gain = corrected_trace_total;
  if (naive_trace_total > 0.0)
    out.redundancy_stats.redund_info_ratio =
        corrected_trace_total / naive_trace_total;
  if (options.mode == "woodbury")
  {
    out.Gamma += delta_gamma;
    out.b += delta_b;
  }
  return out;
}

JointKnotSolve solveJointKnotInformation(
    const JointKnotTrajectory& current,
    const JointKnotTrajectory& imu_prior_mean,
    const JointKnotPrior& prior,
    const std::vector<Residual>& residuals,
    const std::vector<PointXYZCov>& evaluated_points,
    JointKnotResidualTime residual_time,
    const ResidualRedundancyOptions& redundancy_options)
{
  JointKnotSolve out;
  const int D = current.dim();
  std::vector<JointLidarRow> lidar_rows;
  lidar_rows.reserve(residuals.size());
  out.residual_count_by_interval.assign(
      std::max(0, current.knotCount() - 1), 0);
  double abs_r = 0.0;
  for (const Residual& residual : residuals) {
    if (residual.source_index < 0 ||
        residual.source_index >= static_cast<int>(evaluated_points.size())) continue;
    const PointXYZCov& point = evaluated_points[residual.source_index];
    const Eigen::MatrixXd Jpoint = current.worldPointJacobian(
        point.t, point.raw_body_point, residual_time);
    const Eigen::RowVectorXd H = residual.normal.transpose() * Jpoint;
    if (H.squaredNorm() <= 1e-24) ++out.zero_jacobian_residual_count;
    if (!out.residual_count_by_interval.empty()) {
      int interval_index = 0;
      while (interval_index + 1 < current.knotCount() - 1 &&
             point.t >= current.knots()[interval_index + 1].t)
        ++interval_index;
      ++out.residual_count_by_interval[interval_index];
    }
    lidar_rows.push_back(JointLidarRow{
        H, residual.r, residual.sigma_squared,
        residual.plane_var_term, residual.plane_id});
    abs_r += std::abs(residual.r);
  }
  const JointLidarInformation lidar = accumulateJointLidarInformation(
      lidar_rows, D, redundancy_options);
  out.Gamma_L = lidar.Gamma;
  out.b_L = lidar.b;
  out.redundancy_stats = lidar.redundancy_stats;
  out.lidar_information_mode = lidar.mode;
  out.mean_abs_residual = residuals.empty() ? 0.0 : abs_r / residuals.size();
  out.vec = current.boxminus(imu_prior_mean);
  out.A = prior.P_inverse + out.Gamma_L;
  out.K1 = symmetricInverse(out.A);
  out.lidar_rhs = out.b_L;
  out.prior_rhs = prior.P_inverse * out.vec;
  out.rhs = out.lidar_rhs + out.prior_rhs;
  out.lidar_delta = out.K1 * out.lidar_rhs;
  out.prior_delta = out.K1 * out.prior_rhs;
  out.delta = out.lidar_delta + out.prior_delta;
  out.posterior = out.K1;
  return out;
}

Eigen::MatrixXd extractTailStateCovariance(const JointKnotTrajectory& trajectory,
                                           const Eigen::MatrixXd& joint_cov,
                                           const StateGroup& state)
{
  Eigen::MatrixXd out = state.cov();
  const int tail = trajectory.knotOffset(trajectory.knotCount() - 1);
  const int tail_bias = trajectory.biasOffset(trajectory.knotCount() - 1);
  const int gravity = trajectory.gravityOffset();
  const int state_idx[6] = {StateGroup::idxR(), StateGroup::idxP(), StateGroup::idxV(),
                            state.idxBG(), state.idxBA(), state.idxG()};
  const int joint_idx[6] = {tail, tail + 3, tail + 6,
                            tail_bias, tail_bias + 3, gravity};
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      if (state_idx[i] >= 0 && state_idx[j] >= 0)
        out.block<3,3>(state_idx[i],state_idx[j]) =
            joint_cov.block<3,3>(joint_idx[i],joint_idx[j]);
  return 0.5 * (out + out.transpose());
}

}  // namespace livo_recon
