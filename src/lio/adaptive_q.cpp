#include "livo_recon/lio/adaptive_q.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace livo_recon
{

void AdaptiveQ::setNominal(double var_acc_nom, double var_gyr_nom)
{
  nom_acc_ = var_acc_nom;
  nom_gyr_ = var_gyr_nom;
  applied_acc_ = nom_acc_;
  applied_gyr_ = nom_gyr_;
}

void AdaptiveQ::setFloor(double var_acc_floor, double var_gyr_floor)
{
  floor_acc_ = var_acc_floor;
  floor_gyr_ = var_gyr_floor;
}

double AdaptiveQ::zFromRatio(double meas, double nom, double beta)
{
  if (!(meas > 0.0) || !(nom > 0.0) || !(beta > 0.0)) return 0.0;
  const double r = std::log10(meas / nom) / beta;
  // atanh is only defined on (-1,1); anything outside saturates, which is
  // exactly the intended behaviour -- a measurement far outside the
  // excursion band pins z at the edge instead of escaping the band.
  const double c = std::max(-0.999, std::min(0.999, r));
  return std::atanh(c);
}

double AdaptiveQ::applyZ(double nom, double beta, double z)
{
  return nom * std::pow(10.0, beta * std::tanh(z));
}

bool AdaptiveQ::update(const SplineImuResidualStats& st)
{
  if (!opts_.enable) { status_ = "off"; return false; }

  frames_++;

  if (!st.valid()) { status_ = "no_residual"; return false; }
  if (!std::isfinite(st.cov_acc) || !std::isfinite(st.cov_gyr) ||
      st.cov_acc <= 0.0 || st.cov_gyr <= 0.0)
  { status_ = "bad_residual"; return false; }

  // ---- anchor 2: whiteness -----------------------------------------------
  // Checked BEFORE the floor, because a correlated residual is not a noise
  // measurement at all -- its magnitude is not evidence about anything.
  const bool white_acc = std::abs(st.acf1_acc) <= opts_.acf1_max;
  const bool white_gyr = std::abs(st.acf1_gyr) <= opts_.acf1_max;
  if (!white_acc && !white_gyr) { status_ = "not_white"; return false; }

  // ---- anchor 1: the physical floor --------------------------------------
  // A residual below the sensor's own measured noise means the spline is
  // fitting noise -- the control-point count is too high for this window.
  // Refuse rather than clamp: clamping would report a floored estimate as
  // if it were a measurement, and the no-valid-window case is exactly the
  // interesting one.
  const double fl_a = opts_.use_noise_floor ? floor_acc_ * opts_.noise_floor_scale : 0.0;
  const double fl_g = opts_.use_noise_floor ? floor_gyr_ * opts_.noise_floor_scale : 0.0;
  const bool above_a = !(fl_a > 0.0) || st.cov_acc >= fl_a;
  const bool above_g = !(fl_g > 0.0) || st.cov_gyr >= fl_g;
  if (!above_a && !above_g) { status_ = "below_floor"; return false; }

  // Both gates failing on DIFFERENT channels means no n_cp satisfies both
  // conditions simultaneously -- the model-misspecification case named in
  // the header.  Log it distinctly; it is a finding, not a hiccup.
  if ((white_acc && !above_a && !white_gyr && above_g) ||
      (white_gyr && !above_g && !white_acc && above_a))
  { status_ = "no_window"; return false; }

  // ---- EMA on the measurement --------------------------------------------
  if (!primed_) { meas_acc_ = st.cov_acc; meas_gyr_ = st.cov_gyr; primed_ = true; }
  else
  {
    const double a = std::max(0.0, std::min(0.999, opts_.ema));
    if (white_acc && above_a) meas_acc_ = a * meas_acc_ + (1.0 - a) * st.cov_acc;
    if (white_gyr && above_g) meas_gyr_ = a * meas_gyr_ + (1.0 - a) * st.cov_gyr;
  }

  if (frames_ <= opts_.warmup_frames)
  {
    status_ = "warmup";
    applied_acc_ = nom_acc_;
    applied_gyr_ = nom_gyr_;
    return false;
  }

  // ---- bounded excursion, rate limited -----------------------------------
  const double z_a_target = zFromRatio(meas_acc_, nom_acc_, opts_.beta_acc);
  const double z_g_target = zFromRatio(meas_gyr_, nom_gyr_, opts_.beta_gyr);
  const double lim = std::max(0.0, opts_.z_rate_limit);

  if (white_acc && above_a)
    z_acc_ += std::max(-lim, std::min(lim, z_a_target - z_acc_));
  if (white_gyr && above_g)
    z_gyr_ += std::max(-lim, std::min(lim, z_g_target - z_gyr_));

  double aa = applyZ(nom_acc_, opts_.beta_acc, z_acc_);
  double ag = applyZ(nom_gyr_, opts_.beta_gyr, z_gyr_);

  // ---- hard bounds -------------------------------------------------------
  aa = std::max(nom_acc_ * opts_.min_ratio, std::min(nom_acc_ * opts_.max_ratio, aa));
  ag = std::max(nom_gyr_ * opts_.min_ratio, std::min(nom_gyr_ * opts_.max_ratio, ag));
  if (fl_a > 0.0) aa = std::max(aa, fl_a);
  if (fl_g > 0.0) ag = std::max(ag, fl_g);

  if (!std::isfinite(aa) || !std::isfinite(ag) || aa <= 0.0 || ag <= 0.0)
  { status_ = "nonfinite"; return false; }

  const bool changed = (aa != applied_acc_) || (ag != applied_gyr_);
  applied_acc_ = aa;
  applied_gyr_ = ag;
  active_ = true;
  status_ = "ok";
  return changed;
}

std::string AdaptiveQ::csvHeader() const
{
  return "t_abs,status,n,cov_acc_meas,cov_gyr_meas,cov_acc_ema,cov_gyr_ema,"
         "acf1_acc,acf1_gyr,floor_acc,floor_gyr,nom_acc,nom_gyr,"
         "z_acc,z_gyr,applied_acc,applied_gyr,frames";
}

std::string AdaptiveQ::csvRow(double t_abs, const SplineImuResidualStats& st) const
{
  std::ostringstream o;
  o << std::setprecision(10) << t_abs << ',' << status_ << ',' << st.n << ','
    << std::scientific << std::setprecision(6)
    << st.cov_acc << ',' << st.cov_gyr << ','
    << meas_acc_ << ',' << meas_gyr_ << ','
    << std::fixed << std::setprecision(4)
    << st.acf1_acc << ',' << st.acf1_gyr << ','
    << std::scientific << std::setprecision(6)
    << floor_acc_ << ',' << floor_gyr_ << ','
    << nom_acc_ << ',' << nom_gyr_ << ','
    << std::fixed << std::setprecision(5)
    << z_acc_ << ',' << z_gyr_ << ','
    << std::scientific << std::setprecision(6)
    << applied_acc_ << ',' << applied_gyr_ << ','
    << frames_;
  return o.str();
}

}  // namespace livo_recon
