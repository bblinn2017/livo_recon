#include "livo_recon/diagnostics/state_trace.h"

#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <fstream>
#include <iomanip>

namespace livo_recon
{
namespace
{
constexpr int kTraceDim = 18;

void writeDiag(std::ofstream& out, const Eigen::MatrixXd& M)
{
  for (int i = 0; i < kTraceDim; ++i) {
    out << ',';
    if (i < M.rows() && i < M.cols()) out << M(i, i);
  }
}
}  // namespace

void writeStateTraceRow(const std::string& run_id, int scan_id, double t_abs, const char* phase,
                        const StateGroup& s, int residual_count, int completed_iterations,
                        int open_loop_window, int open_loop_reset)
{
  static PersistentLogStream log("state_trace.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first) {
    out << "run_id,scan_id,t_abs,phase,dim,qw,qx,qy,qz,px,py,pz,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,gx,gy,gz,"
           "residual_count,completed_iterations,ol_window,ol_reset";
    for (int i = 0; i < kTraceDim; ++i)
      for (int j = i; j < kTraceDim; ++j) out << ",P_" << i << '_' << j;
    out << '\n';
  }
  const Eigen::MatrixXd& P = s.cov();
  const Eigen::Quaterniond q(s.rot());
  out << std::setprecision(17) << run_id << ',' << scan_id << ',' << t_abs << ',' << phase << ',' << P.rows() << ','
      << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
      << s.pos().x() << ',' << s.pos().y() << ',' << s.pos().z() << ','
      << s.vel().x() << ',' << s.vel().y() << ',' << s.vel().z() << ','
      << s.biasGyr().x() << ',' << s.biasGyr().y() << ',' << s.biasGyr().z() << ','
      << s.biasAcc().x() << ',' << s.biasAcc().y() << ',' << s.biasAcc().z() << ','
      << s.gravity().x() << ',' << s.gravity().y() << ',' << s.gravity().z() << ','
      << residual_count << ',' << completed_iterations << ',' << open_loop_window << ',' << open_loop_reset;
  for (int i = 0; i < kTraceDim; ++i)
    for (int j = i; j < kTraceDim; ++j) {
      out << ',';
      if (i < P.rows() && j < P.cols()) out << P(i, j);
    }
  out << '\n';
  out.flush();
}

void writeImuCovGrowthRow(size_t scan_index, double t_abs, const Eigen::MatrixXd& P_start,
                          const Eigen::MatrixXd& P_end, const Eigen::MatrixXd& Q_acc)
{
  static PersistentLogStream log("imu_cov_growth.csv");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (first) {
    out << "scan_index,t_abs,dim";
    for (const char* tag : {"Pstart", "Pend", "Qacc"})
      for (int i = 0; i < kTraceDim; ++i) out << ',' << tag << '_' << i;
    out << '\n';
  }
  out << std::setprecision(17) << scan_index << ',' << t_abs << ',' << P_end.rows();
  writeDiag(out, P_start);
  writeDiag(out, P_end);
  writeDiag(out, Q_acc);
  out << '\n';
  out.flush();
}

void writeStateReference(const Eigen::Matrix3d& R_ref, const Eigen::Vector3d& p_ref, const Eigen::Vector3d& v_ref,
                         const Eigen::MatrixXd& P0)
{
  static PersistentLogStream log("state_reference.txt");
  bool first = false;
  std::ofstream& out = log.stream(&first);
  if (!first) return;
  const Eigen::Quaterniond q(R_ref);
  out << std::setprecision(17) << "schema state_reference_v1\n"
      << "q_wxyz " << q.w() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << '\n'
      << "p " << p_ref.x() << ' ' << p_ref.y() << ' ' << p_ref.z() << '\n'
      << "v " << v_ref.x() << ' ' << v_ref.y() << ' ' << v_ref.z() << '\n'
      << "P0_dim " << P0.rows() << '\n' << "P0_row_major";
  for (int i = 0; i < P0.rows(); ++i)
    for (int j = 0; j < P0.cols(); ++j) out << ' ' << P0(i, j);
  out << '\n';
  out.flush();
}

}  // namespace livo_recon
