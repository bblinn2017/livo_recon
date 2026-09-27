#pragma once

#include "livo_recon/diagnostics/log/debug_log_dir.h"

#include <Eigen/Core>
#include <map>
#include <string>

namespace livo_recon
{

// Mode-independent named diagnostic payload.  Producers describe WHAT was
// computed; this writer owns HOW it is serialized.  Adding a field therefore
// does not require another hand-written stream block in every estimator mode.
struct PoseControlDiagnosticRecord
{
  std::map<std::string,std::string> labels;
  std::map<std::string,double> scalars;
  std::map<std::string,Eigen::VectorXd> vectors;
  std::map<std::string,Eigen::MatrixXd> matrices;
};

class PoseControlDiagnosticWriter
{
public:
  explicit PoseControlDiagnosticWriter(std::string basename) : stream_(std::move(basename)) {}
  void append(const std::string& section,const PoseControlDiagnosticRecord& record);

  static void writeMatrix(std::ostream&,const std::string&,const Eigen::MatrixXd&);
  static void writeVector(std::ostream&,const std::string&,const Eigen::VectorXd&);

private:
  PersistentLogStream stream_;
};

} // namespace livo_recon
