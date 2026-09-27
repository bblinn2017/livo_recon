#include "livo_recon/diagnostics/pose_control/pose_control_diagnostic_writer.h"

#include <iomanip>
#include <ostream>

namespace livo_recon
{

void PoseControlDiagnosticWriter::writeMatrix(std::ostream& out,const std::string& name,
                                               const Eigen::MatrixXd& value)
{
  out<<"matrix "<<name<<" rows="<<value.rows()<<" cols="<<value.cols()<<'\n'<<std::setprecision(17);
  for(int r=0;r<value.rows();++r){for(int c=0;c<value.cols();++c){if(c)out<<' ';out<<value(r,c);}out<<'\n';}
}

void PoseControlDiagnosticWriter::writeVector(std::ostream& out,const std::string& name,
                                               const Eigen::VectorXd& value)
{
  out<<"vector "<<name<<" size="<<value.size()<<'\n'<<std::setprecision(17);
  for(int i=0;i<value.size();++i){if(i)out<<' ';out<<value(i);}out<<'\n';
}

void PoseControlDiagnosticWriter::append(const std::string& section,
                                         const PoseControlDiagnosticRecord& record)
{
  std::ofstream& out=stream_.stream();
  out<<"=== "<<section<<" ===\n";
  for(const auto& item:record.labels) out<<item.first<<'='<<item.second<<'\n';
  out<<std::setprecision(17);
  for(const auto& item:record.scalars) out<<item.first<<'='<<item.second<<'\n';
  for(const auto& item:record.matrices) writeMatrix(out,item.first,item.second);
  for(const auto& item:record.vectors) writeVector(out,item.first,item.second);
  out.flush();
}

} // namespace livo_recon
