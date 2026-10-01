#pragma once

#include <Eigen/Dense>
#include <cstdlib>
#include <iomanip>
#include <istream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>

// ============================================================================
// R72 (F-398): lossless text dump of named matrices. One block per matrix:
//     ## <name> <rows> <cols>
//     <cols values, 17 significant digits, space separated>      (one line per row)
// 17 significant digits round-trips an IEEE double exactly. Write-only diagnostics: nothing in the estimator reads these files.
// parseMatrixDump() exists for the unit test and for the Python reader's reference behaviour.
// ============================================================================

namespace livo_recon
{

inline void dumpMatrix(std::ostream& o, const std::string& name, const Eigen::MatrixXd& M)
{
  o << "## " << name << ' ' << M.rows() << ' ' << M.cols() << '\n';
  o << std::scientific << std::setprecision(16);
  for (Eigen::Index i = 0; i < M.rows(); ++i)
  {
    for (Eigen::Index j = 0; j < M.cols(); ++j) o << (j ? " " : "") << M(i, j);
    o << '\n';
  }
}

inline void dumpVector(std::ostream& o, const std::string& name, const Eigen::VectorXd& v)
{
  dumpMatrix(o, name, Eigen::MatrixXd(v));
}

inline void dumpScalar(std::ostream& o, const std::string& name, double x)
{
  Eigen::MatrixXd m(1, 1);
  m(0, 0) = x;
  dumpMatrix(o, name, m);
}

inline void dumpText(std::ostream& o, const std::string& name, const std::string& text)
{
  o << "# " << name << ": " << text << '\n';
}

// Returns false on a malformed block. Lines starting with a single '#' are comments.
inline bool parseMatrixDump(std::istream& in, std::map<std::string, Eigen::MatrixXd>& out)
{
  std::string line;
  while (std::getline(in, line))
  {
    if (line.rfind("## ", 0) != 0) continue;
    std::istringstream head(line.substr(3));
    std::string name;
    long rows = 0, cols = 0;
    if (!(head >> name >> rows >> cols) || rows < 0 || cols < 0) return false;
    Eigen::MatrixXd M(rows, cols);
    for (long i = 0; i < rows; ++i)
    {
      if (!std::getline(in, line)) return false;
      std::istringstream row(line);
      for (long j = 0; j < cols; ++j)
      {
        std::string tok;
        if (!(row >> tok)) return false;
        M(i, j) = std::strtod(tok.c_str(), nullptr);   // strtod, not stod: stod throws on a subnormal result
      }
    }
    out[name] = M;
  }
  return true;
}

}  // namespace livo_recon
