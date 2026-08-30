#include "livo_recon/utils/log/debug_log_dir.h"

#include <filesystem>

namespace livo_recon
{

namespace
{
std::string& debugLogDirStorage()
{
  static std::string dir;  // empty -> falls back to /tmp, see header doc comment
  return dir;
}
}  // namespace

void setDebugLogDir(const std::string& dir)
{
  debugLogDirStorage() = dir;
  if (!dir.empty())
    std::filesystem::create_directories(dir);
}

std::string debugLogPath(const std::string& basename)
{
  const std::string& dir = debugLogDirStorage();
  return (dir.empty() ? std::string("/tmp/") : dir + "/") + basename;
}

}  // namespace livo_recon
