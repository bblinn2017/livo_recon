#pragma once
// R74 (F-424): a per-scan log row is built completely in memory, written with one call and flushed at once, so a process that aborts
// right after (SIGABRT, an uncaught exception) leaves every completed row in the file and never a half row.
#include <fstream>
#include <string>

namespace livo_recon
{
inline void appendRowAndFlush(std::ofstream& out, const std::string& row)
{
  out.write(row.data(), static_cast<std::streamsize>(row.size()));
  out.flush();
}
}  // namespace livo_recon
