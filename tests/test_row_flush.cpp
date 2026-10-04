// R74 unit test (F-424): a per-scan row is on disk the moment it is appended, so an abort leaves every completed row
// (include/livo_recon/utils/log/row_flush.h).
//   RF1 a child process appends N = 7 rows with appendRowAndFlush() and then calls std::abort(): the file has the header and exactly 7 rows,
//       each complete (the right number of fields). N = 0 leaves only the header.
//   RF2 control: the same child that writes N rows with plain operator<< and NO flush and then aborts loses rows (fewer than N complete rows),
//       so RF1 can tell a flushed writer from an unflushed one. (A large row keeps it above the stream buffer so the loss is not size luck; the
//       control only has to show that the unflushed pattern can lose rows.)
//   RF3 the row is written by one call: a row of 4 fields is never split (every line has 3 commas).
// POSIX only (fork). Runs in the build directory's temporary files.
#include "livo_recon/utils/log/row_flush.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(c, msg)                                                                       \
  do {                                                                                      \
    if (!(c)) { std::fprintf(stderr, "[FAIL] %s  (%s:%d: %s)\n", msg, __FILE__, __LINE__, #c); std::exit(1); } \
    else { std::printf("[PASS] %s\n", msg); }                                               \
  } while (0)

using namespace livo_recon;

static std::vector<std::string> readLines(const std::string& path)
{
  std::ifstream in(path);
  std::vector<std::string> v;
  std::string l;
  while (std::getline(in, l)) v.push_back(l);
  return v;
}

static void runChild(const std::string& path, int n_rows, bool flushed)
{
  const pid_t pid = fork();
  if (pid == 0)
  {
    std::ofstream out(path, std::ios::trunc);
    out << "frame,a,b,c\n";
    out.flush();   // the header is written like the real writers do (a first write that is not part of the per-row rule)
    for (int i = 1; i <= n_rows; ++i)
    {
      std::ostringstream row;
      row << i << ",1.5,2.5,3.5\n";
      if (flushed) appendRowAndFlush(out, row.str());
      else out << row.str();   // no flush
    }
    std::abort();   // never returns; the destructor of `out` does not run
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFSIGNALED(status)) { std::fprintf(stderr, "[FAIL] child did not abort\n"); std::exit(1); }
}

static int completeRows(const std::string& path)
{
  const auto lines = readLines(path);
  int n = 0;
  for (size_t i = 1; i < lines.size(); ++i)
  {
    int commas = 0;
    for (char ch : lines[i]) if (ch == ',') ++commas;
    if (commas == 3) ++n;
    else CHECK(false, "RF3 a row was split or truncated");
  }
  return n;
}

int main()
{
  const std::string path = "test_row_flush_tmp.csv";
  runChild(path, 7, true);
  CHECK(readLines(path).size() == 8, "RF1 header plus 7 lines after the abort");
  CHECK(completeRows(path) == 7, "RF1/RF3 exactly 7 complete rows after an abort");
  runChild(path, 0, true);
  CHECK(readLines(path).size() == 1 && completeRows(path) == 0, "RF1 N = 0 leaves only the header");
  runChild(path, 7, false);
  const auto lines = readLines(path);
  CHECK(lines.size() < 8, "RF2 control: without the flush the rows written after the header are lost on abort");
  std::remove(path.c_str());
  std::printf("ALL PASS\n");
  return 0;
}
