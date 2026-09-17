#pragma once

#include <fstream>
#include <string>

// Shared destination directory for this codebase's fixed-basename debug
// logs (lio.txt, evo.txt, imu.txt, the myvio per-iteration/per-point dumps,
// etc.) -- previously each of these hardcoded its own "/tmp/<name>.txt"
// path directly, which meant every test run needed a separate copy-out
// step (see run_one()'s DEBUG_LOG_DEST mechanism) to preserve them before
// the next run's SAME hardcoded path overwrote them. Set once, at startup,
// from outputs/debug_log_dir (see LivoReconNode::loadParameters()) -- empty
// (the default, e.g. for ad hoc/manual runs) falls back to "/tmp" exactly
// like the old hardcoded behavior. Mirrors FAST-LIVO2's own fix for the
// identical problem (LIVMapper::readParameters() redirecting vio/log_path/
// livo_vio/log_path to test_output_dir_/log/<basename> when set).
namespace livo_recon
{

void setDebugLogDir(const std::string& dir);

// Returns "<debugLogDir>/<basename>" ("/tmp/<basename>" if unset). Every
// per-file debugLogXxx() helper and every *Options::log_path default in
// this codebase should build its path through this function rather than
// hardcoding "/tmp/...", so a single outputs/debug_log_dir setting (e.g.
// pointed at a test run's own log/ directory) captures all of them.
std::string debugLogPath(const std::string& basename);

// CQ-36: fixes a real regression introduced by CQ-35's "open once, keep the
// handle" optimization. The OLD per-call pattern (a fresh std::ofstream on
// every call) re-resolved debugLogPath() every time, which was self-healing
// against outputs/debug_log_dir being set AFTER a debugLogXxx() function's
// very first call in the process (confirmed happening in practice: every
// fixed-basename debug file landed at /tmp, 0 bytes, for the entire
// duration of a real run -- CQ-35's static ofstream froze onto whatever
// debugLogPath() returned on the FIRST call and never re-checked, and never
// flushed, so a run's diagnostic files were both silently misplaced AND
// invisible to any monitoring/tail until process exit). This class restores
// both properties -- re-resolves the path on every stream() call (reopening
// only when it actually changed, so the common case still pays no
// open-per-call cost) and flushes after use is the caller's job (flush()
// below) -- while a caller still only needs one instance (a function-local
// static) rather than the raw open/close-every-call code this replaces.
class PersistentLogStream
{
public:
  explicit PersistentLogStream(std::string basename) : basename_(std::move(basename)) {}

  // Returns a ready-to-write stream. *just_opened (if non-null) reports
  // whether THIS call (re)opened the file -- true on the very first call,
  // and again if outputs/debug_log_dir ever changes underneath a long-lived
  // process -- so a caller can gate a one-time header write on it, exactly
  // like the old first_call flag did.
  std::ofstream& stream(bool* just_opened = nullptr)
  {
    const std::string path = debugLogPath(basename_);
    bool opened_now = false;
    if (path != opened_path_)
    {
      if (ofs_.is_open()) ofs_.close();
      ofs_.open(path, std::ios::trunc);
      opened_path_ = path;
      opened_now = true;
    }
    if (just_opened) *just_opened = opened_now;
    return ofs_;
  }

private:
  std::string basename_;
  std::string opened_path_;
  std::ofstream ofs_;
};

}  // namespace livo_recon
