#pragma once

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

}  // namespace livo_recon
