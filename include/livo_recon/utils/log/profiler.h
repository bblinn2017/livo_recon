#pragma once

#include <chrono>
#include <string>
#include <map>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <mutex>
#include <memory>

namespace livo_recon
{

class FrameProfiler
{
  using Clock = std::chrono::steady_clock;

public:
  void add(const std::string& name, double ms)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    frame_acc_[name] += ms;
  }

  // Flush the current frame: updates running global averages, rewrites the
  // export file if configured, and returns a formatted report string.
  std::string report()
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (frame_acc_.empty()) return {};

    std::ostringstream ss;
    ss << "[profiler]";
    double total = 0.0;
    for (const auto& [name, ms] : frame_acc_)
    {
      auto& stat = global_stats_[name];
      stat.count++;
      stat.avg_ms += (ms - stat.avg_ms) / stat.count;
      ss << "\n  " << name << ": " << std::fixed << std::setprecision(1) << ms << " ms";
      total += ms;
    }
    ss << "\n  total: " << std::fixed << std::setprecision(1) << total << " ms";

    frame_acc_.clear();
    return ss.str();
  }

  void exportReport()
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!export_path_.empty())
      writeReport();
  }

  void setExportPath(const std::string& path)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    export_path_ = path;
  }

private:
  // Called with mutex_ held. Overwrites the export file with current averages.
  void writeReport()
  {
    std::ofstream f(export_path_);
    if (!f.is_open()) return;
    f << std::fixed << std::setprecision(3);
    for (const auto& [name, stat] : global_stats_)
    {
      f << std::left << std::setw(40) << name
        << stat.avg_ms << " ms"
        << "  (" << stat.count << " frames)\n";
    }
  }

  struct GlobalStat
  {
    double avg_ms = 0.0;
    int count     = 0;
  };

  std::string export_path_;

  // std::map keeps keys alphabetically sorted so related keys (lio/ekf,
  // lio/ekf/build_residuals, lio/ekf/solve) group together in output.
  std::map<std::string, double> frame_acc_;
  std::map<std::string, GlobalStat> global_stats_;
  mutable std::mutex mutex_;
};

using ProfilerPtr = std::shared_ptr<FrameProfiler>;

class TimedScope
{
  using Clock = std::chrono::steady_clock;

public:
  TimedScope(const ProfilerPtr& profiler, const std::string& name)
    : profiler_(profiler), name_(name), t0_(Clock::now()) {}

  ~TimedScope()
  {
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    profiler_->add(name_, ms);
  }

  TimedScope(const TimedScope&) = delete;
  TimedScope& operator=(const TimedScope&) = delete;

private:
  ProfilerPtr profiler_;
  std::string name_;
  Clock::time_point t0_;
};

}  // namespace livo_recon
