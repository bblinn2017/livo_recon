#include "livo_recon/vio/tracker_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace livo_recon
{

namespace
{

template <typename T>
bool readPod(std::ifstream& f, T& v)
{
  f.read(reinterpret_cast<char*>(&v), sizeof(T));
  return static_cast<bool>(f);
}

bool readStr(std::ifstream& f, std::string& s)
{
  uint32_t len = 0;
  if (!readPod(f, len)) return false;
  s.resize(len);
  if (len > 0) f.read(&s[0], len);
  return static_cast<bool>(f);
}

bool readUvJacobian(std::ifstream& f, Eigen::Matrix2d& J)
{
  double m00 = 0, m01 = 0, m10 = 0, m11 = 0;
  if (!readPod(f, m00) || !readPod(f, m01) || !readPod(f, m10) || !readPod(f, m11)) return false;
  J << m00, m01, m10, m11;
  return true;
}

// Mirrors cache_tracker_output.cpp's writeTrackedFrame() field-for-field.
bool readTrackedFrame(std::ifstream& f, TrackedFrame& tf)
{
  if (!readPod(f, tf.timestamp)) return false;
  if (!readPod(f, tf.prev_timestamp)) return false;
  uint8_t ok = 0;
  if (!readPod(f, ok)) return false;
  tf.ok = (ok != 0);
  uint8_t reseeded = 0;
  if (!readPod(f, reseeded)) return false;
  tf.reseeded = (reseeded != 0);

  uint32_t n_uv = 0;
  if (!readPod(f, n_uv)) return false;
  tf.uv_curr.resize(n_uv);
  for (uint32_t i = 0; i < n_uv; ++i)
  {
    float x = 0.0f, y = 0.0f;
    if (!readPod(f, x) || !readPod(f, y)) return false;
    tf.uv_curr[i] = cv::Point2f(x, y);
  }
  tf.uv_curr_uv_jacobian.resize(n_uv);
  for (uint32_t i = 0; i < n_uv; ++i)
    if (!readUvJacobian(f, tf.uv_curr_uv_jacobian[i])) return false;
  tf.indices.resize(n_uv);
  for (uint32_t i = 0; i < n_uv; ++i)
    if (!readPod(f, tf.indices[i])) return false;

  uint32_t n_anchors = 0;
  if (!readPod(f, n_anchors)) return false;
  tf.anchors.resize(n_anchors);
  for (uint32_t i = 0; i < n_anchors; ++i)
  {
    float x = 0.0f, y = 0.0f;
    if (!readPod(f, x) || !readPod(f, y)) return false;
    tf.anchors[i].uv_seed = cv::Point2f(x, y);
    if (!readUvJacobian(f, tf.anchors[i].uv_jacobian_seed)) return false;
  }
  return true;
}

}  // namespace

bool TrackerCacheReader::load(const std::string& path, std::string& err)
{
  frames_.clear();
  header_ = TrackerCacheHeader();

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
  {
    err = "failed to open " + path;
    return false;
  }

  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, "RLTC", 4) != 0)
  {
    err = path + ": bad magic (not a .rlivtrackcache file)";
    return false;
  }
  uint32_t version = 0;
  if (!readPod(f, version) || version != 3)
  {
    err = path + ": unsupported version";
    return false;
  }
  if (!readPod(f, header_.grid_rows) || !readPod(f, header_.grid_cols))
  {
    err = path + ": truncated header (grid dims)";
    return false;
  }
  if (!readStr(f, header_.tracker_format) || !readStr(f, header_.bag_basename))
  {
    err = path + ": truncated header (strings)";
    return false;
  }
  if (!readPod(f, header_.n_frames))
  {
    err = path + ": truncated header (n_frames)";
    return false;
  }

  frames_.reserve(header_.n_frames);
  for (uint32_t i = 0; i < header_.n_frames; ++i)
  {
    TrackedFrame tf;
    if (!readTrackedFrame(f, tf))
    {
      std::ostringstream oss;
      oss << path << ": truncated at frame " << i << "/" << header_.n_frames;
      err = oss.str();
      frames_.clear();
      return false;
    }
    frames_.push_back(std::move(tf));
  }
  return true;
}

size_t TrackerCacheReader::findByTimestamp(double ts) const
{
  // frames_ is strictly increasing in timestamp (built from one forward
  // pass over the bag) -- lower_bound then check the one candidate it
  // lands on (and, defensively, its immediate predecessor, in case of a
  // hairline floating-point ordering edge) within tolerance.
  auto it = std::lower_bound(frames_.begin(), frames_.end(), ts,
      [](const TrackedFrame& f, double t) { return f.timestamp < t - 1e-6; });
  if (it != frames_.end() && std::fabs(it->timestamp - ts) < 1e-6)
    return static_cast<size_t>(it - frames_.begin());
  if (it != frames_.begin())
  {
    auto prev = it - 1;
    if (std::fabs(prev->timestamp - ts) < 1e-6)
      return static_cast<size_t>(prev - frames_.begin());
  }
  return frames_.size();
}

}  // namespace livo_recon
