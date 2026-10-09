// R82 (F-504): unit test of the per-point time read (header only, ROS-free). Algebraic identities and refusals only (rule 3.1). Planner sandbox, not evidence.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "livo_recon/utils/lidar/point_time.h"

using namespace livo_recon;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

// A point record of 24 bytes: x y z f32 (12), a pad (4), the time field at byte 16 (u32) or 16 (f64, 8 bytes).
static std::vector<uint8_t> buildF64(const std::vector<double>& v, size_t step, size_t off)
{ std::vector<uint8_t> d(v.size() * step, 0); for (size_t i = 0; i < v.size(); ++i) std::memcpy(&d[i * step + off], &v[i], 8); return d; }
static std::vector<uint8_t> buildU32(const std::vector<uint32_t>& v, size_t step, size_t off)
{ std::vector<uint8_t> d(v.size() * step, 0); for (size_t i = 0; i < v.size(); ++i) std::memcpy(&d[i * step + off], &v[i], 4); return d; }

template <class F> static bool throws(F f) { try { f(); } catch (const std::runtime_error&) { return true; } return false; }

int main()
{
  std::vector<double> offs; std::vector<char> keep; PointTimeStats st;
  const double hdr = 1609059013.5;
  // Absolute mode, header ref: offset = value - header; tolerance 1e-6 s because one double step at 1.6e9 s is 2.4e-7 s.
  const std::vector<double> v = {hdr, hdr + 0.0125, hdr + 0.05, hdr + 0.099};
  const auto d64 = buildF64(v, 32, 16);
  PointTimeSpec a; a.field = "timestamp"; a.mode = PointTimeSpec::Mode::AbsoluteSecF64; a.explicit_spec = true;
  readPointTimeOffsets(d64.data(), v.size(), 32, 16, kPointFieldFloat64, a, hdr, offs, keep, st);
  CHECK(offs[0] == 0.0 && std::fabs(offs[1] - 0.0125) < 1e-6 && std::fabs(offs[3] - 0.099) < 1e-6);
  CHECK(st.n_points == 4 && st.n_nonfinite == 0 && st.field_found && offs.size() == 4);
  CHECK(st.offset_min == 0.0 && std::fabs(st.offset_max - 0.099) < 1e-6);
  // Header reference moves every offset by the same constant (the header-minus-first-point difference), nothing else.
  readPointTimeOffsets(d64.data(), v.size(), 32, 16, kPointFieldFloat64, a, hdr - 0.004, offs, keep, st);
  CHECK(std::fabs(offs[0] - 0.004) < 1e-6 && std::fabs(offs[2] - offs[0] - 0.05) < 1e-6);
  // min_point reference: the smallest finite value is offset 0, whatever the header says.
  PointTimeSpec m = a; m.ref = PointTimeSpec::Ref::MinPoint;
  readPointTimeOffsets(d64.data(), v.size(), 32, 16, kPointFieldFloat64, m, hdr + 123.0, offs, keep, st);
  CHECK(offs[0] == 0.0 && st.offset_min == 0.0);
  // A non-finite value is dropped (keep 0), counted, and does not enter the min / max.
  std::vector<double> vn = v; vn[2] = std::nan("");
  const auto dn = buildF64(vn, 32, 16);
  readPointTimeOffsets(dn.data(), vn.size(), 32, 16, kPointFieldFloat64, m, hdr, offs, keep, st);
  CHECK(keep[2] == 0 && keep[0] == 1 && st.n_nonfinite == 1 && std::fabs(st.offset_max - 0.099) < 1e-6);
  // Default spec on a u32 "t": nanoseconds * 1e-9, bit for bit with the R81 expression.
  const std::vector<uint32_t> tn = {0u, 12500000u, 99999999u};
  const auto du = buildU32(tn, 32, 16);
  PointTimeSpec def;
  readPointTimeOffsets(du.data(), tn.size(), 32, 16, kPointFieldUint32, def, hdr, offs, keep, st);
  CHECK(offs[1] == static_cast<double>(12500000u) * 1e-9 && offs[2] == static_cast<double>(99999999u) * 1e-9);
  // Default spec, field absent: zero offsets, no error (the R81 fall back); explicit spec, field absent: error.
  readPointTimeOffsets(du.data(), tn.size(), 32, -1, 0, def, hdr, offs, keep, st);
  CHECK(!st.field_found && offs[0] == 0.0 && offs[2] == 0.0 && keep[1] == 1);
  PointTimeSpec ex = def; ex.explicit_spec = true;
  CHECK(throws([&] { readPointTimeOffsets(du.data(), tn.size(), 32, -1, 0, ex, hdr, offs, keep, st); }));
  // Default spec reads a field of any declared datatype as u32 (R81); an explicit spec refuses a datatype that does not match the mode.
  readPointTimeOffsets(du.data(), tn.size(), 32, 16, kPointFieldFloat64, def, hdr, offs, keep, st);
  CHECK(st.field_found);
  CHECK(throws([&] { readPointTimeOffsets(du.data(), tn.size(), 32, 16, kPointFieldFloat64, ex, hdr, offs, keep, st); }));
  CHECK(throws([&] { readPointTimeOffsets(d64.data(), v.size(), 32, 16, kPointFieldUint32, a, hdr, offs, keep, st); }));
  // R83: the datatype codes are the sensor_msgs::PointField ones (UINT32 = 6, FLOAT64 = 8); a FLOAT32 (7) field is refused in absolute mode and the message names FLOAT64 (8), not (7) as R82 did.
  CHECK(kPointFieldUint32 == 6 && kPointFieldFloat64 == 8);
  {
    std::string what;
    try { readPointTimeOffsets(d64.data(), v.size(), 32, 16, 7, a, hdr, offs, keep, st); } catch (const std::runtime_error& e) { what = e.what(); }
    CHECK(what.find("FLOAT64 (8)") != std::string::npos && what.find("FLOAT64 (7)") == std::string::npos);
  }
  // An empty message is not an error.
  readPointTimeOffsets(nullptr, 0, 32, 16, kPointFieldFloat64, a, hdr, offs, keep, st);
  CHECK(offs.empty() && st.n_points == 0 && st.offset_min == 0.0);
  std::printf(g_fail ? "FAILED %d\n" : "ALL PASS\n", g_fail);
  return g_fail ? 1 : 0;
}
