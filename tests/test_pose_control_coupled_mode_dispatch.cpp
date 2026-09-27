#include "livo_recon/processing/pose_control_coupled_modes.h"

#include <cassert>
#include <string>

using namespace livo_recon;

int main()
{
  struct Expected { const char* name; CoupledModeFamily family; bool physical; };
  const Expected cases[] = {
    {"local_spline", CoupledModeFamily::LocalSpline, false},
    {"single_tail", CoupledModeFamily::SingleTail, false},
    {"direct_lidar_imu", CoupledModeFamily::DirectLidarImu, false},
    {"covariance_all_knots", CoupledModeFamily::CovarianceAllKnots, false},
    {"physical_rpv_local_spline", CoupledModeFamily::LocalSpline, true},
    {"physical_rpv_single_tail", CoupledModeFamily::SingleTail, true},
    {"physical_rpv_direct_lidar_imu", CoupledModeFamily::DirectLidarImu, true},
    {"physical_rpv_covariance_all_knots", CoupledModeFamily::CovarianceAllKnots, true},
    {"physical_rpv_tail", CoupledModeFamily::SingleTail, true},
    {"physical_rpv_all_knots", CoupledModeFamily::CovarianceAllKnots, true},
  };
  for (const auto& expected : cases) {
    CoupledModeSelection actual;
    assert(parseCoupledMode(expected.name, actual));
    assert(actual.family == expected.family);
    assert(actual.physical_rpv == expected.physical);
    assert(std::string(coupledModeFamilyName(actual.family)).size() > 0);
  }
  CoupledModeSelection invalid;
  assert(!parseCoupledMode("physical_rpv_not_a_mode", invalid));
  return 0;
}
