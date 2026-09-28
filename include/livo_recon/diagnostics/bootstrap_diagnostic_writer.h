#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

// Deterministic order-sensitive fingerprint over one flattened prepared
// bootstrap population (point + raw_body_point + sensor_cov). Order-
// sensitive on purpose: aggregate and sequential modes must receive this
// exact same flattened sequence per the calibration-owned bootstrap design's
// controlled-comparison invariant, so any reordering is itself a defect the
// hash should catch, not something to normalize away.
uint64_t hashBootstrapPopulation(const std::vector<PointXYZCov>& flattened);

// Written once, immediately after CalibProc::prepareBootstrapObservations()
// finishes, before VoxelMap::bootstrap() is ever called. Reports the raw and
// prepared/downsampled point count of every retained stationary calibration
// observation plus a hash of the flattened prepared population, so aggregate
// vs. sequential runs can be confirmed to have received an identical
// prepared population before any map-structure comparison is trusted.
void writeBootstrapPreparationDiagnostic(
    const std::vector<std::vector<PointXYZT>>& raw_observations,
    const std::vector<std::vector<PointXYZCov>>& prepared_observations);

// Written once, from inside VoxelMap::bootstrap(), bracketing the actual
// aggregate/sequential insertion. state_before/state_after are snapshots of
// the SAME StateGroup taken immediately before and after insertion -- since
// this bootstrap design never mutates state/P0, before/after should always
// report identical values; this file exists to make that an observed fact
// per run rather than an assumption.
void writeBootstrapMapSeedDiagnostic(
    const std::string& mode,
    int frame_idx_before, int frame_idx_after,
    const StateGroup& state_before, const StateGroup& state_after,
    uint64_t flattened_population_hash, size_t flattened_population_count);

}  // namespace livo_recon
