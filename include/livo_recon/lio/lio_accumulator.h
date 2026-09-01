#pragma once

#include <vector>

#include "livo_recon/utils/map/voxelmap_utils.h"
#include "livo_recon/utils/algo/ekf.h"
#include "livo_recon/cuda/lio_cuda.h"

// History (9-16): see docs/livo_recon_changelog.md#include-livo_recon-lio-lio_accumulator.h-9
namespace livo_recon
{

// History (20-32): see docs/livo_recon_changelog.md#include-livo_recon-lio-lio_accumulator.h-20
void accumulateLioResiduals(const std::vector<Residual>& residuals, EkfUpdate& out);

// CUDA accumulation: same contract, using the H^T*W*H reduction kernel via
// runLioAccumulation(). `buf` is caller-owned (LioProc keeps one persistent
// LioCudaBuffers across frames to amortize allocation).
void accumulateLioResidualsCuda(const std::vector<Residual>& residuals, EkfUpdate& out,
                                 LioCudaBuffers& buf);

}  // namespace livo_recon
