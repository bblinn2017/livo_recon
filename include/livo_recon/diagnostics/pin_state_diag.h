#pragma once
// TEMPORARY diagnostic (2026-10-27, apples-to-apples ask-comparison
// investigation). Overwrites the live EKF state from one row of a
// REFERENCE run's state_trace.csv, at a configured scan_id, right before
// that frame's IMU propagation runs -- see livo_recon_node.h's
// pin_state_before_scan_ doc comment for the motivation (pin sequential's
// and combined's trajectories identical up to the pivot/motion frame, so
// their first-motion-frame "ask" diagnostics are no longer confounded by
// the two modes' own, independently-diverged, trajectories).
//
// Delete this file + its call site (livo_recon_node.cpp's estimateState())
// + the diag/pin_state_before_scan and diag/pin_state_file options once
// the investigation is done.
#include <string>

#include "livo_recon/utils/state/state.h"

namespace livo_recon
{

// Reads state_trace.csv at `csv_path`, finds the row matching `scan_id`
// and `phase` exactly (e.g. "post_lio"), and overwrites rot/pos/vel/
// bias_gyr/bias_acc/gravity/cov on `state` from it. Honors the row's own
// `dim` column (15 when gravity isn't a live EKF state, 18 when it is) --
// refuses (returns false, state untouched) if `dim` doesn't match
// state.dimState(), rather than silently loading a mismatched-size
// covariance. Returns false (state untouched) if the file can't be read
// or no matching row is found.
bool loadPinState(const std::string& csv_path, int scan_id, const std::string& phase, StateGroup& state);

}  // namespace livo_recon
