#pragma once

#include <Eigen/Dense>
#include <vector>

// ============================================================================
// 2026-09-23: surface-agnostic directional information analysis for the
// pose-control LiDAR factor (items 26-30 of the implementation spec).
//
// THIS IS NOT residual_redundancy.h's correction. residual_redundancy.h
// (mode=="woodbury"/"woodbury_rescale"/"woodbury_directional") models a
// SPECIFIC shared-noise source (points matched to the same mapped plane
// sharing that plane's own fit-uncertainty draw, via plane_id grouping) and
// CORRECTS the information those residuals contribute. This file answers a
// different question with no correction: "which REDUCED-STATE directions
// does this scan's LiDAR geometry actually constrain, and how strongly,
// regardless of whether the redundant residuals happen to share a mapped
// plane's ID or not" -- e.g. many points on the SAME real physical surface
// but assigned to different voxel/plane fits (no shared plane_id) still
// geometrically redundant in exactly the same directional sense, which
// plane_id-based correlation correction cannot see at all (item 28/32).
//
// METHOD (item 26/27): each LiDAR residual's whitened Jacobian row is
// h_i = H_i / sigma_i, H_i its (dense, reduced-state) Jacobian row. Rather
// than ever materializing the M x d whitened matrix or an M x M Gram matrix
// (item 27: "Do NOT create M x M"), this reuses the d x d information
// matrix the LiDAR factor ALREADY accumulates one residual at a time,
//     Lambda = sum_i h_i h_i^T = sum_i (1/sigma_i^2) H_i H_i^T,
// which is exactly addPoseControlLidarFactor()'s own A_lidar_raw (projected
// to z-space as Lambda_meas_z in lio_coupled.cpp) -- an O(M*d^2) streaming
// accumulation, never O(M^2) or M x M (item 32). This file's own
// contribution is analyzeDirectionalRedundancy(): the d x d EIGEN-SPECTRUM
// of that already-accumulated matrix -- eigenvalues (information, not
// covariance -- large eigenvalue = well-observed direction), effective
// rank, condition number, dominant/weak eigenvector directions, and the
// cumulative information fraction curve. This is mathematically the
// correct object for the question asked: Lambda's eigenvectors ARE the
// state-space directions LiDAR geometry constrains, and Lambda = sum h_i
// h_i^T is precisely "how much of that direction's information got
// contributed, and by how many effectively-independent whitened rows" --
// many repeated/near-parallel h_i (redundant directional observations,
// e.g. a large flat wall) inflate ONE eigenvalue without adding rank; truly
// independent geometry (corner returns, multiple wall orientations) adds
// eigenvalues in ADDITIONAL directions.
// ============================================================================

namespace livo_recon
{

struct DirectionalRedundancyStats
{
  int raw_residual_count = 0;   // M -- number of whitened rows h_i that fed Lambda
  int reduced_state_dimension = 0;  // d -- Lambda's own dimension
  int effective_rank = 0;       // eigenvalues > rel_rank_thresh * lambda_max
  double condition_number = 0.0;    // lambda_max / max(lambda_min_nonzero, eps)
  // Descending eigenvalues/cumulative information fraction -- size d each.
  Eigen::VectorXd eigenvalues;
  Eigen::VectorXd cumulative_information_fraction;
  // Column j = eigenvector for eigenvalues(j) (same descending order).
  Eigen::MatrixXd eigenvectors;
  bool valid = false;
};

// Lambda must be symmetric PSD (an information matrix accumulated as
// sum h_i h_i^T, as above) and raw_residual_count is the M this Lambda was
// built from (for reporting only -- not used in the eigendecomposition
// itself, which only ever touches the d x d Lambda).
DirectionalRedundancyStats analyzeDirectionalRedundancy(
    const Eigen::MatrixXd& Lambda, int raw_residual_count, double rel_rank_thresh = 1e-6);

}  // namespace livo_recon
