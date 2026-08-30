# livo_recon

A tightly-coupled LiDAR-Inertial-Visual odometry/reconstruction system for
ROS1 Noetic. LIO and VIO share a single EKF state; the map backend and the
visual point tracker are both pluggable.

- **Map backend**: a persistent per-voxel plane-fit map (`voxel`, PCA or
  debiased plane fitting) or an AKF-LIO-style Gaussian-summary map (`akf`),
  selected per-dataset via `voxel_map/backend` in the config.
- **Point tracker**: CoTracker, TAPNext, or TrackOn, run natively via
  LibTorch (optionally CUDA-accelerated, with AOT-compiled variants for
  the pair-tracking step). See `include/livo_recon/vio/point_tracker.h`.
- **Offline/cached tracking**: point tracks can be precomputed once with the
  `cache_tracker_output_livo_recon` tool and replayed from a cache file, so
  parameter sweeps over LIO/VIO fusion behavior don't have to re-run the
  (expensive) visual tracker every time.

## Datasets

Ships configs/launch files for:

| Dataset | Config | Launch |
|---|---|---|
| NTU-VIRAL | `config/ntu_viral.yaml` | `launch/livo_recon_ntu_viral.launch` |
| Hilti 2022 (handheld) | `config/slam_2022_handheld.yaml` | `launch/livo_recon_slam_2022.launch` |
| Hilti 2023 (handheld) | `config/slam_2023_handheld.yaml` | `launch/livo_recon_slam_2023.launch` |
| Hilti 2023 (robot) | `config/slam_2023_robot.yaml` | `launch/livo_recon_slam_2023_robot.launch` |
| Custom / Mid-360 | `config/mid360.yaml` | `launch/livo_recon_mid360.launch` |
| Custom / handheld | `config/hk.yaml` | `launch/livo_recon_hk.launch` |

Each dataset also ships a `*_no_overrides.yaml`, which is the config with
every dataset-specific override stripped back to defaults — useful as a
diff target when tuning.

## Build

Requires:

- Ubuntu 20.04 + ROS Noetic, built with `catkin_make` in a normal catkin
  workspace (this package assumes `livox_ros_driver` and `jsk_rviz_plugins`
  are also present in the workspace).
- CUDA (the CUDA kernels in `src/cuda/` are compiled for `sm_89` by
  default — edit `CMakeLists.txt`'s `TORCH_CUDA_ARCH_LIST` /
  `-arch=sm_89` if targeting a different GPU generation).
- [LibTorch](https://pytorch.org/get-started/locally/) (C++ distribution,
  CUDA build) — `find_package(Torch REQUIRED)` needs `CMAKE_PREFIX_PATH`
  pointed at it.
- PCL, OpenCV, Eigen3, OpenMP, TBB.

```bash
cd ~/catkin_ws
catkin_make --pkg livo_recon
```

## Model weights

`checkpoints/` is not included in this repository (traced/exported
LibTorch weights for CoTracker and TAPNext, several hundred MB to ~1 GB
each — too large for a git repo). To reproduce them:

- **CoTracker**: export from [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)'s
  pretrained checkpoint into the frame-encoder / pair-tracker `.pt`
  format `point_tracker.h` expects (see the export-script references in
  that header). AOT-compiled (`_aot_*x*.pt2`) and fp16 variants are
  optional performance variants of the same exported model.
- **TAPNext**: export from Google DeepMind's TAPNext checkpoint into the
  `tapnext_init.pt` / `tapnext_step.pt` pair `tapnext_backend.h` expects.

Point `include/livo_recon/vio/point_tracker.h`'s tracker-backend
selection at wherever you place the exported weights.

## Running

The launch files start the node itself (with `overrides_file`/`rviz` args);
play the corresponding rosbag separately:

```bash
roslaunch livo_recon livo_recon_ntu_viral.launch
# in another terminal
rosbag play /path/to/eee_01.bag --clock
```

See each dataset's config file for the full set of tunable LIO/VIO
parameters (extrinsics, noise models, tracker choice, map backend, cache
paths, evo scoring options).

## License

BSD-3-Clause — see `LICENSE`.
