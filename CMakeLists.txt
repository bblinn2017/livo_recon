cmake_minimum_required(VERSION 3.16)

project(livo_recon)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)

# Restrict host-only flags to CXX so nvcc doesn't receive them directly
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-march=native>
  $<$<COMPILE_LANGUAGE:CXX>:-g>
)

# Match system PCL's Eigen alignment config. libpcl_filters.so allocates
# pcl::PointCloud::points with posix_memalign (16-byte path), but -march=native
# can raise EIGEN_MAX_ALIGN_BYTES to 32/64 which switches our deallocation to
# handmade_aligned_free — an incompatible scheme that reads a header prefix that
# posix_memalign never wrote, causing a heap-buffer-overflow on free.
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=16)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf
  pcl_ros
  cv_bridge
  image_transport
  message_generation
  livox_ros_driver
  jsk_rviz_plugins
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)  # provides ${CUDA_LIBRARIES} (includes libcudart)

# Skip Torch's own try_run-based GPU arch auto-detection (it fails under this
# CMake/CUDA setup with a "requires CUDA17 dialect" error) — declare the
# architecture explicitly (matches livo_cuda's -arch=sm_89 below).
set(TORCH_CUDA_ARCH_LIST "8.9")

find_package(Torch REQUIRED)  # LibTorch, for running CoTracker natively

# Torch's config sets CMAKE_CUDA_STANDARD=17, a dialect this CMake (3.16)
# doesn't know how to generate flags for. Reset it for our own CUDA target.
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf
    pcl_ros
    cv_bridge
    image_transport
    livox_ros_driver
    jsk_rviz_plugins
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

add_library(livo_cuda STATIC
  src/cuda/lio_kernels.cu
)

set_target_properties(livo_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_STANDARD 14
  CUDA_STANDARD_REQUIRED OFF
)

target_include_directories(livo_cuda PRIVATE
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(livo_cuda PRIVATE
  # -arch=sm_89 instead of CMAKE_CUDA_ARCHITECTURES (requires CMake 3.18)
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_89 --use_fast_math -O3 -lineinfo>
)

add_executable(livo_recon_node
  src/utils/state/state.cpp
  src/utils/algo/covariance_math.cpp
  src/utils/algo/chi_square.cpp
  src/utils/lidar/lidar.cpp
  src/utils/data/data_queues.cpp
  src/utils/data/measures.cpp
  src/diagnostics/log/debug_log_dir.cpp
  src/diagnostics/log/consistency_log.cpp
  src/diagnostics/eval/nees_logger.cpp
  src/diagnostics/gt_interp.cpp
  src/diagnostics/stationary_diagnostic_writer.cpp
  src/diagnostics/state_trace.cpp
  src/diagnostics/calibration_p0_writer.cpp
  src/diagnostics/motion_q_writer.cpp
  src/diagnostics/r66_writers.cpp
  src/diagnostics/bootstrap_diagnostic_writer.cpp
  src/main.cpp
  src/livo_recon_node.cpp
  src/node_context.cpp
  src/processing/cbk_processing.cpp
  src/processing/pub_processing.cpp
  src/processing/calib_processing.cpp
  src/processing/imu_processing.cpp
  src/processing/vio_processing.cpp
  src/processing/lio_base.cpp
  src/processing/lio_iekf.cpp
  src/processing/lio_iekf_shadow.cpp
  src/lio/shadow_lio.cpp
  src/processing/combined_processing.cpp
  src/processing/evo_processing.cpp
  src/lio/voxelplane.cpp
  src/lio/voxelnode.cpp
  src/lio/lio_accumulator.cpp
  src/lio/deskew.cpp
  src/lio/residual_redundancy.cpp
  src/lio/info_scale.cpp
  src/map/voxelmap.cpp
  src/map/akf_map.cpp
  src/map/surface_map.cpp
  src/map/binned_voxelmap.cpp
  src/map/map_backend_factory.cpp
  src/vio/crop_utils.cpp
  src/vio/point_tracker.cpp
  src/vio/tracker.cpp
  src/vio/tracker_cache.cpp
  src/vio/vio_accumulator.cpp
  src/vio/cotracker/cotracker_backend.cpp
  src/vio/tapnext/tapnext_backend.cpp
  src/vio/trackon/trackon_backend.cpp
)

target_compile_options(livo_recon_node PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>
)

target_link_libraries(livo_recon_node
  livo_cuda
  surface_core
  ${CUDA_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  OpenMP::OpenMP_CXX
  tbb
)

# Standalone tracker-output cache-generation tool -- see src/tools/
# cache_tracker_output.cpp's header comment for the full rationale/format.
# Shares every tracker-related source with livo_recon_node except the
# node's own main.cpp/livo_recon_node.cpp/processing/lio/map files (this
# tool drives Tracker directly, no LIO/VIO/publishing machinery needed).
# Target named cache_tracker_output_livo_recon, NOT cache_tracker_output --
# FAST-LIVO2/CMakeLists.txt already defines an executable named
# cache_tracker_output (task #199, same catkin workspace), and catkin_make
# builds one top-level CMake project spanning every package, so executable
# target names must be unique workspace-wide, not just within a package
# (confirmed: plain "cache_tracker_output" here failed CMake configure with
# "add_executable cannot create target ... already exists").
add_executable(cache_tracker_output_livo_recon
  src/utils/state/state.cpp
  src/tools/cache_tracker_output.cpp
  src/vio/crop_utils.cpp
  src/vio/point_tracker.cpp
  src/vio/tracker.cpp
  src/vio/tracker_cache.cpp
  src/vio/cotracker/cotracker_backend.cpp
  src/vio/tapnext/tapnext_backend.cpp
  src/vio/trackon/trackon_backend.cpp
)

target_compile_options(cache_tracker_output_livo_recon PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS}>
)

target_link_libraries(cache_tracker_output_livo_recon
  livo_cuda
  ${CUDA_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  OpenMP::OpenMP_CXX
  tbb
)

# Unit tests.
add_subdirectory(tests/active)
add_subdirectory(tests/reference_math)

# R66: the R52 stationary-map research library, its CLI and its test were removed (harness parked and retired by Bryce).

# Surface map core (Eigen + std only): additive cell statistics and the dirty-set merge/split layer used by the
# "surface" MapBackend (src/map/surface_map.cpp). Kept ROS-free so it can be unit-tested on its own.
add_library(surface_core
  src/map/surface/surface_core.cpp
)
target_include_directories(surface_core PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(surface_core Eigen3::Eigen)
add_executable(test_surface_core tests/test_surface_core.cpp)
target_link_libraries(test_surface_core surface_core)
add_test(NAME surface_core COMMAND test_surface_core)

# R63 (coding agent addition, not part of the planning agent's patch): StateGroup's s2 gravity
# model, exercised ROS-free via #define private public (no ros::NodeHandle/loadParameters() path
# needed for these three checks). state.cpp needs roscpp/xmlrpcpp for state.h's ros::NodeHandle/
# XmlRpcValue types even though this test never calls ros::init() or touches a live master.
add_library(state_lib
  src/utils/state/state.cpp
)
target_include_directories(state_lib PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(state_lib Eigen3::Eigen ${catkin_LIBRARIES})
add_executable(test_gravity_s2 tests/test_gravity_s2.cpp)
target_link_libraries(test_gravity_s2 state_lib)
add_test(NAME gravity_s2 COMMAND test_gravity_s2)

# R64 (planning agent's patch, section 4 of R64_INSTRUCTIONS.md): debiased plane-fit variants
# unit test.
add_executable(test_debias_modes tests/test_debias_modes.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_debias_modes PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_debias_modes ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME debias_modes COMMAND test_debias_modes)

# R65 (planning agent's patch, section 4 of R65_INSTRUCTIONS.md): per-point pose covariance and
# voxel-history unit tests.
add_executable(test_point_cov tests/test_point_cov.cpp)
target_link_libraries(test_point_cov state_lib)
add_test(NAME point_cov COMMAND test_point_cov)

add_executable(test_r83b_instrumentation tests/test_r83b_instrumentation.cpp src/lio/voxelplane.cpp
               src/diagnostics/log/debug_log_dir.cpp src/diagnostics/motion_q_writer.cpp)
target_include_directories(test_r83b_instrumentation PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_r83b_instrumentation ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME r83b_instrumentation COMMAND test_r83b_instrumentation)

add_executable(test_voxel_history tests/test_voxel_history.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_voxel_history PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_voxel_history ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME voxel_history COMMAND test_voxel_history)

# R74 (F-419 extension): plane covariance exactly symmetric (same sources as test_voxel_history).
add_executable(test_plane_var_symmetry tests/test_plane_var_symmetry.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_plane_var_symmetry PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_plane_var_symmetry ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME plane_var_symmetry COMMAND test_plane_var_symmetry)
add_executable(test_line_residual_glue tests/test_line_residual_glue.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_line_residual_glue PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_line_residual_glue ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME line_residual_glue COMMAND test_line_residual_glue)
# R77 (F-453, F-433): line variance tilt term and plane eigengap floor (same sources as test_line_residual_glue).
add_executable(test_line_variance_r77 tests/test_line_variance_r77.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_line_variance_r77 PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_line_variance_r77 ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME line_variance_r77 COMMAND test_line_variance_r77)
# R80 (F-453): rows = distance for hard line voxels (same sources as test_line_variance_r77; the D4 source-contract check reads src/map/voxelmap.cpp).
add_executable(test_line_distance_r80 tests/test_line_distance_r80.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_line_distance_r80 PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_compile_definitions(test_line_distance_r80 PRIVATE LIVO_R80_VOXELMAP_SRC="${CMAKE_CURRENT_SOURCE_DIR}/src/map/voxelmap.cpp")
target_link_libraries(test_line_distance_r80 ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME line_distance_r80 COMMAND test_line_distance_r80)
# R81 (F-468, F-485 Part 3): plane_var_scan_s2 / plane_var_min (pure arithmetic + VoxelPlane::gate() / computeResidual(); the source check reads src/map/voxelmap.cpp).
add_executable(test_plane_var_r81 tests/test_plane_var_r81.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_plane_var_r81 PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_compile_definitions(test_plane_var_r81 PRIVATE LIVO_R81_VOXELMAP_SRC="${CMAKE_CURRENT_SOURCE_DIR}/src/map/voxelmap.cpp")
target_link_libraries(test_plane_var_r81 ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME plane_var_r81 COMMAND test_plane_var_r81)

# R78 (planning agent's patch): which point set the IEKF uses (lio/iekf_point_stream); header only, ROS-free.
add_executable(test_iekf_stream_r78 tests/test_iekf_stream_r78.cpp)
target_include_directories(test_iekf_stream_r78 PUBLIC include)
add_test(NAME iekf_stream_r78 COMMAND test_iekf_stream_r78)

# R79 (planning agent's patch): which frames write a voxel-map snapshot (voxel_map/map/snapshot_frames); header only, ROS-free.
add_executable(test_snapshot_frames_r79 tests/test_snapshot_frames_r79.cpp)
target_include_directories(test_snapshot_frames_r79 PUBLIC include)
add_test(NAME snapshot_frames_r79 COMMAND test_snapshot_frames_r79)

# R80: per-message LiDAR log helpers + pushLidar overlap rule (header-only, ROS-free)
add_executable(test_lidar_msg_log_r80 tests/test_lidar_msg_log_r80.cpp)
target_include_directories(test_lidar_msg_log_r80 PUBLIC include)
add_test(NAME lidar_msg_log_r80 COMMAND test_lidar_msg_log_r80)

# R80C (F-484): retime grid time and retime_trim point count (header-only, ROS-free)
add_executable(test_lidar_retime_r80c tests/test_lidar_retime_r80c.cpp)
target_include_directories(test_lidar_retime_r80c PUBLIC include)
add_test(NAME lidar_retime_r80c COMMAND test_lidar_retime_r80c)
# R86A (F-532): second-stream retime_trim (header-only, ROS-free)
add_executable(test_dry_run_trim_r86a tests/test_dry_run_trim_r86a.cpp)
target_include_directories(test_dry_run_trim_r86a PUBLIC include)
add_test(NAME dry_run_trim_r86a COMMAND test_dry_run_trim_r86a)
add_executable(test_point_time_r82 tests/test_point_time_r82.cpp)
target_include_directories(test_point_time_r82 PUBLIC include)
add_test(NAME point_time_r82 COMMAND test_point_time_r82)
add_executable(test_first_pose_align_r82 tests/test_first_pose_align_r82.cpp)
target_include_directories(test_first_pose_align_r82 PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_first_pose_align_r82 Eigen3::Eigen)
add_test(NAME first_pose_align_r82 COMMAND test_first_pose_align_r82)
add_executable(test_plane_var_diag_r82 tests/test_plane_var_diag_r82.cpp)
target_include_directories(test_plane_var_diag_r82 PUBLIC include)
add_test(NAME plane_var_diag_r82 COMMAND test_plane_var_diag_r82)
add_executable(test_line_tilt_r82 tests/test_line_tilt_r82.cpp)
target_include_directories(test_line_tilt_r82 PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_line_tilt_r82 Eigen3::Eigen)
add_test(NAME line_tilt_r82 COMMAND test_line_tilt_r82)
# R82 (D4): drives VoxelPlane directly (same source set as test_line_variance_r77).
add_executable(test_debiased_line_r82 tests/test_debiased_line_r82.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_debiased_line_r82 PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_debiased_line_r82 Eigen3::Eigen ${OpenCV_LIBS})
add_test(NAME debiased_line_r82 COMMAND test_debiased_line_r82)
add_executable(test_voxel_class_r82 tests/test_voxel_class_r82.cpp)
target_include_directories(test_voxel_class_r82 PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_voxel_class_r82 Eigen3::Eigen)
add_test(NAME voxel_class_r82 COMMAND test_voxel_class_r82)
add_executable(test_lidar_offset_r80d tests/test_lidar_offset_r80d.cpp)
target_include_directories(test_lidar_offset_r80d PUBLIC include)
add_test(NAME lidar_offset_r80d COMMAND test_lidar_offset_r80d)

# R70 (planning agent's patch): block information scale identities (ROS-free, Eigen only).
add_executable(test_info_scale tests/test_info_scale.cpp src/lio/info_scale.cpp)
target_include_directories(test_info_scale PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_info_scale Eigen3::Eigen)
add_test(NAME info_scale COMMAND test_info_scale)

# R72 (planning agent's patch; R82: step cap F-403 retired): Q-adaptation alternatives (F-401), failure dump round trip (F-398).
# ROS-free header-only modules, Eigen only.
foreach(r72_test q_adapt failure_capture symmetrize renamed_keys shadow_options bag_clock row_flush plane_test_values line_residual line_residual_r75 line_residual_r76 process_noise_switch)
  add_executable(test_${r72_test} tests/test_${r72_test}.cpp)
  target_include_directories(test_${r72_test} PUBLIC include ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(test_${r72_test} Eigen3::Eigen)
  add_test(NAME ${r72_test} COMMAND test_${r72_test})
endforeach()
add_executable(test_dump_rules_r86b tests/test_dump_rules_r86b.cpp)
target_include_directories(test_dump_rules_r86b PUBLIC include)
add_test(NAME dump_rules_r86b COMMAND test_dump_rules_r86b)
add_executable(test_admission_rule_r87 tests/test_admission_rule_r87.cpp)
target_include_directories(test_admission_rule_r87 PUBLIC include)
add_test(NAME admission_rule_r87 COMMAND test_admission_rule_r87)

# R88: the bin accumulator of the "binned" map type (header-only, Eigen + std only).
add_executable(test_binned_bins_r88 tests/test_binned_bins_r88.cpp)
target_include_directories(test_binned_bins_r88 PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_binned_bins_r88 Eigen3::Eigen)
add_test(NAME binned_bins_r88 COMMAND test_binned_bins_r88)

# R89 (planning agent's patch): exact-zero known blocks of P0 (header-only, Eigen only).
add_executable(test_p0_pin_r89 tests/test_p0_pin_r89.cpp)
target_include_directories(test_p0_pin_r89 PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_p0_pin_r89 Eigen3::Eigen)
add_test(NAME p0_pin_r89 COMMAND test_p0_pin_r89)
# R88: VoxelPlane::refitFromBins() and VoxelNode's binned debiased branch (same source set as test_plane_var_r81).
add_executable(test_binned_node_r88 tests/test_binned_node_r88.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_binned_node_r88 PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_binned_node_r88 ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME binned_node_r88 COMMAND test_binned_node_r88)
