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
  src/diagnostics/init_consistency.cpp
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
  src/processing/lio_coupled.cpp
  src/processing/lio_coupled_process.cpp
  src/processing/lio_coupled_update_log.cpp
  src/processing/combined_processing.cpp
  src/processing/evo_processing.cpp
  src/lio/voxelplane.cpp
  src/lio/voxelnode.cpp
  src/lio/lio_accumulator.cpp
  src/lio/deskew.cpp
  src/lio/residual_redundancy.cpp
  src/lio/info_scale.cpp
  src/lio/joint_knot_estimator.cpp
  src/diagnostics/joint_knots/joint_knot_diagnostic_writer.cpp
  src/map/voxelmap.cpp
  src/map/akf_map.cpp
  src/map/surface_map.cpp
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

# Canonical coupled joint-knot tests.
add_subdirectory(tests/active)
add_subdirectory(tests/joint_knots)
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

add_executable(test_voxel_history tests/test_voxel_history.cpp src/lio/voxelnode.cpp src/lio/voxelplane.cpp src/diagnostics/log/debug_log_dir.cpp)
target_include_directories(test_voxel_history PUBLIC include ${EIGEN3_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
target_link_libraries(test_voxel_history ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME voxel_history COMMAND test_voxel_history)

# R70 (planning agent's patch): block information scale identities (ROS-free, Eigen only).
add_executable(test_info_scale tests/test_info_scale.cpp src/lio/info_scale.cpp)
target_include_directories(test_info_scale PUBLIC include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_info_scale Eigen3::Eigen)
add_test(NAME info_scale COMMAND test_info_scale)

# R72 (planning agent's patch): step cap (F-403), Q-adaptation alternatives (F-401), ZUPT (F-401), failure dump round trip (F-398).
# ROS-free header-only modules, Eigen only. (The solver-level failure-capture test, test_joint_knot_failure_capture, is picked up by
# tests/joint_knots/CMakeLists.txt's glob.)
foreach(r72_test step_cap q_adapt zupt failure_capture symmetrize renamed_keys shadow_options)
  add_executable(test_${r72_test} tests/test_${r72_test}.cpp)
  target_include_directories(test_${r72_test} PUBLIC include ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(test_${r72_test} Eigen3::Eigen)
  add_test(NAME ${r72_test} COMMAND test_${r72_test})
endforeach()
