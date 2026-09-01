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
  jsk_rviz_plugins  # OverlayText, for publishEvoStats' rviz table (see pub_processing.cpp)
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
  src/utils/lidar/lidar.cpp
  src/utils/data/data_queues.cpp
  src/utils/data/measures.cpp
  src/utils/log/debug_log_dir.cpp
  src/utils/log/consistency_log.cpp
  src/main.cpp
  src/livo_recon_node.cpp
  src/node_context.cpp
  src/processing/cbk_processing.cpp
  src/processing/pub_processing.cpp
  src/processing/calib_processing.cpp
  src/processing/imu_processing.cpp
  src/processing/vio_processing.cpp
  src/processing/lio_processing.cpp
  src/processing/combined_processing.cpp
  src/processing/evo_processing.cpp
  src/lio/voxelplane.cpp
  src/lio/voxelnode.cpp
  src/lio/lio_accumulator.cpp
  src/lio/deskew.cpp
  src/lio/spline.cpp
  src/lio/adaptive_q.cpp
  src/map/voxelmap.cpp
  src/map/akf_map.cpp
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