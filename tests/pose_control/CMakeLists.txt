# Pose-control unit-test build infrastructure.
# Common production sources are compiled once into pose_control_test_core.
# New root-level test_pose_control_*.cpp files are discovered automatically.

set(POSE_CONTROL_TEST_CORE_SOURCES
  ${PROJECT_SOURCE_DIR}/src/utils/state/state.cpp
  ${PROJECT_SOURCE_DIR}/src/diagnostics/log/debug_log_dir.cpp
  ${PROJECT_SOURCE_DIR}/src/diagnostics/pose_control/pose_control_physical_diagnostics.cpp
  ${PROJECT_SOURCE_DIR}/src/diagnostics/pose_control/pose_control_gt_diagnostics.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/spline.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/adaptive_q.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/pose_control_adaptive_q.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/pose_control_directional_redundancy.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/pose_control_spline.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/pose_control_imu_prior_builder.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/pose_control_covariance.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/pose_control_lidar_factor.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/pose_control_lidar_correlation.cpp
)

add_library(pose_control_test_core STATIC
  ${POSE_CONTROL_TEST_CORE_SOURCES}
)

target_include_directories(pose_control_test_core PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/reference
)

target_link_libraries(pose_control_test_core PUBLIC
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)

add_library(pose_control_test_reference STATIC
  reference/physical_trajectory_reference.cpp
)

target_include_directories(pose_control_test_reference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/reference
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
)

# New files are discovered automatically. CMake will notice a changed file
# set, but editing an existing test does not invalidate unrelated targets.
file(GLOB POSE_CONTROL_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../test_pose_control_*.cpp"
)

set(POSE_CONTROL_TEST_TARGETS)
foreach(test_source IN LISTS POSE_CONTROL_TEST_SOURCES)
  get_filename_component(test_name "${test_source}" NAME_WE)
  add_executable(${test_name} "${test_source}")
  target_link_libraries(${test_name}
    pose_control_test_core
    pose_control_test_reference
  )
  list(APPEND POSE_CONTROL_TEST_TARGETS ${test_name})
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

add_custom_target(pose_control_tests
  DEPENDS ${POSE_CONTROL_TEST_TARGETS}
)
