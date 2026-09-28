# Joint physical-knot estimator tests.
set(JOINT_KNOT_TEST_CORE_SOURCES
  ${PROJECT_SOURCE_DIR}/src/utils/state/state.cpp
  ${PROJECT_SOURCE_DIR}/src/diagnostics/log/debug_log_dir.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/joint_knot_estimator.cpp
)

add_library(joint_knot_test_core STATIC
  ${JOINT_KNOT_TEST_CORE_SOURCES}
)

target_include_directories(joint_knot_test_core PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(joint_knot_test_core PUBLIC
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)

file(GLOB JOINT_KNOT_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../test_joint_knot_*.cpp"
)

set(JOINT_KNOT_TEST_TARGETS)
foreach(test_source IN LISTS JOINT_KNOT_TEST_SOURCES)
  get_filename_component(test_name "${test_source}" NAME_WE)
  add_executable(${test_name} "${test_source}")
  target_link_libraries(${test_name} joint_knot_test_core)
  list(APPEND JOINT_KNOT_TEST_TARGETS ${test_name})
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

add_custom_target(joint_knot_tests DEPENDS ${JOINT_KNOT_TEST_TARGETS})
