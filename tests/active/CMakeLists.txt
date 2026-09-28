# Retained tests for production code that still exists.  Keeping these in
# CTest prevents a source file from being described as "active" while no
# configured build ever compiles it.
add_executable(test_spline
  ${PROJECT_SOURCE_DIR}/tests/test_spline.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/spline.cpp
  ${PROJECT_SOURCE_DIR}/src/lio/adaptive_q.cpp
  ${PROJECT_SOURCE_DIR}/src/diagnostics/log/debug_log_dir.cpp
)
target_link_libraries(test_spline
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)
add_test(NAME test_spline COMMAND test_spline)

add_executable(test_ds_csr
  ${PROJECT_SOURCE_DIR}/tests/test_ds_csr.cpp
)
target_link_libraries(test_ds_csr
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
add_test(NAME test_ds_csr COMMAND test_ds_csr)

add_custom_target(active_math_tests DEPENDS test_spline test_ds_csr)
