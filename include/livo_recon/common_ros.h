#pragma once

// History (3-10): see docs/livo_recon_changelog.md#include-livo_recon-common_ros.h-3

#include <ros/ros.h>
#include <ros/package.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <livox_ros_driver/CustomMsg.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <optional>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>
