#pragma once
// R73 (F-404): ROS wrapper over renamed_key_table.h (the table and message are ROS-free so tests/test_renamed_keys.cpp can check them).
#include <ros/ros.h>

#include <stdexcept>

#include "livo_recon/utils/log/renamed_key_table.h"

namespace livo_recon
{
inline void refuseRenamedKeys(ros::NodeHandle& pnh)
{
  for (const auto& kv : renamedKeyTable())
    if (pnh.hasParam(kv.first)) throw std::invalid_argument(renamedKeyMessage(kv.first));
  for (const auto& k : removedKeyTable())   // R74 (F-427)
    if (pnh.hasParam(k)) throw std::invalid_argument(removedKeyMessage(k));
}
}  // namespace livo_recon
