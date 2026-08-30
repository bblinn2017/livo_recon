#pragma once

#include <vector>
#include <cstdint>
#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>

#include "livo_recon/utils/algo/math.h"
#include "livo_recon/utils/data/data_wrappers.h"
#include "livo_recon/utils/data/measures.h"

namespace livo_recon
{

// Livox (avia/mid360) per-point extraction: computes each point's absolute
// timestamp from its offset_time (smoothed/clamped to guard against rare
// out-of-order values), applies the return-type tag filter, and shares the
// same blind-radius/measures->isValid() filtering as pointsFromPointCloud2.
// point_filter_num > 1 keeps only every point_filter_num-th point of the
// raw scan (see CbkProcOptions::point_filter_num).
std::vector<PointXYZT> pointsFromLivoxMsg(
    const livox_ros_driver::CustomMsg& msg, double scan_time,
    uint8_t tag_mask, uint8_t tag_max_keep,
    double blind_sqr, const MeasuresPtr& measures,
    int point_filter_num = 1);

// Generic Ouster/PCL PointCloud2 extraction: reads each point's own "t"
// field (nanoseconds since scan start), if present, for a true per-point
// timestamp instead of one timestamp for the whole scan — this lets
// downstream consumers split a scan exactly at a sync boundary instead of
// only being able to take it whole. Falls back to the scan's header stamp
// for every point if no "t" field is present. Applies the same
// blind-radius/measures->isValid() filtering as pointsFromLivoxMsg.
// point_filter_num > 1 keeps only every point_filter_num-th point of the
// raw scan (see CbkProcOptions::point_filter_num).
std::vector<PointXYZT> pointsFromPointCloud2(
    const sensor_msgs::PointCloud2& msg, double scan_time,
    double blind_sqr, const MeasuresPtr& measures,
    int point_filter_num = 1);

}  // namespace livo_recon
