#!/usr/bin/env python3
"""Dumps a bag's IMU topic to text for the vibration-canceller tools: t wx wy wz ax ay az (as received; acc in g on the
mid360) and the first image stamp (the reference for 'frames 1-5').
usage: dump_imu.py BAG OUT_IMU_TXT [--topic /livox/imu] [--image-topic /ximea_cam/image_raw] [--seconds S]"""
import argparse
import rosbag
import rospy

ap = argparse.ArgumentParser()
ap.add_argument("bag"); ap.add_argument("out")
ap.add_argument("--topic", default="/livox/imu"); ap.add_argument("--image-topic", default="/ximea_cam/image_raw")
ap.add_argument("--seconds", type=float, default=0.0, help="only the first S seconds (0: all)")
a = ap.parse_args()
b = rosbag.Bag(a.bag)
end = rospy.Time.from_sec(b.get_start_time() + a.seconds) if a.seconds > 0 else None
n = 0
with open(a.out, "w") as f:
    for _, m, _ in b.read_messages(topics=[a.topic], end_time=end):
        f.write("%.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (m.header.stamp.to_sec(), m.angular_velocity.x, m.angular_velocity.y,
                m.angular_velocity.z, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
        n += 1
first_img = next((m.header.stamp.to_sec() for _, m, _ in b.read_messages(topics=[a.image_topic], end_time=end)), None)
print("%d IMU samples -> %s; first image stamp %s" % (n, a.out, "%.9f" % first_img if first_img else "none"))
