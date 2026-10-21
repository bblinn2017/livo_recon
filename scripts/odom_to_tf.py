#!/usr/bin/env python3
"""Bridge /livo_recon/odometry (nav_msgs/Odometry) to a TF broadcast.

livo_recon itself never publishes TF -- every display (PointCloud2,
Odometry, Path) carries data already expressed in header.frame_id, so the
pipeline never needed it. rviz's ThirdPersonFollower camera, however, can
only track a live TF frame via its Target Frame setting. This node closes
that gap for visualization only: it republishes each Odometry message's
own pose as camera_init -> body, with no effect on the pipeline itself.
"""
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

br = None


def cb(msg: Odometry):
    t = TransformStamped()
    t.header.stamp = msg.header.stamp
    t.header.frame_id = msg.header.frame_id
    t.child_frame_id = msg.child_frame_id
    t.transform.translation.x = msg.pose.pose.position.x
    t.transform.translation.y = msg.pose.pose.position.y
    t.transform.translation.z = msg.pose.pose.position.z
    t.transform.rotation = msg.pose.pose.orientation
    br.sendTransform(t)


if __name__ == "__main__":
    rospy.init_node("odom_to_tf", anonymous=True)
    br = tf2_ros.TransformBroadcaster()
    rospy.Subscriber("/livo_recon/odometry", Odometry, cb, queue_size=50)
    rospy.spin()
