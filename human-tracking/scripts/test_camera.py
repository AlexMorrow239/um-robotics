#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

def callback(msg):
    rospy.loginfo(f"Received image: {msg.width}x{msg.height}")

rospy.init_node('test_camera')
sub = rospy.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_rect_color', Image, callback)
rospy.loginfo("Waiting for images...")
rospy.spin()