#!/usr/bin/env python3

import ast
import rospy
import sys
import numpy as np

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from visualization_msgs.msg import Marker, MarkerArray

from rospkg import RosPack; rp = RosPack()
repo_path = rp.get_path('hsr_isaac_localization')
sys.path.append(repo_path)

from scripts.general_helpers.rviz_helper import RvizHelper

class odom_correct_groundtruth:
    def __init__(self):
        rospy.init_node('odom_correct_groundtruth', anonymous=True)

        self.correct_flag = rospy.get_param('~correct_flag', True)

        self.class_name = self.__class__.__name__

        self.rviz_helper = RvizHelper()

        self.base_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)

        self.isaac_sim_robot_pose_pub = rospy.Publisher(f'{self.class_name}/isaac_sim_robot_pose/marker',
                                                        Marker,
                                                        queue_size=10)

    def establish_isaac_sim_robot_pose_marker(self):
        try:
            isaac_sim_robot_pose_marker = self.rviz_helper.generate_marker(ns='odom',
                                                    frame_id='map',
                                                    marker_type=Marker.ARROW,
                                                    marker_action=Marker.ADD,
                                                    marker_id=0,
                                                    marker_scale=(0.25, 0.05, 0.05),
                                                    marker_color=(0, 1, 0, 1),
                                                    marker_position=[self.current_isaac_robot_pose[0], self.current_isaac_robot_pose[1], 0],
                                                    marker_orientation=[0, 0, self.current_isaac_robot_pose[2]])
                
            self.isaac_sim_robot_pose_pub.publish(isaac_sim_robot_pose_marker)

        except Exception as e:
            print(f'Exception:', e)

    def move_base_pose(self, desired_pose_position, desired_pose_orientation):
        self.correcting_base = True

        pose_cov = PoseWithCovarianceStamped()
        pose_cov.header.frame_id = 'map'  # Ensure this matches your coordinate frame
        pose_cov.header.stamp = rospy.Time.now()

        # print(f'header.stamp: {pose_cov.header.stamp}')

        pose_cov.pose.pose.position.x = desired_pose_position[0]  # Set your desired x position
        pose_cov.pose.pose.position.y = desired_pose_position[1]  # Set your desired y position
        pose_cov.pose.pose.position.z = desired_pose_position[2]  # Typically 0 for 2D navigation
        
        quaternion = quaternion_from_euler(ai=desired_pose_orientation[0],
                                            aj=desired_pose_orientation[1], 
                                            ak=desired_pose_orientation[2])
        
        pose_cov.pose.pose.orientation.x = quaternion[0]
        pose_cov.pose.pose.orientation.y = quaternion[1]
        pose_cov.pose.pose.orientation.z = quaternion[2]
        pose_cov.pose.pose.orientation.w = quaternion[3]

        # Set the covariance matrix (example values)
        pose_cov.pose.covariance = [0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0]

        # self.tf_helper.tf_listener.waitForTransform(target_frame='map',
        #                             source_frame='base_footprint',
        #                             time=rospy.Time(0),
        #                             timeout=rospy.Duration(1))

        # odom_trans, odom_rot = self.tf_helper.tf_listener.lookupTransform(target_frame='map',
        #                                      source_frame='base_footprint',
        #                                      time=rospy.Time(0))
        
        # odom_rot_euler = euler_from_quaternion(quaternion=odom_rot)
        
        # reached_goal_translation = True
        # reached_goal_orientation = True
        
        # # Checks translation
        # for desired_val, odom_trans_val in zip(desired_pose_position, odom_trans):
        #     if round(desired_val, 2) != round(odom_trans_val, 2):
        #         reached_goal_translation = False
        #         break

        # # Checks rotation
        # for desired_val, odom_rot_val in zip(desired_pose_orientation, odom_rot_euler):
        #     if round(desired_val, 2) != round(odom_rot_val, 2):
        #         reached_goal_orientation = False
        #         break

        # print(f'current_odom_trans: {odom_trans}')
        # print(f'current_odom_rot_euler: {odom_rot_euler}')

        # print(f'desired_pose_position: {desired_pose_position}')
        # print(f'desired_pose_orientation: {desired_pose_orientation}')

        # print(f'reached_goal_translation: {reached_goal_translation}')
        # print(f'reached_goal_orientation: {reached_goal_orientation}')

        # print(f'CORRECTING BASE...')
        self.base_pose_pub.publish(pose_cov)
        
        # if reached_goal_translation and reached_goal_orientation:
        #     print(f'REACHED DESIRED GOAL!')

        # else:
        #     print(f'CORRECTING BASE...')
        #     self.base_pose_pub.publish(pose_cov)

        # self.current_odom_data = None
        # self.prev_odom_data = None

        self.prev_odom_correction_pose = np.array([desired_pose_position[0], desired_pose_position[1], desired_pose_orientation[2]])

        self.correcting_base = False

    def isaac_robot_pose_cb(self, robot_pose_data):
        robot_pose_data = ast.literal_eval(robot_pose_data.data)

        # print(f'received robot pose data: {robot_pose_data}')
        # print(f'receive robot pose data type: {type(robot_pose_data)}')

        self.current_isaac_robot_pose = robot_pose_data
        
        # print(f'self.current_isaac_robot_pose[0]: {self.current_isaac_robot_pose[0]}')
        # print(f'self.current_isaac_robot_pose[0] type: {type(self.current_isaac_robot_pose[0])}')

        # print(f'self.current_isaac_robot_pose:', self.current_isaac_robot_pose)

        robot_isaac_position = [self.current_isaac_robot_pose[0], self.current_isaac_robot_pose[1], 0]
        robot_isaac_orientation = [0, 0, self.current_isaac_robot_pose[2]]

        # print(f'robot_isaac_position:', robot_isaac_position)
        # print(f'robot_isaac_orientation:', robot_isaac_orientation)

        self.establish_isaac_sim_robot_pose_marker()

        if self.correct_flag is True:
            self.move_base_pose(desired_pose_position=robot_isaac_position, desired_pose_orientation=robot_isaac_orientation)

    def run_nodes(self):
        print('In run_nodes!')

        # Subscribed topics

        # Isaac Sim Robot Pose
        rospy.Subscriber("/isaac_sim/robot_pose", String, self.isaac_robot_pose_cb)

        # rospy.on_shutdown(self.shutdown_cb)

        rospy.spin()
    
if __name__ == '__main__':
    corrector = odom_correct_groundtruth()

    corrector.run_nodes()