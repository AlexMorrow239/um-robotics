#!/usr/bin/env python3

import actionlib
import json
import math
import moveit_commander
import os
import rospy

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

from hsrb_interface import Robot

ros_master_uri_string = os.environ['ROS_MASTER_URI']
print(f'ros_master_uri_string: {ros_master_uri_string}')

if 'hsrb.local' in ros_master_uri_string:
    ros_master_status = 'hm'

else:
    ros_master_status = 'sm'

class hsr_simple_mover_sim:
    def move_arm_neutral(self):
        self.arm.set_named_target('neutral')
        return self.arm.go()
    
    def move_arm_to_go(self):
        self.arm.set_named_target('go')
        return self.arm.go()
    
    def move_to_pose(self, group, joint_dict):
        group.set_joint_value_target(joint_dict)
        
        plan = group.go(wait=True)

        group.stop()
        
        print(f'group [{group.get_name()}] - finished move_to_pose!')
    
    def move_head_neutral(self):
        self.move_to_pose(group=self.head, joint_dict={'head_pan_joint': 0.0,
                                                       'head_tilt_joint': 0.0})
        
    def move_robot_to_neutral(self):
        self.move_arm_neutral()
        self.move_head_neutral()

    def move_robot_to_go(self):
        self.move_arm_to_go()
        self.move_head_neutral()

    def move_base_abs_goal(self, x, y, theta):
        print(f'Starting move_base_abs goal: {(x, y, theta)}')

        goal = MoveBaseGoal()

        goal.target_pose.header.frame_id = "map"

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        theta_rad = math.radians(theta)

        quaternion = quaternion_from_euler(0, 0, theta_rad)
        
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        self.navclient.send_goal(goal)
        self.navclient.wait_for_result()
        state = self.navclient.get_state()

        print(f'End move_base_abs goal: {(x, y, theta)}')
        
        return True if state == 3 else False
    
    def move_base_to_origin(self):
        self.move_base_abs_goal(x=0, y=0, theta=0)
    
    def __init__(self):
        self.navclient = actionlib.SimpleActionClient('/move_base', MoveBaseAction)

        self.base = moveit_commander.MoveGroupCommander("base")
        self.base.set_max_acceleration_scaling_factor(1)
        self.base.set_max_velocity_scaling_factor(1)

        self.current_base_state = self.base.get_current_pose()

        self.whole_body_light = moveit_commander.MoveGroupCommander("whole_body_light")
        self.whole_body_light.set_max_acceleration_scaling_factor(1)
        self.whole_body_light.set_max_velocity_scaling_factor(1)

        self.arm = moveit_commander.MoveGroupCommander('arm')
        self.arm.set_max_acceleration_scaling_factor(1)                                                                                                                                                                                
        self.arm.set_max_velocity_scaling_factor(1)

        self.gripper = moveit_commander.MoveGroupCommander('gripper')
        self.gripper.set_max_acceleration_scaling_factor(1)
        self.gripper.set_max_velocity_scaling_factor(1)

        self.head = moveit_commander.MoveGroupCommander('head')
        self.head.set_max_acceleration_scaling_factor(1)
        self.head.set_max_velocity_scaling_factor(1)

        # self.move_robot_to_neutral()
        self.move_robot_to_go()

class hsr_simple_mover_real:
    def move_arm_neutral(self):
        return self.whole_body.move_to_neutral()
    
    def move_arm_to_go(self):
        return self.whole_body.move_to_go()

    def move_to_pose(self, joint_dict):
        return self.whole_body.move_to_joint_positions(joint_dict)

    def move_robot_to_neutral(self):
        return self.whole_body.move_to_neutral()

    def move_robot_to_go(self):
        return self.whole_body.move_to_go()
    
    def move_base_abs_goal(self, x, y, theta):
        print(f'Starting move_base_abs goal: {(x, y, theta)}')

        theta_rad = math.radians(theta)

        self.omni_base.go_abs(x, y, theta_rad)

        print(f'End move_base_abs goal: {(x, y, theta)}')

        return True
    
    def move_base_to_origin(self):
        self.move_base_abs_goal(x=0, y=0, theta=0)
    
    def __init__(self):
        self.robot = Robot()
        self.omni_base = self.robot.try_get('omni_base')
        self.tts = self.robot.try_get('default_tts')
        self.tts.language = self.tts.ENGLISH
        self.whole_body = self.robot.try_get('whole_body')
        self.gripper = self.robot.try_get('gripper')


if __name__ == '__main__':
    rospy.init_node('hsr_simple_mover', anonymous=True)

    simple_mover = hsr_simple_mover_sim()

    simple_mover.move_robot_to_go()
    simple_mover.move_base_abs_goal(x=0.0, y=0.0, theta=0)

    simple_mover.move_robot_to_neutral()
    simple_mover.move_to_pose(group=simple_mover.head, joint_dict={'head_pan_joint': 1.65,
                                                                   'head_tilt_joint': -1})

    rospy.sleep(2)

    # This is a small example of a combination of robot move_goal + relevant robot motion groups.
    simple_mover.move_robot_to_go()
    simple_mover.move_base_abs_goal(x=-1.0, y=-0.3, theta=-90)

    simple_mover.move_robot_to_neutral()

    # Degree conversion to radians to match rviz motion planning visualization.

    # Look at default scene camera.
    simple_mover.move_to_pose(group=simple_mover.head, joint_dict={'head_pan_joint': -0.95,
                                                                   'head_tilt_joint': 0.17})
    
    # Move arm up to demonstrate all joint motion groups.
    simple_mover.move_to_pose(group=simple_mover.arm, joint_dict={'arm_lift_joint': 0.35,
                                                                  'arm_flex_joint': -0.70,
                                                                  'arm_roll_joint': 0.20,
                                                                  'wrist_flex_joint': 0.73,
                                                                  'wrist_roll_joint': 2.0})
    
    # Open gripper, wait, then close.
    simple_mover.move_to_pose(group=simple_mover.gripper, joint_dict={'hand_motor_joint': 1.0})

    rospy.sleep(2)

    simple_mover.move_to_pose(group=simple_mover.gripper, joint_dict={'hand_motor_joint': 0.0})
    
    rospy.sleep(2)

    simple_mover.move_robot_to_neutral()