#!/usr/bin/env python3

import sys
import rospy

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rospkg import RosPack; rp = RosPack()
repo_path = rp.get_path('hsr_isaac_localization')

sys.path.append(repo_path)

# from scripts.hsr_simple_mover import hsr_simple_mover_sim

class hsr_isaac_controller_joy:
    def __init__(self):
        rospy.init_node('hsr_isaac_controller_joy')

        # self.simple_mover = hsr_simple_mover_sim()

        self.update_rate = 30

        # Base Control
        self.twist_base_cmd_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)

        # Head Control
        self.head_joint_cmd_pub = rospy.Publisher('/hsrb/head_trajectory_controller/command', JointTrajectory, queue_size=10)

        self.head_pan = 0
        self.head_tilt = 0

        self.HEAD_PAN_MIN, self.HEAD_PAN_MAX = -1.5, 1.5
        self.HEAD_TILT_MIN, self.HEAD_TILT_MAX = -1.0, 1.0

        self.head_pan_speed = 0.02
        self.head_tilt_speed = 0.02

        # Arm Control
        self.arm_joint_cmd_pub = rospy.Publisher('/hsrb/arm_trajectory_controller/command', JointTrajectory, queue_size=10)

        self.arm_lift = 0
        self.ARM_LIFT_MIN, self.ARM_LIFT_MAX = 0, 0.35
        self.arm_lift_speed = 0.005

        self.arm_flex = 0
        self.ARM_FLEX_MIN, self.ARM_FLEX_MAX = -1.5, 0.0
        self.arm_flex_speed = 0.005

        self.arm_roll = 0
        self.ARM_ROLL_MIN, self.ARM_ROLL_MAX = 0.0, 0.0
        self.arm_roll_speed = 0.005

        self.wrist_flex = 0
        self.WRIST_FLEX_MIN, self.WRIST_FLEX_MAX = 0.0, 0.0
        self.wrist_flex_speed = 0.005

        self.wrist_roll = 0
        self.WRIST_ROLL_MIN, self.WRIST_ROLL_MAX = 0.0, 0.0
        self.wrist_roll_speed = 0.005

        self.wrist_flex = 0
        self.WRIST_FLEX_MIN, self.WRIST_FLEX_MAX = 0.0, 0.0
        self.wrist_flex_speed = 0.005

        self.hand_motor = 0
        self.HAND_MOTOR_MIN, self.HAND_MOTOR_MAX = 0.0, 0.0
        self.hand_motor_speed = 0.005

        self.hand_l_proximal = 0
        self.HAND_L_PROXIMAL_MIN, self.HAND_L_PROXIMAL_MAX = 0.0, 0.0
        self.hand_l_proximal_speed = 0.005

        self.hand_r_proximal = 0
        self.HAND_R_PROXIMAL_MIN, self.HAND_R_PROXIMAL_MAX = 0.0, 0.0
        self.hand_r_proximal_speed = 0.005

    def joy_callback(self, data):
        # print(f'data:', data)

        axes = data.axes
        print(f'axes:', axes)

        buttons = data.buttons
        print(f'buttons:', buttons)

        # Xbox Controller Configuration
        ab_activated = bool(buttons[0])
        print(f'ab_activated:', ab_activated)

        bb_activated = bool(buttons[1])
        print(f'bb_activated:', bb_activated)

        xb_activated = bool(buttons[2])
        print(f'xb_activated:', xb_activated)

        yb_activated = bool(buttons[3])
        print(f'yb_activated:', yb_activated)

        lb_activated = bool(buttons[4])
        print(f'lb_activated:', lb_activated)

        rb_activated = bool(buttons[5])
        print(f'rb_activated:', rb_activated)

        sb_activated = bool(buttons[7])
        print(f'sb_activated:', sb_activated)

        r3_activated = bool(buttons[10])
        print(f'r3_activated:', r3_activated)

        lt_activated = (axes[2] == -1)
        print(f'lt_activated:', lt_activated)

        lr_left_thumb = axes[0]
        fb_left_thumb = axes[1]

        print('lr_left_thumb:', lr_left_thumb)
        print(f'fb_left_thumb:', fb_left_thumb)
        
        lr_right_thumb = axes[3]
        fb_right_thumb = axes[4]

        print('lr_right_thumb:', lr_right_thumb)
        print('fb_right_thumb:', fb_right_thumb)

        lr_dpad = axes[6]
        fb_dpad = axes[7]

        print('lr_dpad:', lr_dpad)
        print('fb_dpad:', fb_dpad)

        twist_msg = Twist()
        head_msg = JointTrajectory()

        if lb_activated:
            print(f'Base Motion Activated!')

            if abs(fb_left_thumb) > 0.1:
                twist_msg.linear.x = fb_left_thumb

            if abs(lr_left_thumb) > 0.1:
                twist_msg.linear.y = lr_left_thumb

            if abs(lr_right_thumb) > 0.1:
                twist_msg.angular.z = lr_right_thumb

        elif lt_activated:
            head_msg.joint_names = ['head_pan_joint', 'head_tilt_joint']

            if abs(lr_right_thumb) > 0.1:
                self.head_pan += lr_right_thumb * self.head_pan_speed

            if abs(fb_right_thumb) > 0.1:
                self.head_tilt += fb_right_thumb * self.head_tilt_speed

            self.head_pan = max(self.HEAD_PAN_MIN, min(self.HEAD_PAN_MAX, self.head_pan))
            self.head_tilt = max(self.HEAD_TILT_MIN, min(self.HEAD_TILT_MAX, self.head_tilt))

            point = JointTrajectoryPoint()
            point.positions = [self.head_pan, self.head_tilt]
            point.time_from_start = rospy.Duration(1/self.update_rate)

            head_msg.points.append(point)

        # elif sb_activated:
        #     self.simple_mover.move_robot_to_neutral()
        #     pass

        elif rb_activated:
            if abs(fb_left_thumb) > 0.1:
                self.arm_lift += fb_left_thumb * self.arm_lift_speed
                self.arm_lift = max(self.ARM_LIFT_MIN, min(self.ARM_LIFT_MAX, self.arm_lift))

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['arm_lift_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.arm_lift]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

            if abs(fb_right_thumb) > 0.1:
                self.arm_flex += fb_right_thumb * self.arm_flex_speed

                self.arm_flex = max(self.ARM_FLEX_MIN, min(self.ARM_FLEX_MAX, self.arm_flex))

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['arm_flex_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.arm_flex]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

            if ab_activated:
                self.arm_roll -= self.arm_roll_speed

                # self.arm_roll = max(self.ARM_ROLL_MIN, min(self.ARM_ROLL_MAX, self.arm_roll))

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['arm_roll_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.arm_roll]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

            if yb_activated:
                self.arm_roll += self.arm_roll_speed

                # self.arm_roll = max(self.ARM_ROLL_MIN, min(self.ARM_ROLL_MAX, self.arm_roll))

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['arm_roll_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.arm_roll]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

            if bb_activated:
                self.wrist_flex -= self.wrist_flex_speed

                # self.arm_roll = max(self.ARM_ROLL_MIN, min(self.ARM_ROLL_MAX, self.arm_roll))

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['wrist_flex_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.wrist_flex]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

            if xb_activated:
                self.wrist_flex += self.wrist_flex_speed

                # self.arm_roll = max(self.ARM_ROLL_MIN, min(self.ARM_ROLL_MAX, self.arm_roll))

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['wrist_flex_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.wrist_flex]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

            if abs(lr_dpad) > 0.1:
                self.wrist_roll += lr_dpad * self.wrist_flex_speed

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['wrist_roll_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.wrist_roll]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

            elif abs(fb_dpad) > 0.1:
                self.hand_l_proximal += fb_dpad * self.hand_l_proximal_speed
                self.hand_r_proximal += fb_dpad * self.hand_r_proximal_speed

                arm_msg = JointTrajectory()
                arm_msg.joint_names = ['hand_l_proximal_joint', 'hand_r_proximal_joint']
                point = JointTrajectoryPoint()
                point.positions = [self.hand_l_proximal, self.hand_r_proximal]
                point.time_from_start = rospy.Duration(1 / self.update_rate)
                arm_msg.points.append(point)
                self.arm_joint_cmd_pub.publish(arm_msg)

        self.head_joint_cmd_pub.publish(head_msg)
        self.twist_base_cmd_pub.publish(twist_msg)

        rospy.Rate(hz=self.update_rate)
        
    def run_nodes(self):
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        rospy.spin()

if __name__ == '__main__':
    controller = hsr_isaac_controller_joy()

    controller.run_nodes()