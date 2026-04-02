#!/usr/bin/env python3
"""Navigation controller for the HSR robot in the simulated lab.

Manages route execution, pose tracking, and movement commands. Reads predefined
waypoint routes and drives the robot through them using the move base action server.
"""

import ast
import datetime
import json
import os
import rospy
import subprocess
import sys

import tf

import numpy as np

from std_msgs.msg import String

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

from rospkg import RosPack; rp = RosPack()
repo_path = rp.get_path('hsr-localization-navigation')

sys.path.append(repo_path)

from scripts.hsr_simple_mover import hsr_simple_mover_sim, hsr_simple_mover_real
from scripts.general_helpers.general_helper import GeneralHelper
from scripts.general_helpers.rviz_helper import RvizHelper

ros_master_uri_string = os.environ['ROS_MASTER_URI']
print(f'ros_master_uri_string: {ros_master_uri_string}')

if 'hsrb.local' in ros_master_uri_string:
    from dynamic_reconfigure.client import Client
    ros_master_status = 'hm'

else:
    ros_master_status = 'sm'

print(f'ros_master_status: {ros_master_status}')

class hsr_lab_mover:
    def establish_dynamic_reconfig(self):
        reconf_base = Client('tmc_map_merger/inputs/base_scan/obstacle_circle')
        reconf_head = Client('tmc_map_merger/inputs/head_rgbd_sensor/obstacle_circle')

        # Original
        reconf_base.update_configuration({'forbid_radius': 0.25, 'obstacle_radius': 0.35, 'obstacle_occupancy': 10})
        reconf_head.update_configuration({'forbid_radius': 0.25, 'obstacle_radius': 0.35, 'obstacle_occupancy': 10})

        reconf_base.update_configuration({'forbid_radius': 0.25, 'obstacle_radius': 0.35, 'obstacle_occupancy': 10})
        reconf_head.update_configuration({'forbid_radius': 0.25, 'obstacle_radius': 0.35, 'obstacle_occupancy': 10})

    def init_logging_data(self):
        self.num_target_poses = 0
        self.num_reached_poses = 0
        self.robot_center_camera_groundtruth_pose_list = []
        self.current_isaac_robot_pose = None
        self.isaac_robot_pose_list = []
        self.odom_pose_list = []
        self.amcl_mean_pose_list = []
        self.amcl_var_list = []
        self.amcl_std_list = []
        self.current_mean_pf_occ_coverage = None
        self.pf_occ_coverage_list = []
        self.pf_mean_pose_list = []
        self.pf_var_list = []
        self.pf_std_list = []
        self.landmarks_seen_list = []
        self.groundtruth_amcl_x_pos_diff_list = []
        self.groundtruth_amcl_y_pos_diff_list = []
        self.groundtruth_amcl_xy_pos_diff_list = []
        self.groundtruth_amcl_angle_diff_list = []
        self.groundtruth_pf_x_pos_diff_list = []
        self.groundtruth_pf_y_pos_diff_list = []
        self.groundtruth_pf_xy_pos_diff_list = []
        self.groundtruth_pf_angle_diff_list = []

    def read_locations_json(self, locations_json_file: str) -> dict:
        locations_json_path = os.path.join(repo_path, 'config', 'robot_move', locations_json_file)

        locations_json = json.load(open(locations_json_path))

        return locations_json
    
    def configure_node_rate(self, hz):
        self.node_rate = hz
        self.node_time_interval = 1/self.node_rate

    def establish_rviz_pubs(self):
        self.locations_marker_array_pub = rospy.Publisher(f'{self.class_name}/locations/markers',
                                                          MarkerArray,
                                                          queue_size=10)
        
        self.locations_label_markers_array_pub = rospy.Publisher(f'{self.class_name}/locations_label/markers',
                                                                 MarkerArray,
                                                                 queue_size=10)
            
    def __init__(self):
        rospy.init_node('hsr_lab_mover', anonymous=True)

        if ros_master_status == 'hm':
            self.establish_dynamic_reconfig()

        self.general_helper = GeneralHelper()
        self.rviz_helper = RvizHelper()

        self.tf_listener = tf.TransformListener()

        self.class_name = self.__class__.__name__

        self.chosen_algorithm = rospy.get_param('~chosen_algorithm', 'pf')
        self.pose_delay_time = rospy.get_param('~pose_delay_time', None)

        self.init_logging_data()

        self.locations_json = self.read_locations_json(locations_json_file='rc_lab_pos.json')

        self.general_helper.pprint_dict(dict=self.locations_json)

        if ros_master_status == 'hm':
            self.lap_name = 'real_short_lap'
        else:
            self.lap_name = 'sim_full_front_lap'

        self.lap_name = rospy.get_param('~lap_name', self.lap_name)

        print(f'Current lap name: {self.lap_name}')

        self.locations_dict = self.locations_json['routes'][self.lap_name]
        print(f'self.locations_dict:', self.locations_dict)

        self.locations_dict_radians = {k: [v[0], v[1], np.radians(v[2])] for k,v in list(self.locations_json['routes'][self.lap_name].items())}
        print(f'self.locations_dict_radians:', self.locations_dict_radians)

        self.locations_list = list(self.locations_dict.items())

        self.start_idx = None
        self.end_idx = None

        if self.start_idx is not None and self.end_idx is not None:
            self.locations_list = self.locations_list[self.start_idx:self.end_idx]
            
        print(f'self.locations_list:', self.locations_list)

        self.configure_node_rate(hz=10)

        self.establish_rviz_pubs()

        print(f'Listening for start signal...')

        if ros_master_status == 'hm':
            self.lab_mover_behavior_start()
        else:
            self.behavior_started = False
            rospy.Subscriber('/isaac_sim/robot_behavior_start', String, self.start_behavior_cb)

        rospy.on_shutdown(self.shutdown_cb)

        rospy.spin()

    def publish_move_locations_timed_cb(self, event):
        self.publish_move_locations()

    def get_robot_center_camera_groundtruth_timed_cb(self, event):
        if ros_master_status == 'hm':
            self.robot_center_camera_groundtruth_trans, self.robot_center_camera_groundtruth_rot = self.get_transform(target_frame='map', source_frame='robot_center_camera_groundtruth')

            if self.robot_center_camera_groundtruth_rot is not None:
                self.robot_center_camera_groundtruth_rot = euler_from_quaternion(quaternion=self.robot_center_camera_groundtruth_rot)

            if self.robot_center_camera_groundtruth_trans is not None and self.robot_center_camera_groundtruth_rot is not None:
                self.current_robot_center_camera_groundtruth_pose = [self.robot_center_camera_groundtruth_trans[0], self.robot_center_camera_groundtruth_trans[1], self.robot_center_camera_groundtruth_rot[2]]
            else:
                self.current_robot_center_camera_groundtruth_pose = None

    def delete_all_markers(self, ns):
        print(f'Deleting all markers: {ns}')

        markers = []

        marker = Marker()
        marker.ns = ns
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.DELETEALL

        markers.append(marker)

        marker_array = MarkerArray(markers)

        if ns == 'locations':
            self.locations_marker_array_pub.publish(marker_array)

        if ns == 'locations_label':
            self.locations_label_markers_array_pub.publish(marker_array)

    def shutdown_cb(self):
        self.delete_all_markers(ns='locations')
        self.delete_all_markers(ns='locations_label')

    def publish_move_locations(self):
        location_markers, locations_label_markers = [], []
        label_position_counts = {}

        for i, kv in enumerate(self.locations_list):
            k, v = kv
            x, y = v[0], v[1]

            # Track how many times the same position has been encountered for label markers
            if (x, y) in label_position_counts:
                label_position_counts[(x, y)] += 1
            else:
                label_position_counts[(x, y)] = 1

            z_offset = 0.1 * label_position_counts[(x, y)]  # Adjust height only for label markers

            # Location marker (unchanged)
            location_marker = self.rviz_helper.generate_marker(ns='locations',
                                                            frame_id='map',
                                                            marker_type=Marker.CUBE,
                                                            marker_action=Marker.ADD,
                                                            marker_id=i,
                                                            marker_scale=(0.02, 0.02, 0.02),
                                                            marker_color=(0, 0, 0, 1),
                                                            marker_position=[x, y, 0])

            # Location label marker (with z_offset to prevent overlap)
            location_label_marker = self.rviz_helper.generate_marker(ns='locations_labels',
                                                                    frame_id='map',
                                                                    marker_type=Marker.TEXT_VIEW_FACING,
                                                                    marker_action=Marker.ADD,
                                                                    marker_id=i,
                                                                    marker_text=str(k),
                                                                    marker_scale=(0.1, 0.1, 0.1),
                                                                    marker_color=(1, 1, 1, 1),
                                                                    marker_position=[x, y, 0 + z_offset])

            location_markers.append(location_marker)
            locations_label_markers.append(location_label_marker)

        location_markers = MarkerArray(location_markers)
        locations_label_markers = MarkerArray(locations_label_markers)

        self.locations_marker_array_pub.publish(location_markers)
        self.locations_label_markers_array_pub.publish(locations_label_markers)

    def pose_from_marker(self, marker):
        position = marker.pose.position

        position = [position.x, position.y, position.z]

        orientation = marker.pose.orientation

        orientation_euler = euler_from_quaternion(quaternion=[orientation.x, orientation.y, orientation.z, orientation.w])

        pose = [position[0], position[1], orientation_euler[2]]

        return pose
    
    def get_transform(self, target_frame, source_frame):
        try:
            now = rospy.Time.now()
            self.tf_listener.waitForTransform(target_frame=target_frame,
                                              source_frame=source_frame,
                                              time=rospy.Time(0),
                                              timeout=rospy.Duration(1.0))
            
            (trans, rot) = self.tf_listener.lookupTransform(target_frame=target_frame,
                                                            source_frame=source_frame,
                                                            time=rospy.Time(0))
            
            return trans, rot
        
        except Exception as e:
            rospy.logwarn(f'Failed to get transform from {source_frame} to {target_frame}. Exception: {e}')
            return None, None

    def isaac_robot_pose_cb(self, marker):
        self.current_isaac_robot_pose = self.pose_from_marker(marker=marker)

    def odom_pose_cb(self, marker):
        self.current_odom_robot_pose = self.pose_from_marker(marker=marker)

    def amcl_pose_cb(self, msg):
        pose = msg.pose.pose
        covariance = msg.pose.covariance

        pose_pos = pose.position

        pose_pos_x = pose_pos.x
        pose_pos_y = pose_pos.y
        pose_pos_z = pose_pos.z

        orientation = pose.orientation

        orientation_euler = euler_from_quaternion(quaternion=[orientation.x, orientation.y, orientation.z, orientation.w])

        self.current_amcl_mean_pose = [pose_pos_x, pose_pos_y, orientation_euler[2]]

        var_x = float(covariance[0])
        var_y = float(covariance[7])
        var_theta = float(covariance[35])

        self.current_amcl_var = [var_x, var_y, var_theta]

        std_x = np.sqrt(max(0.0, var_x))
        std_y = np.sqrt(max(0.0, var_y))
        std_theta = np.sqrt(max(0.0, var_theta))

        self.current_amcl_std = [std_x, std_y, std_theta]

    def mean_pf_pose_cb(self, marker):
        self.current_mean_pf_robot_pose = self.pose_from_marker(marker=marker)

    def pf_particle_mean_occ_coverage_cb(self, data):
        self.current_mean_pf_occ_coverage = float(ast.literal_eval(data.data))

    def pf_var_cb(self, data):
        self.current_pf_var = [float(elem) for elem in ast.literal_eval(data.data)]

    def pf_std_cb(self, data):
        self.current_pf_std = [float(elem) for elem in ast.literal_eval(data.data)]

    def num_landmarks_seen_data_cb(self, data):
        self.current_num_landmarks_seen = int(str(data.data))

    def capture_robot_positions(self):
        print(f'IN CAPTURE ROBOT POSITIONS!')

        if self.chosen_algorithm == 'amcl':
            rospy.Subscriber('/odom_correct_groundtruth/isaac_sim_robot_pose/marker', Marker, self.isaac_robot_pose_cb)

            rospy.Subscriber('/odom_publisher/odom/marker', Marker, self.odom_pose_cb)

            rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_cb)

        if self.chosen_algorithm == 'pf':
            if ros_master_status == 'hm':
                # Current groundtruth robot pose in real lab.
                rospy.Timer(rospy.Duration(self.node_time_interval), self.get_robot_center_camera_groundtruth_timed_cb)

            if ros_master_status == 'sm':
                # Current groundtruth robot pose in isaac sim.
                rospy.Subscriber('/hsr_pf_localization/isaac_sim_robot_pose/marker', Marker, self.isaac_robot_pose_cb)

            # Current odom robot pose.
            rospy.Subscriber('/hsr_pf_localization/odom/marker', Marker, self.odom_pose_cb)

            # Current mean pf robot pose.
            rospy.Subscriber('/hsr_pf_localization/pf_particles_mean/marker', Marker, self.mean_pf_pose_cb)

            # Current mean pf particle occupancy map coverage.
            rospy.Subscriber('/hsr_pf_localization/pf_particles_mean_occ_coverage', String, self.pf_particle_mean_occ_coverage_cb)

            # Current pf var.
            rospy.Subscriber('/hsr_pf_localization/pf_particles_variance', String, self.pf_var_cb)

            # Current pf std.
            rospy.Subscriber('/hsr_pf_localization/pf_particles_std', String, self.pf_std_cb)

            # Current num landmarks seen
            rospy.Subscriber('/hsr_pf_localization/pf_sensor_data/num_landmarks_seen', String, self.num_landmarks_seen_data_cb)

    def write_logging_data(self, data_file_name):
        logging_dir = os.path.join(repo_path, 'data', 'routes', f'{self.lap_name}_{self.chosen_algorithm}', 'json')

        logging_file_path = os.path.join(logging_dir, data_file_name)

        print(self.general_helper.emph_sep)
        print(f'logging file path: {logging_file_path}\n')
        print(self.general_helper.emph_sep)

        if self.chosen_algorithm == 'amcl':
            locations_poses_recording = {
                'num_target_poses' : self.num_target_poses,
                'num_reached_poses' : self.num_reached_poses,
                'route_poses' : self.locations_dict_radians,
                'odom_poses' : self.odom_pose_list,
                'amcl_mean_poses' : self.amcl_mean_pose_list,
                'amcl_variances' : self.amcl_var_list,
                'amcl_stds' : self.amcl_std_list,
                'num_landmarks_seen': self.landmarks_seen_list,
                'groundtruth_amcl_x_pos_diffs' : self.groundtruth_amcl_x_pos_diff_list,
                'groundtruth_amcl_y_pos_diffs' : self.groundtruth_amcl_y_pos_diff_list,
                'groundtruth_amcl_xy_pos_diffs' : self.groundtruth_amcl_xy_pos_diff_list,
                'groundtruth_amcl_angle_diffs' : self.groundtruth_amcl_angle_diff_list,
                'mean_amcl_variance': {
                    'mean_amcl_variance_x_pos': self.mean_var_x,
                    'mean_amcl_variance_y_pos': self.mean_var_y,
                    'mean_amcl_variance_angle': self.mean_var_theta
                },
                'mean_amcl_std': {
                    'mean_amcl_std_x_pos': self.mean_std_x,
                    'mean_amcl_std_y_pos': self.mean_std_y,
                    'mean_amcl_std_angle': self.mean_std_theta
                },
                'rmse' : {
                    'rmse_x_pos' : self.rmse_x_pos,
                    'rmse_y_pos' : self.rmse_y_pos,
                    'rmse_xy_pos' : self.rmse_xy_pos,
                    'rmse_angle' : self.rmse_angle,
                },
                'mae' : {
                    'mae_x_pos' : self.mae_x_pos,
                    'mae_y_pos' : self.mae_y_pos,
                    'mae_xy_pos' : self.mae_xy_pos,
                    'mae_angle' : self.mae_angle, 
                }
            }

        if self.chosen_algorithm == 'pf':
            locations_poses_recording = {
                'num_target_poses' : self.num_target_poses,
                'num_reached_poses' : self.num_reached_poses,
                'route_poses' : self.locations_dict_radians,
                'odom_poses' : self.odom_pose_list,
                'pf_mean_poses' : self.pf_mean_pose_list,
                'pf_occ_coverages': self.pf_occ_coverage_list,
                'pf_variances' : self.pf_var_list,
                'pf_stds' : self.pf_std_list,
                'num_landmarks_seen' : self.landmarks_seen_list,
                'groundtruth_pf_x_pos_diffs' : self.groundtruth_pf_x_pos_diff_list,
                'groundtruth_pf_y_pos_diffs' : self.groundtruth_pf_y_pos_diff_list,
                'groundtruth_pf_xy_pos_diffs' : self.groundtruth_pf_xy_pos_diff_list,
                'groundtruth_pf_angle_diffs' : self.groundtruth_pf_angle_diff_list,
                'mean_pf_variance': {
                    'mean_pf_variance_x_pos': self.mean_var_x,
                    'mean_pf_variance_y_pos': self.mean_var_y,
                    'mean_pf_variance_angle': self.mean_var_theta
                },
                'mean_pf_std': {
                    'mean_pf_std_x_pos': self.mean_std_x,
                    'mean_pf_std_y_pos': self.mean_std_y,
                    'mean_pf_std_angle': self.mean_std_theta
                },
                'rmse' : {
                    'rmse_x_pos' : self.rmse_x_pos,
                    'rmse_y_pos' : self.rmse_y_pos,
                    'rmse_xy_pos' : self.rmse_xy_pos,
                    'rmse_angle' : self.rmse_angle,
                },
                'mae' : {
                    'mae_x_pos' : self.mae_x_pos,
                    'mae_y_pos' : self.mae_y_pos,
                    'mae_xy_pos' : self.mae_xy_pos,
                    'mae_angle' : self.mae_angle, 
                }
            }

        if ros_master_status == 'hm':
            locations_poses_recording['groundtruth_poses'] = self.robot_center_camera_groundtruth_pose_list
        if ros_master_status == 'sm':
            locations_poses_recording['groundtruth_poses'] = self.isaac_robot_pose_list

        logging_dict = locations_poses_recording

        self.general_helper.write_json_file(data_dict=logging_dict, data_file_path=logging_file_path)

    def move_behavior(self, capture_data_trigger=False, current_iter=None):
        self.init_logging_data()

        if ros_master_status == 'hm':
            self.simple_mover = hsr_simple_mover_real()
        else:
            self.simple_mover = hsr_simple_mover_sim()

        self.simple_mover.move_robot_to_go()
        # simple_mover.move_base_to_origin() # If localization is good, this will work correctly. 

        self.num_target_poses = len(self.locations_list)

        if ros_master_status == 'hm':
            if self.pose_delay_time is not None:
                rospy.sleep(int(self.pose_delay_time))
            else:
                rospy.sleep(1)

        for k,v in self.locations_list:
            print(f'k: {k}, v:{v}')

            self.simple_mover.move_robot_to_go()

            x,y,theta = v

            self.simple_mover.move_base_abs_goal(x=x, y=y, theta=theta)

            if self.pose_delay_time is not None:
                rospy.sleep(int(self.pose_delay_time))
            else:
                rospy.sleep(1)

            if self.chosen_algorithm == 'amcl':
                if ros_master_status == 'hm':
                    x_dist_groundtruth_amcl = np.abs(self.current_robot_center_camera_groundtruth_pose[0] - self.current_amcl_mean_pose[0])
                    y_dist_groundtruth_amcl = np.abs(self.current_robot_center_camera_groundtruth_pose[1] - self.current_amcl_mean_pose[1])
                    euclid_dist_groundtruth_amcl = np.linalg.norm(np.array(self.current_robot_center_camera_groundtruth_pose[:2]) - np.array(self.current_amcl_mean_pose[:2]))
                    angular_diff_groundtruth_amcl = (self.current_robot_center_camera_groundtruth_pose[2] - self.current_amcl_mean_pose[2] + np.pi) % (2 * np.pi) - np.pi
                else:
                    x_dist_groundtruth_amcl = None
                    y_dist_groundtruth_amcl = None
                    euclid_dist_groundtruth_amcl = None
                    angular_diff_groundtruth_amcl = None

                if ros_master_status == 'sm':
                    x_dist_groundtruth_amcl = np.abs(self.current_isaac_robot_pose[0] - self.current_amcl_mean_pose[0])
                    y_dist_groundtruth_amcl = np.abs(self.current_isaac_robot_pose[1] - self.current_amcl_mean_pose[1])
                    euclid_dist_groundtruth_amcl = np.linalg.norm(np.array(self.current_isaac_robot_pose[:2]) - np.array(self.current_amcl_mean_pose[:2]))
                    angular_diff_groundtruth_amcl = (self.current_isaac_robot_pose[2] - self.current_amcl_mean_pose[2] + np.pi) % (2 * np.pi) - np.pi
                else:
                    x_dist_groundtruth_amcl = None
                    y_dist_groundtruth_amcl = None
                    euclid_dist_groundtruth_amcl = None
                    angular_diff_groundtruth_amcl = None

                print(f'current_x_dist_groundtruth_amcl:', x_dist_groundtruth_amcl)
                print(f'current y_dist_groundtruth_amcl:', y_dist_groundtruth_amcl)
                print(f'current euclid_dist_groundtruth_pf:', euclid_dist_groundtruth_pf)
                print(f'current angular_diff_groundtruth_pf:', angular_diff_groundtruth_pf)

            if self.chosen_algorithm == 'pf':
                if ros_master_status == 'hm':
                    if self.current_robot_center_camera_groundtruth_pose is not None:
                        x_dist_groundtruth_pf = np.abs(self.current_robot_center_camera_groundtruth_pose[0] - self.current_mean_pf_robot_pose[0])
                        y_dist_groundtruth_pf = np.abs(self.current_robot_center_camera_groundtruth_pose[1] - self.current_mean_pf_robot_pose[1])
                        euclid_dist_groundtruth_pf = np.linalg.norm(np.array(self.current_robot_center_camera_groundtruth_pose[:2]) - np.array(self.current_mean_pf_robot_pose[:2]))
                        angular_diff_groundtruth_pf = (self.current_robot_center_camera_groundtruth_pose[2] - self.current_mean_pf_robot_pose[2] + np.pi) % (2 * np.pi) - np.pi
                    else:
                        x_dist_groundtruth_pf = None
                        y_dist_groundtruth_pf = None
                        euclid_dist_groundtruth_pf = None
                        angular_diff_groundtruth_pf = None

                if ros_master_status == 'sm':
                    if self.current_isaac_robot_pose is not None:
                        x_dist_groundtruth_pf = np.abs(self.current_isaac_robot_pose[0] - self.current_mean_pf_robot_pose[0])
                        y_dist_groundtruth_pf = np.abs(self.current_isaac_robot_pose[1] - self.current_mean_pf_robot_pose[1])
                        euclid_dist_groundtruth_pf = np.linalg.norm(np.array(self.current_isaac_robot_pose[:2]) - np.array(self.current_mean_pf_robot_pose[:2]))
                        angular_diff_groundtruth_pf = (self.current_isaac_robot_pose[2] - self.current_mean_pf_robot_pose[2] + np.pi) % (2 * np.pi) - np.pi

                    else:
                        x_dist_groundtruth_pf = None
                        y_dist_groundtruth_pf = None
                        euclid_dist_groundtruth_pf = None
                        angular_diff_groundtruth_pf = None

                print(f'current x_dist_groundtruth_pf:', x_dist_groundtruth_pf)
                print(f'current y_dist_groundtruth_pf:', y_dist_groundtruth_pf)
                print(f'current euclid_dist_groundtruth pf:', euclid_dist_groundtruth_pf)
                print(f'current angular_diff_groundtruth pf:', angular_diff_groundtruth_pf)

            if capture_data_trigger:
                if ros_master_status == 'hm':
                    self.robot_center_camera_groundtruth_pose_list.append(self.current_robot_center_camera_groundtruth_pose)
                else:
                    self.isaac_robot_pose_list.append(self.current_isaac_robot_pose)

                self.odom_pose_list.append(self.current_odom_robot_pose)

                if self.chosen_algorithm == 'amcl':
                    self.amcl_mean_pose_list.append(self.current_amcl_mean_pose)
                    self.amcl_var_list.append(self.current_amcl_var)
                    self.amcl_std_list.append(self.current_amcl_var)

                    self.landmarks_seen_list.append(60)

                if self.chosen_algorithm == 'pf':
                    self.pf_mean_pose_list.append(self.current_mean_pf_robot_pose)
                    self.pf_occ_coverage_list.append(self.current_mean_pf_occ_coverage)
                    self.pf_var_list.append(self.current_pf_var)
                    self.pf_std_list.append(self.current_pf_std)

                    self.landmarks_seen_list.append(self.current_num_landmarks_seen)

                if ros_master_status == 'hm':
                    self.num_reached_poses = len(self.robot_center_camera_groundtruth_pose_list)
                else:
                    self.num_reached_poses = len(self.isaac_robot_pose_list)

                if self.chosen_algorithm == 'amcl':
                    self.groundtruth_amcl_x_pos_diff_list.append(x_dist_groundtruth_amcl)
                    self.groundtruth_amcl_y_pos_diff_list.append(y_dist_groundtruth_amcl)
                    self.groundtruth_amcl_xy_pos_diff_list.append(euclid_dist_groundtruth_amcl)
                    self.groundtruth_amcl_angle_diff_list.append(angular_diff_groundtruth_amcl)

                if self.chosen_algorithm == 'pf':
                    self.groundtruth_pf_x_pos_diff_list.append(x_dist_groundtruth_pf)
                    self.groundtruth_pf_y_pos_diff_list.append(y_dist_groundtruth_pf)
                    self.groundtruth_pf_xy_pos_diff_list.append(euclid_dist_groundtruth_pf)
                    self.groundtruth_pf_angle_diff_list.append(angular_diff_groundtruth_pf)

                print('self.num_target_poses:', self.num_target_poses)
                print('self.num_reached_poses:', self.num_reached_poses)
                print('route_poses:', self.locations_dict_radians)
                print(f'self.isaac_robot_pose_list:', self.isaac_robot_pose_list)
                print(f'self.odom_pose_list:', self.odom_pose_list)

                if self.chosen_algorithm == 'amcl':
                    print(f'self.amcl_mean_pose_list:', self.amcl_mean_pose_list)
                    print(f'self.amcl_var_list:', self.amcl_var_list)
                    print(f'self.amcl_std_list:', self.amcl_std_list)
                    print(f'self.landmarks_seen_list:', self.landmarks_seen_list)
                    print(f'self.groundtruth_amcl_x_pos_diff_list:', self.groundtruth_amcl_x_pos_diff_list)
                    print(f'self.groundtruth_amcl_y_pos_diff_list:', self.groundtruth_amcl_y_pos_diff_list)
                    print(f'self.groundtruth_amcl_xy_pos_diff_list:', self.groundtruth_amcl_xy_pos_diff_list)
                    print(f'self.groundtruth_amcl_angle_diff_list:', self.groundtruth_amcl_angle_diff_list)

                if self.chosen_algorithm == 'pf':
                    print(f'self.pf_mean_pose_list:', self.pf_mean_pose_list)
                    print(f'self.pf_occ_coverage_list:', self.pf_occ_coverage_list)
                    print(f'self.pf_var_list:', self.pf_var_list)
                    print(f'self.pf_std_list:', self.pf_std_list)
                    print(f'self.landmarks_seen_list:', self.landmarks_seen_list)
                    print(f'self.groundtruth_pf_x_pos_diff_list:', self.groundtruth_pf_x_pos_diff_list)
                    print(f'self.groundtruth_pf_y_pos_diff_list:', self.groundtruth_pf_y_pos_diff_list)
                    print(f'self.groundtruth_pf_xy_pos_diff_list:', self.groundtruth_pf_xy_pos_diff_list)
                    print(f'self.groundtruth_pf_angle_diff_list:', self.groundtruth_pf_angle_diff_list)
        
        if self.chosen_algorithm == 'amcl':
            amcl_variances_np = np.array(self.amcl_var_list)

            self.mean_var_x = np.mean(amcl_variances_np[:, 0])
            self.mean_var_y = np.mean(amcl_variances_np[:, 1])
            self.mean_var_theta = np.mean(amcl_variances_np[:, 2])

            amcl_std_np = np.array(self.amcl_std_list)

            self.mean_std_x = np.mean(amcl_std_np[:, 0])
            self.mean_std_y = np.mean(amcl_std_np[:, 1])
            self.mean_std_theta = np.mean(amcl_std_np[:, 2])
            
            # Calculate rmse here
            self.rmse_x_pos = np.sqrt(np.mean(np.array(self.groundtruth_amcl_x_pos_diff_list) ** 2))
            self.rmse_y_pos = np.sqrt(np.mean(np.array(self.groundtruth_amcl_y_pos_diff_list) ** 2))
            self.rmse_xy_pos= np.sqrt(np.mean(np.array(self.groundtruth_amcl_xy_pos_diff_list) ** 2))
            self.rmse_angle = np.sqrt(np.mean(np.array(self.groundtruth_amcl_angle_diff_list) ** 2))

            print(f'Final RMSE X Position: {self.rmse_x_pos:.4f}')
            print(f'Final RMSE Y Position: {self.rmse_y_pos:.4f}')
            print(f'Final RMSE XY Position: {self.rmse_xy_pos:.4f}')
            print(f'Final RMSE Orientation: {self.rmse_angle:.4f} rad')

            self.mae_x_pos = np.mean(np.abs(np.array(self.groundtruth_amcl_x_pos_diff_list)))
            self.mae_y_pos = np.mean(np.abs(np.array(self.groundtruth_amcl_y_pos_diff_list)))
            self.mae_xy_pos = np.mean(np.abs(np.array(self.groundtruth_amcl_xy_pos_diff_list)))
            self.mae_angle = np.mean(np.abs(np.array(self.groundtruth_amcl_angle_diff_list)))

            print(f'Final MAE X Position:', self.mae_x_pos)
            print(f'Final MAE Y Position:', self.mae_y_pos)
            print(f'Final MAE XY Position:', self.mae_xy_pos)
            print(f'Final MAE Orientation:', self.mae_angle)

        if self.chosen_algorithm == 'pf':
            # Calculate mean variance here
            pf_variances_np = np.array(self.pf_var_list)

            self.mean_var_x = np.mean(pf_variances_np[:, 0])
            self.mean_var_y = np.mean(pf_variances_np[:, 1])
            self.mean_var_theta = np.mean(pf_variances_np[:, 2])

            # Calculate mean std here
            pf_std_np = np.array(self.pf_std_list)

            self.mean_std_x = np.mean(pf_std_np[:, 0])
            self.mean_std_y = np.mean(pf_std_np[:, 1])
            self.mean_std_theta = np.mean(pf_std_np[:, 2])
            
            # Calculate rmse here
            self.rmse_x_pos = np.sqrt(np.mean(np.array(self.groundtruth_pf_x_pos_diff_list) ** 2))
            self.rmse_y_pos = np.sqrt(np.mean(np.array(self.groundtruth_pf_y_pos_diff_list) ** 2))
            self.rmse_xy_pos= np.sqrt(np.mean(np.array(self.groundtruth_pf_xy_pos_diff_list) ** 2))
            self.rmse_angle = np.sqrt(np.mean(np.array(self.groundtruth_pf_angle_diff_list) ** 2))

            print(f'Final RMSE X Position: {self.rmse_x_pos:.4f}')
            print(f'Final RMSE Y Position: {self.rmse_y_pos:.4f}')
            print(f'Final RMSE XY Position: {self.rmse_xy_pos:.4f}')
            print(f'Final RMSE Orientation: {self.rmse_angle:.4f} rad')

            self.mae_x_pos = np.mean(np.abs(np.array(self.groundtruth_pf_x_pos_diff_list)))
            self.mae_y_pos = np.mean(np.abs(np.array(self.groundtruth_pf_y_pos_diff_list)))
            self.mae_xy_pos = np.mean(np.abs(np.array(self.groundtruth_pf_xy_pos_diff_list)))
            self.mae_angle = np.mean(np.abs(np.array(self.groundtruth_pf_angle_diff_list)))

            print(f'Final MAE X Position:', self.mae_x_pos)
            print(f'Final MAE Y Position:', self.mae_y_pos)
            print(f'Final MAE XY Position:', self.mae_xy_pos)
            print(f'Final MAE Orientation:', self.mae_angle)

        if capture_data_trigger:
            now = datetime.datetime.now()
            local_now = now.astimezone()
            local_tz = local_now.tzinfo
            local_tzname = local_tz.tzname(local_now)

            if current_iter is not None:
                log_file_name = f'{now.strftime("%Y%m%d_%H%M%S")}_{local_tzname.strip().lower()}_lap_{current_iter}.json'
            else:
                log_file_name = f'{now.strftime("%Y%m%d_%H%M%S")}_{local_tzname.strip().lower()}.json'

            self.write_logging_data(data_file_name=log_file_name)

    def lab_mover_behavior_start(self):
        print('Starting move behavior!!!')

        rospy.Timer(rospy.Duration(self.node_time_interval), self.publish_move_locations_timed_cb)

        self.capture_robot_positions()

        start = 0
        num_iterations = 30

        try:
            for i in range(start, num_iterations):
                print(f'ITERATION - START:', i)

                self.move_behavior(capture_data_trigger=True, current_iter=i)

                print(f'ITERATION - END:', i)
        except Exception as e:
            print(f'Exception:', e)

        print("End of move behavior!!!")

    def start_behavior_cb(self, data):
        data_str = data.data

        if data_str == 'start' and not self.behavior_started:
            self.behavior_started = True

            self.lab_mover_behavior_start()

if __name__ == '__main__':
    lab_mover = hsr_lab_mover()
