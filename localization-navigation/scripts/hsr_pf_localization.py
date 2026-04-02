#!/usr/bin/env python3
"""Particle filter localization for the Toyota HSR robot.

Implements Monte Carlo localization using a particle filter with Numba-accelerated
raycasting. Subscribes to laser scan and odometry data, publishes estimated pose
and particle cloud visualizations to RViz.
"""

import ast
import cv2
import datetime
import json
import scipy
import math
import multiprocessing
import pickle
import time
import os
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.stats import norm
from sklearn.cluster import DBSCAN

from numba import njit, prange

import rospy

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from functools import partial

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from rospkg import RosPack; rp = RosPack()
repo_path = rp.get_path('hsr-localization-navigation')
sys.path.append(repo_path)

from scripts.general_helpers.general_helper import GeneralHelper
from scripts.general_helpers.rviz_helper import RvizHelper
from scripts.general_helpers.tf_helper import TFHelper

ros_master_uri_string = os.environ['ROS_MASTER_URI']
print(f'ros_master_uri_string: {ros_master_uri_string}')

robot_names = ['hsrb.local', 'hsrc.local']

running_on_robot = False
for robot_name in robot_names:
    if robot_name in ros_master_uri_string:
        running_on_robot = True
        break

if running_on_robot:
    ros_master_status = 'hm'
else:
    ros_master_status = 'sm'

print(f'ros_master_status:', ros_master_status)

@njit
def free_cells_to_world(free_cells, resolution, map_origin_x, map_origin_y, cos_y, sin_y):
    n = free_cells.shape[0]
    free_world = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        my, mx = free_cells[i]
        x_map = (mx + 0.5) * resolution
        y_map = (my + 0.5) * resolution
        wx = cos_y * x_map - sin_y * y_map + map_origin_x
        wy = sin_y * x_map + cos_y * y_map + map_origin_y
        free_world[i, 0] = wx
        free_world[i, 1] = wy
    return free_world

@njit
def raycast_particle_numba(particle_pose, raycast_particle_step_size, 
                           raycast_particle_max_raycast_dist, occupancy_map_resolution,
                           laser_angles, laser_max_range,
                           occupancy_map_origin, occupancy_map_yaw,
                           occupancy_grid):
    x_p, y_p, theta_p = particle_pose
    num_angles = len(laser_angles)

    raycast_particle_max_steps = int(raycast_particle_max_raycast_dist / raycast_particle_step_size)

    # raycast_particle_step_size = 0.05
    # raycast_particle_max_steps = int(12.0 / raycast_particle_step_size)

    # print(f'init laser_max_range:', laser_max_range)

    laser_readings = np.full(num_angles, laser_max_range)

    # print(f'init laser_readings:', laser_readings)

    ray_angles = theta_p + laser_angles

    # print(f'ray_angles:', ray_angles)

    # ✅ Precompute trigonometric values
    cos_ray_angles = np.cos(ray_angles)
    sin_ray_angles = np.sin(ray_angles)
    dxs = cos_ray_angles * raycast_particle_step_size
    dys = sin_ray_angles * raycast_particle_step_size

    x_vals = np.full(num_angles, x_p)
    y_vals = np.full(num_angles, y_p)
    active_rays = np.ones(num_angles, dtype=np.bool_)

    cos_y = math.cos(occupancy_map_yaw)
    sin_y = math.sin(occupancy_map_yaw)

    for _ in range(raycast_particle_max_steps):
        if not np.any(active_rays):
            break

        for i in range(num_angles):
            if not active_rays[i]:
                continue

            x_vals[i] += dxs[i]
            y_vals[i] += dys[i]

            dx = x_vals[i] - occupancy_map_origin[0]
            dy = y_vals[i] - occupancy_map_origin[1]

            x_map = cos_y * dx + sin_y * dy
            y_map = -sin_y * dx + cos_y * dy

            grid_x = int(x_map / occupancy_map_resolution)
            grid_y = int(y_map / occupancy_map_resolution)

            # grid_x = int((x_vals[i] - occupancy_map_origin[0]) / occupancy_map_resolution)
            # grid_y = int((y_vals[i] - occupancy_map_origin[1]) / occupancy_map_resolution)

            if grid_x < 0 or grid_x >= occupancy_grid.shape[1] or \
               grid_y < 0 or grid_y >= occupancy_grid.shape[0]:
                active_rays[i] = False
                continue

            # print(f'occupancy_grid[grid_x, grid_y]:', occupancy_grid[grid_y, grid_x])

            if occupancy_grid[grid_y, grid_x] == 100:
                # ✅ Use math.hypot for better performance & precision
                laser_readings[i] = math.hypot(x_vals[i] - x_p, y_vals[i] - y_p)
                active_rays[i] = False

    # print(f'returned laser_readings:', laser_readings)

    return laser_readings

# Measurement Model
@njit
def process_particle_raycast_numba(elem, laser_ranges, 
                                   raycast_particle_step_size, raycast_particle_max_raycast_dist,
                                   occupancy_map_resolution, laser_angles, 
                                   laser_max_range, occupancy_map_origin, 
                                   occupancy_map_yaw, occupancy_grid, 
                                   measurement_weight_sigma, measurement_weight_epsilon):
    # expected = raycast_particle_numba(elem, occupancy_map_resolution, laser_angles, laser_max_range, occupancy_map_origin, occupancy_map_yaw, occupancy_grid)
    
    expected = raycast_particle_numba(particle_pose=elem, 
                                      raycast_particle_step_size=raycast_particle_step_size, 
                                      raycast_particle_max_raycast_dist=raycast_particle_max_raycast_dist, 
                                      occupancy_map_resolution=occupancy_map_resolution,
                                      laser_angles=laser_angles,
                                      laser_max_range=laser_max_range,
                                      occupancy_map_origin=occupancy_map_origin,
                                      occupancy_map_yaw=occupancy_map_yaw,
                                      occupancy_grid=occupancy_grid)

    observed = laser_ranges

    # print(f'particle_pose:', elem)
    # print(f'observed:', observed)
    # print(f'expected:', expected)

    diff = observed - expected

    # print(f'diff:', diff)

    sigma = measurement_weight_sigma
    epsilon = measurement_weight_epsilon

    # sigma = 0.1
    # epsilon = 1
    # epsilon = 0.01

    # print(f'sigma:', sigma)
    # print(f'epsilon:', epsilon)

    likelihoods = np.exp((-1 * (diff ** 2)) / (2 * (sigma ** 2))) + epsilon

    # print(f'likelihoods:', likelihoods)

    weight = np.prod(likelihoods)
    return weight

@njit(parallel=True, fastmath=True)
def compute_weights_numba(X_new, laser_ranges,
                          raycast_particle_step_size, raycast_particle_max_raycast_dist,
                          occupancy_map_resolution, laser_angles, 
                          laser_max_range, occupancy_map_origin, 
                          occupancy_map_yaw, occupancy_grid,
                          measurement_weight_sigma, measurement_weight_epsilon):

    # print(f'IN COMPUTE WEIGHTS NUMBA!')
    weights = np.zeros(len(X_new))

    for i in prange(len(X_new)):
        weights[i] = process_particle_raycast_numba(elem=X_new[i], 
                                                    laser_ranges=laser_ranges, 
                                                    raycast_particle_step_size=raycast_particle_step_size,
                                                    raycast_particle_max_raycast_dist=raycast_particle_max_raycast_dist,
                                                    occupancy_map_resolution=occupancy_map_resolution,
                                                    laser_angles=laser_angles, 
                                                    laser_max_range=laser_max_range, 
                                                    occupancy_map_origin=occupancy_map_origin,
                                                    occupancy_map_yaw=occupancy_map_yaw,
                                                    occupancy_grid=occupancy_grid,
                                                    measurement_weight_sigma=measurement_weight_sigma,
                                                    measurement_weight_epsilon=measurement_weight_epsilon)
        
    # print(f'numba_weights:', weights)

    return weights

class hsr_pf_localization:
    def establish_rviz_pubs(self):
        self.odom_marker_pub = rospy.Publisher(f'{self.class_name}/odom/marker', 
                                               Marker,
                                               queue_size=10)
        
        self.isaac_sim_robot_pose_pub = rospy.Publisher(f'{self.class_name}/isaac_sim_robot_pose/marker',
                                                        Marker,
                                                        queue_size=10)
        
        self.global_laser_marker_array_pub = rospy.Publisher(f'{self.class_name}/base_scan_global/points/markers',
                                                             MarkerArray,
                                                             queue_size=10)
        self.current_global_laser_markers = None

        self.pf_sensor_data_marker_array_pub = rospy.Publisher(f'{self.class_name}/pf_sensor_data/markers',
                                                               MarkerArray,
                                                               queue_size=10)
        
        self.current_pf_sensor_data_markers = None

        self.pf_sensor_data_num_landmarks_seen_pub = rospy.Publisher(f'{self.class_name}/pf_sensor_data/num_landmarks_seen',
                                                                 String,
                                                                 queue_size=10)

        self.pf_particles_marker_array_pub = rospy.Publisher(f'{self.class_name}/pf_particles/markers',
                                                             MarkerArray,
                                                             queue_size=10)
        
        self.pf_particles_mean_marker_pub = rospy.Publisher(f'{self.class_name}/pf_particles_mean/marker',
                                                            Marker,
                                                            queue_size=10)
        
        self.pf_particles_mean_raycast_occ_coverage_pub = rospy.Publisher(f'{self.class_name}/pf_particles_mean_raycast_occ_coverage',
                                                                  String,
                                                                  queue_size=10)
        
        self.pf_particles_mean_live_scan_occ_coverage_pub = rospy.Publisher(f'{self.class_name}/pf_particles_mean_live_scan_occ_coverage',
                                                                            String,
                                                                            queue_size=10)
        
        self.pf_particles_convergence_time_pub = rospy.Publisher(f'{self.class_name}/pf_particles_convergence_time',
                                                                     String,
                                                                     queue_size=10)
        
        self.pf_particles_variance_pub = rospy.Publisher(f'{self.class_name}/pf_particles_variance',
                                                         String,
                                                         queue_size=10)
        
        self.pf_particles_std_pub = rospy.Publisher(f'{self.class_name}/pf_particles_std',
                                                    String,
                                                    queue_size=10)
        
        self.pf_particles_endpoints_marker_array_pub = rospy.Publisher(f'{self.class_name}/pf_particles_endpoints/markers',
                                                            MarkerArray,
                                                            queue_size=10)

    def configure_node_rate(self, hz):
        self.node_rate = hz
        self.node_time_interval = 1/self.node_rate

    def init_timer(self):
        self.current_time_secs = 0
        self.timer_callback_count = 1

    def init_odom_data(self):
        self.current_odom_data = None
        self.prev_odom_data = None

    def init_pf_data(self):
        print(f'INIT PF_DATA!')
        self.X = np.empty((0,0))
        self.mean = None

        self.anchor_particles = None
        self.anchor_particle_start_index = None
        self.num_anchor_particles = None

        self.pf_first_time = True
        self.pf_kidnap_trigger = True

        self.particle_filter_iter = 0

        self.current_laser_data = None

        self.processed_occupancy_data = False

        self.odom_particle = None
        self.adjusted_sensor_offset = None

        self.current_robot_center_camera_groundtruth_pose = None

        self.kidnap_start_time = None
        self.convergence_time = None
        self.converged = False

        self.kidnap_correct_position, self.kidnap_correct_orientation = None, None
        self.reached_kidnap_correct_pose = False

    def init_launch_parameters(self):
        # Parameters for adjusting pf occupancy grid processing behavior
        self.occupied_threshold = rospy.get_param('~occupied_threshold', None)

        if self.occupied_threshold is not None: self.occupied_threshold = float(self.occupied_threshold)
        else:
            print(f'self.occupied_threshold not set through launch file!')
            exit()

        # Parameters for adjusting pf laser scan behavior
        self.laser_scan_num_factor = rospy.get_param('~laser_scan_num_factor', None)

        if self.laser_scan_num_factor is not None: 
            self.laser_scan_num_factor = float(self.laser_scan_num_factor)
            if self.laser_scan_num_factor > 1:
                self.laser_scan_num_factor = 1
        else:
            print(f'self.laser_scan_num_factor not set through launch file!')
            exit()

        # Parameter for adjusting pf num particles
        self.num_tracking_particles = rospy.get_param('~num_tracking_particles', None)

        if self.num_tracking_particles is not None: self.num_tracking_particles = int(self.num_tracking_particles)
        else: 
            print(f'self.num_tracking_particles not set through launch file!')
            exit()

        # Parameters adjusting characteristics of the measurement model using numba.
        self.raycast_particle_step_size = rospy.get_param('~raycast_particle_step_size', None)
        self.raycast_particle_max_raycast_dist = rospy.get_param('~raycast_particle_max_raycast_dist', None)
        
        if self.raycast_particle_step_size is not None: self.raycast_particle_step_size = float(self.raycast_particle_step_size)
        else: 
            print(f'self.raycast_particle_step_size not set through launch file!')
            exit()
        if self.raycast_particle_max_raycast_dist is not None: self.raycast_particle_max_raycast_dist = float(self.raycast_particle_max_raycast_dist)
        else: 
            print(f'self.raycast_particle_max_raycast_dist not set through launch file!')
            exit()

        # Parameters adjusting gaussian noise in received odom estimate
        self.odom_gaussian_noise_mean = rospy.get_param('~odom_gaussian_noise_mean', None)
        self.odom_gaussian_noise_stdev = rospy.get_param('~odom_gaussian_noise_stdev', None)

        if self.odom_gaussian_noise_mean is not None: self.odom_gaussian_noise_mean = float(self.odom_gaussian_noise_mean)
        else: 
            print(f'self.odom_gaussian_noise_mean not set through launch file!')
            exit()
        if self.odom_gaussian_noise_stdev is not None: self.oddom_gaussian_noise_stdev = float(self.odom_gaussian_noise_stdev)
        else: 
            print(f'self.gaussian_noise_stdev is not set through launch file!')
            exit()

        # Parameters adjusting characteristics of the sample motion model
        self.sample_motion_model_x_factor = rospy.get_param('~sample_motion_model_x_factor', None)
        self.sample_motion_model_y_factor = rospy.get_param('~sample_motion_model_y_factor', None)
        self.sample_motion_model_theta_factor = rospy.get_param('~sample_motion_model_theta_factor', None)
        self.sample_motion_model_mu = rospy.get_param('~sample_motion_model_mu', None)
        self.sample_motion_model_sigma = rospy.get_param('~sample_motion_model_sigma', None)

        if self.sample_motion_model_x_factor is not None: self.sample_motion_model_x_factor = float(self.sample_motion_model_x_factor)
        else: 
            print(f'self.sample_motion_model_x_factor not set through launch file!')
            exit()
        if self.sample_motion_model_y_factor is not None: self.sample_motion_model_y_factor = float(self.sample_motion_model_y_factor)
        else: 
            print(f'self.sample_motion_model_y_factor not set through launch file!')
            exit()
        if self.sample_motion_model_theta_factor is not None: self.sample_motion_model_theta_factor = float(self.sample_motion_model_theta_factor)
        else: 
            print(f'self.sample_motion_model_theta_factor is not set through launch file!')
            exit()
        if self.sample_motion_model_mu is not None: self.sample_motion_model_mu = float(self.sample_motion_model_mu)
        else: 
            print(f'self.sample_motion_model_mu is not set through launch file!')
            exit()
        if self.sample_motion_model_sigma is not None: self.sample_motion_model_sigma = float(self.sample_motion_model_sigma)
        else: 
            print(f'self.sample_motion_model_sigma is not set through launch file!')
            exit()

        # Parameters adjusting characteristics of the measurement model
        self.measurement_weight_sigma = rospy.get_param('~measurement_weight_sigma', None)
        self.measurement_weight_epsilon = rospy.get_param('~measurement_weight_epsilon', None)

        if self.measurement_weight_sigma is not None: self.measurement_weight_sigma = float(self.measurement_weight_sigma)
        else: 
            print(f'self.measurement_weight_sigma is not set through launch file!')
            exit()
        if self.measurement_weight_epsilon is not None: self.measurement_weight_epsilon = float(self.measurement_weight_epsilon)
        else: 
            print(f'self.measurement_weight_epsilong is not set through launch file!')
            exit()

        # Parameters adjusting pf global kidnapping
        self.global_kidnapping_num_particles = rospy.get_param('~global_kidnapping_num_particles', None)
        self.global_kidnapping_num_particle_angles = rospy.get_param('~global_kidnapping_num_particle_angles', None)
        self.global_kidnapping_max_orientation = rospy.get_param('~global_kidnapping_max_orientation', None)
        self.global_kidnapping_occ_map_xmin = rospy.get_param('~global_kidnapping_occ_map_xmin', None)
        self.global_kidnapping_occ_map_xmax = rospy.get_param('~global_kidnapping_occ_map_xmax', None)
        self.global_kidnapping_occ_map_ymin = rospy.get_param('~global_kidnapping_occ_map_ymin', None)
        self.global_kidnapping_occ_map_ymax = rospy.get_param('~global_kidnapping_occ_map_ymax', None)

        if self.global_kidnapping_num_particles is not None: self.global_kidnapping_num_particles = float(self.global_kidnapping_num_particles)
        else: 
            print(f'self.global_kidnapping_num_particles is not set through launch file!')
            exit()
        if self.global_kidnapping_num_particle_angles is not None: self.global_kidnapping_num_particle_angles = int(self.global_kidnapping_num_particle_angles)
        else: 
            print(f'self.global_kidnapping_num_particle_angles it not set through launch file!')
            exit()
        if self.global_kidnapping_max_orientation is not None: self.global_kidnapping_max_orientation = float(self.global_kidnapping_max_orientation)
        else: 
            print(f'self.global_kidnapping_max_orientation is not set through launch file!')
            exit()
        if self.global_kidnapping_occ_map_xmin is not None: self.global_kidnapping_occ_map_xmin = float(self.global_kidnapping_occ_map_xmin)
        else: 
            print(f'self.global_kidnapping_occ_map_xmin is not set through launch file!')
            exit()
        if self.global_kidnapping_occ_map_xmax is not None: self.global_kidnapping_occ_map_xmax = float(self.global_kidnapping_occ_map_xmax)
        else: 
            print(f'self.global_kidnapping_occ_map_xmax is not set through launch file!')
            exit()
        if self.global_kidnapping_occ_map_ymin is not None: self.global_kidnapping_occ_map_ymin = float(self.global_kidnapping_occ_map_ymin)
        else: 
            print(f'self.global_kidnapping_occ_map_ymin is not set through launch file!')
            exit()
        if self.global_kidnapping_occ_map_ymax is not None: self.global_kidnapping_occ_map_ymax = float(self.global_kidnapping_occ_map_ymax)
        else: 
            print(f'self.global_kidnapping_occ_map_ymax is not set through launch file!')
            exit()

        # Parameter for enabling odom correction via TMC 2D Pose Estimate.
        self.enable_odom_correction = rospy.get_param('~enable_odom_correction', None)

        if self.enable_odom_correction is not None: self.enable_odom_correction = bool(self.enable_odom_correction)
        else: 
            print(f'self.enable_odom_correction is not set through launch file!')
            exit()

    def read_occupancy_map(self, map_dir_path):
        map_file_path = os.path.join(map_dir_path, 'map.pgm')
        yaml_file_path = os.path.join(map_dir_path, 'map.yaml')

        # Load the YAML file to get resolution, origin, and thresholds
        with open(yaml_file_path, 'r') as file:
            map_info = yaml.safe_load(file)

        self.occupancy_map_resolution = map_info['resolution']
        self.occupancy_map_origin = map_info['origin']
        
        # Load the thresholds from the YAML file
        occupied_thresh = map_info['occupied_thresh']  # between 0.0 and 1.0
        free_thresh = map_info['free_thresh']          # between 0.0 and 1.0
        negate = map_info['negate']                    # inversion flag for the map
        
        # Convert thresholds to pixel values (0-255 range)
        occupied_pixel_thresh = int(occupied_thresh * 255)
        free_pixel_thresh = int(free_thresh * 255)

        # Load the map image (PGM format)
        map_img = cv2.imread(map_file_path, cv2.IMREAD_GRAYSCALE)
        map_height, map_width = map_img.shape

        print(f'map_height: {map_height}')
        print(f'map_width: {map_width}')
        print(f'map_img type: {type(map_img)}')

        # Create the occupancy grid (0 for free, 1 for occupied, -1 for unknown)
        self.occupancy_grid = np.full((map_height, map_width), -1, dtype=np.int8)  # Initialize as unknown (-1)
        
        # Handle map inversion if needed
        if negate == 0:
            # No inversion: 0 = occupied, 255 = free
            self.occupancy_grid[map_img <= occupied_pixel_thresh] = 1  # Occupied
            self.occupancy_grid[map_img >= free_pixel_thresh] = 0      # Free
        else:
            # Inverted: 0 = free, 255 = occupied
            self.occupancy_grid[map_img <= free_pixel_thresh] = 0      # Free
            self.occupancy_grid[map_img >= occupied_pixel_thresh] = 1  # Occupied

        return self.occupancy_grid

    def find_occupied_boundaries_in_world(self):
        """
        This function finds the xmin, xmax, ymin, ymax boundaries of the occupied cells
        in the occupancy grid.
        Returns:
            xmin, xmax, ymin, ymax (in pixel coordinates)
        """
        # Get indices of occupied cells (where the grid value is 1)
        occupied_indices = np.argwhere(self.occupancy_grid > 0)

        print(f'occupied_indices.size:', occupied_indices.size)
        
        if occupied_indices.size == 0:
            # If there are no occupied cells, return None or raise an exception
            print("No occupied cells found.")
            return None
        
        # Find xmin, xmax, ymin, ymax from the occupied indices
        ymin = np.min(occupied_indices[:, 0])  # Minimum row (y-axis in pixel space)
        ymax = np.max(occupied_indices[:, 0])  # Maximum row (y-axis in pixel space)
        xmin = np.min(occupied_indices[:, 1])  # Minimum column (x-axis in pixel space)
        xmax = np.max(occupied_indices[:, 1])  # Maximum column (x-axis in pixel space)

        print(f'xmin: {xmin}')
        print(f'xmax: {xmax}')
        print(f'ymin: {ymin}')
        print(f'ymax: {ymax}')

        # Convert pixel coords to world coords.
        self.occ_xmin_world = xmin * self.occupancy_map_resolution + self.occupancy_map_origin[0]
        self.occ_xmax_world = xmax * self.occupancy_map_resolution + self.occupancy_map_origin[0]
        self.occ_ymin_world = ymin * self.occupancy_map_resolution + self.occupancy_map_origin[1]
        self.occ_ymax_world = ymax * self.occupancy_map_resolution + self.occupancy_map_origin[1]

        self.get_center_occupancy_grid_in_world()

        # Coordinate system is flipped on y for world.
        # self.occ_ymin_world = -self.occ_ymin_world
        # self.occ_ymax_world = -self.occ_ymax_world

        print(f'xmin_world: {self.occ_xmin_world}')
        print(f'xmax_world: {self.occ_xmax_world}')
        print(f'ymin_world: {self.occ_ymin_world}')
        print(f'ymax_world: {self.occ_ymax_world}')

        # Get occupancy region width and height
        self.occ_width_world = self.occ_xmax_world + self.occ_xmin_world
        self.occ_height_world = self.occ_ymin_world + self.occ_ymax_world

        self.occ_radius_world = (self.occ_width_world / 2) if self.occ_width_world >= self.occ_height_world else (self.occ_height_world / 2)

        print(f'width_world: {self.occ_width_world}')
        print(f'height_world: {self.occ_height_world}')

    def get_center_occupancy_grid_in_world(self):
        self.occ_center_x_world = (self.occ_xmin_world + self.occ_xmax_world) / 2
        self.occ_center_y_world = (self.occ_ymin_world + self.occ_ymax_world) / 2

        print(f'center_x_world: {self.occ_center_x_world}')
        print(f'center_y_world: {self.occ_center_y_world}')

    def establish_pose_pubs(self):
        if ros_master_status == 'hm':
            # self.base_pose_pub = rospy.Publisher('/amcl_pose', PoseWithCovarianceStamped, queue_size=10)
            self.base_pose_pub = rospy.Publisher('/laser_2d_correct_pose', PoseWithCovarianceStamped, queue_size=10)
        else:
            self.base_pose_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)

    def __init__(self, record_logging_data=False):
        rospy.init_node('hsr_pf_localization', anonymous=True)

        self.general_helper = GeneralHelper()
        self.rviz_helper = RvizHelper()
        self.tf_helper = TFHelper()

        self.class_name = self.__class__.__name__

        self.establish_rviz_pubs()
        self.establish_pose_pubs()

        # if ros_master_status == 'hm':
        #     self.map_dir_path = os.path.join(repo_path, 'maps', 'caneslab', 'raw')
        # else:
        #     self.map_dir_path = os.path.join(repo_path, 'maps', 'isaac_rc_lab_uyen', 'raw')
        
        # self.map_yaml_path = os.path.join(self.map_dir_path, 'map.yaml')
        # self.map_yaml_data = self.general_helper.read_yaml_file(data_file_path=self.map_yaml_path)

        # self.map_resolution = self.map_yaml_data['resolution']
        # print(f'map_resolution:', self.map_resolution)

        # self.map_origin = self.map_yaml_data['origin']
        # print(f'map_origin: {self.map_origin}')
        
        # self.read_occupancy_map(map_dir_path=self.map_dir_path)
        # print(f'self.occupancy_grid: {self.occupancy_grid}')
        # print(f'self.occupancy_grid shape: {np.shape(self.occupancy_grid)}')
        # print(f'self.occupancy_grid type: {type(self.occupancy_grid)}')

        # self.find_occupied_boundaries_in_world()
        # self.get_center_occupancy_grid_in_world()

        self.configure_node_rate(hz=60)
        self.init_timer()

        self.init_odom_data()
        self.init_pf_data()

        self.init_launch_parameters()

        if ros_master_status == 'sm':
            print(f'\nListening for start signal...\n')

            self.behavior_started = False

            rospy.Subscriber('/isaac_sim/robot_behavior_start', String, self.start_behavior_cb)
            rospy.spin()

    def start_behavior_cb(self, data):
        data_str = data.data
        # print(f'received data: {data_str}')

        if data_str == 'start' and not self.behavior_started:
            self.behavior_started = True

            print('Starting behavior!!!')

            self.run_nodes()
            
            print("End of behavior!!!")

    def clock_cb(self, clock_data):
        # print(f'clock_data: {clock_data}')

        clock_secs = clock_data.clock.secs
        # print(f'clock_secs: {clock_secs}')

        if self.current_time_secs:
            if clock_secs > self.current_time_secs:
                print(f'A second has passed! - {clock_secs}')
                self.timer_callback_count = 1

        self.current_time_secs = clock_secs

    def occupancy_grid_cb(self, occupancy_grid_data):
        # print(f'occupancy_grid_data:', occupancy_grid_data)
        # print(f'occupancy_grid data type:', type(occupancy_grid_data))

        self.occupancy_map_resolution = occupancy_grid_data.info.resolution
        self.occupancy_map_origin = np.array([occupancy_grid_data.info.origin.position.x, occupancy_grid_data.info.origin.position.y])
        self.occupancy_map_width, self.occupancy_map_height = occupancy_grid_data.info.width, occupancy_grid_data.info.height
        
        q = occupancy_grid_data.info.origin.orientation
        self.occupancy_map_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        print(f'self.occupancy_map_resolution:', self.occupancy_map_resolution)
        print(f'self.occupancy_map_origin:', self.occupancy_map_origin)
        print(f'self.occupancy_map_width, self.occupancy_map_height:', (self.occupancy_map_width, self.occupancy_map_height))

        self.occupancy_grid = np.array(occupancy_grid_data.data, dtype=np.int8).reshape((self.occupancy_map_height, self.occupancy_map_width))
        print(f'self.occupancy_grid_data:', self.occupancy_grid)

        self.find_occupied_boundaries_in_world()

        self.processed_occupancy_data = True

    def get_laser_data(self):
        # Taken from HSR laser data frame.
        self.laser_angle_min = self.current_laser_data.angle_min
        self.laser_angle_max = self.current_laser_data.angle_max
        self.laser_angle_increment = self.current_laser_data.angle_increment
        # print(f'self.laser_angle_increment:', self.laser_angle_increment)
        self.laser_time_increment = self.current_laser_data.time_increment
        self.laser_scan_time = self.current_laser_data.scan_time
        self.laser_min_range = self.current_laser_data.range_min
        self.laser_max_range = self.current_laser_data.range_max

        self.laser_ranges = np.array(self.current_laser_data.ranges)
        self.laser_intensities = self.current_laser_data.intensities

        self.laser_angles = np.arange(self.laser_angle_min, self.laser_angle_max, self.laser_angle_increment)

        total_laser_ranges = len(self.laser_ranges)

        target_num_points = int(total_laser_ranges * self.laser_scan_num_factor)

        # target_num_points = total_laser_ranges
        # target_num_points = int(total_laser_ranges / 2)
        # target_num_points = 640
        # target_num_points = 480
        # target_num_points = 360
        # target_num_points = 240
        # target_num_points = 120
        # target_num_points = 60
        # target_num_points = 30

        # print(f'total_laser_ranges:', total_laser_ranges)
        # print(f'target_num_points:', target_num_points)

        sampled_indices = np.linspace(0, total_laser_ranges - 1, target_num_points, dtype=int)
        # print(f'sampled_indices:', sampled_indices)

        self.laser_ranges = self.laser_ranges[sampled_indices]
        # print(f'self.laser_ranges:', self.laser_ranges)
        # print(f'len(self.laser_ranges):', len(self.laser_ranges))

        self.laser_min_range = min(self.laser_ranges)
        self.laser_max_range = max(self.laser_ranges)

        self.laser_angles = self.laser_angles[sampled_indices]
        # print(f'self.laser_angles:', self.laser_angles)

        self.laser_angle_min = min(self.laser_angles)
        self.laser_angle_max = max(self.laser_angles)

        # laser_angle_range = self.current_laser_data.angle_max - self.current_laser_data.angle_min
        # print(f'laser_angle_range:', laser_angle_range)

        # target_angle_increment = laser_angle_range / target_num_points
        # print(f'target_angle_increment:', target_angle_increment)

        # self.laser_angles = np.arange(self.laser_angle_min, self.laser_angle_max, self.laser_angle_increment)

        # print(f'self.laser_angles:', self.laser_angles)
        # print(f'len(self.laser_angles):', len(self.laser_angles))
        # print(f'len(self.laser_angles):', len(self.laser_angles))

        # self.laser_angle_min = -2.094395160675049  # Starting angle in radians
        # self.laser_angle_max = 2.076941728591919   # Ending angle in radians
        # self.laser_angle_increment = 0.01745329238474369  # Angle increment in radians
        
        # self.laser_min_range =  0.30000001192092896
        # self.laser_max_range = 60.0

        # self.laser_angles = np.arange(self.laser_angle_min, self.laser_angle_max, self.laser_angle_increment)
        # print(f'self.laser_angles:', self.laser_angles)
        # print(f'len(self.laser_angles):', len(self.laser_angles))

    def laser_scan_cb(self, laser_data):
        self.current_laser_data = laser_data
        # print(f'current laser scan data: {self.current_laser_data}')

        # self.init_laser_data()

        # if self.pf_first_time:
        #     self.init_laser_data()
        #     pass
            
        # print(f'ranges:', self.current_laser_data.ranges)
        # print(f'len(ranges):', len(self.current_laser_data.ranges))

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

        # self.move_base_pose(desired_pose_position=robot_isaac_position, desired_pose_orientation=robot_isaac_orientation)

    def get_transform(self, target_frame, source_frame):
        try:
            now = rospy.Time.now()
            self.tf_helper.tf_listener.waitForTransform(target_frame=target_frame,
                                              source_frame=source_frame,
                                              time=rospy.Time(0),
                                              timeout=rospy.Duration(1.0))
            
            (trans, rot) = self.tf_helper.tf_listener.lookupTransform(target_frame=target_frame,
                                                            source_frame=source_frame,
                                                            time=rospy.Time(0))
            
            return trans, rot
        
        except Exception as e:
            rospy.logwarn(f'Failed to get transform from {source_frame} to {target_frame}. Exception: {e}')
            return None, None

    def get_robot_center_camera_groundtruth_timed_cb(self, event):
        if ros_master_status == 'hm':
            self.robot_center_camera_groundtruth_trans, self.robot_center_camera_groundtruth_rot = self.get_transform(target_frame='map', source_frame='robot_center_camera_groundtruth')

            if self.robot_center_camera_groundtruth_rot is not None:
                self.robot_center_camera_groundtruth_rot = euler_from_quaternion(quaternion=self.robot_center_camera_groundtruth_rot)

            # print(f'self.robot_center_camera_groundtruth_trans:', self.robot_center_camera_groundtruth_trans)
            # print(f'self.robot_center_camera_grountruth_rot:', self.robot_center_camera_groundtruth_rot)

            if self.robot_center_camera_groundtruth_trans is not None and self.robot_center_camera_groundtruth_rot is not None:
                self.current_robot_center_camera_groundtruth_pose = [self.robot_center_camera_groundtruth_trans[0], self.robot_center_camera_groundtruth_trans[1], self.robot_center_camera_groundtruth_rot[2]]
            else:
                self.current_robot_center_camera_groundtruth_pose = None
            
            # print(f'self.robot_center_camera_groundtruth_pose:', self.current_robot_center_camera_groundtruth_pose)

    def establish_odom_marker(self, odom_position_euler):
        odom_marker = self.rviz_helper.generate_marker(ns='odom',
                                                frame_id='map',
                                                marker_type=Marker.ARROW,
                                                marker_action=Marker.ADD,
                                                marker_id=0,
                                                marker_scale=(0.25, 0.05, 0.05),
                                                marker_color=(0, 0, 1, 1),
                                                marker_position=[odom_position_euler[0], odom_position_euler[1], 0],
                                                marker_orientation=[0, 0, odom_position_euler[2]])
            
        self.odom_marker_pub.publish(odom_marker)

    def add_gaussian_noise(self, data, mean, std_dev):
        data_shape = np.shape(data)
        # print(f'data_shape: {data_shape}')

        if std_dev == 0:
            noise = 0
        else:
            noise = np.random.normal(mean, std_dev, size=data_shape)

        return data + noise


    def get_odom_data(self):
        # This is the real odom position of the robot. Not subject to the error discrepancy of the odometry frame from map. 
        try:
            if self.current_odom_data is not None:
                self.prev_odom_data = self.current_odom_data.copy()

            # self.current_odom_data = None

            self.tf_helper.tf_listener.waitForTransform(target_frame='map',
                                            source_frame='base_footprint',
                                            time=rospy.Time(0),
                                            timeout=rospy.Duration(1))

            odom_trans, odom_rot = self.tf_helper.tf_listener.lookupTransform(target_frame='map',
                                             source_frame='base_footprint',
                                             time=rospy.Time(0))

            # print(f'base_footprint position: {odom_trans}')
            # print(f'base_footprint_rotation: {odom_rot}')

            odom_position = np.array(odom_trans)
            # print(f'odom_position: {odom_position}')

            odom_orientation = np.array(odom_rot)
            # print(f'odom_orientation: {odom_orientation}')

            odom_orientation_euler = np.array(euler_from_quaternion(odom_orientation))
            # print(f'odom_orientation_euler: {odom_orientation_euler}')

            odom_position_euler = np.array((odom_position[0], odom_position[1], odom_orientation_euler[2]))
            # print(f'odom_position_euler: {odom_position_euler}')

            self.establish_odom_marker(odom_position_euler=odom_position_euler)

            if ros_master_status == 'hm': 
                # print(f'self.odom_gaussian_noise_mean:', self.odom_gaussian_noise_mean)
                # print(f'self.odom_gaussian_noise_stdev:', self.odom_gaussian_noise_stdev)

                if self.odom_gaussian_noise_mean is not None and self.odom_gaussian_noise_stdev is not None:
                    odom_position_euler = self.add_gaussian_noise(data=odom_position_euler,
                                                                mean=float(self.odom_gaussian_noise_mean),
                                                                std_dev=float(self.odom_gaussian_noise_stdev))
                else:
                   odom_position_euler = self.add_gaussian_noise(data=odom_position_euler,
                                                                mean=0,
                                                                std_dev=0)
            else:
                if self.odom_gaussian_noise_mean is not None and self.odom_gaussian_noise_stdev is not None:
                    odom_position_euler = self.add_gaussian_noise(data=odom_position_euler,
                                                                mean=float(self.odom_gaussian_noise_mean),
                                                                std_dev=float(self.odom_gaussian_noise_stdev))
                else:
                    odom_position_euler = self.add_gaussian_noise(data=odom_position_euler,
                                                                    mean=0,
                                                                    std_dev=0.002)

            self.current_odom_data = odom_position_euler
            self.odom_particle = np.array([odom_position_euler[0], odom_position_euler[1], odom_position_euler[2]])

            # print(f'self.odom_particle:', self.odom_particle)

            self.odom_diff = []
            if self.current_odom_data is not None and self.prev_odom_data is not None:
                joined_data = zip(self.current_odom_data, self.prev_odom_data)
                for a,b in joined_data:
                    result = a-b
                    result = round(result, 3)
                    self.odom_diff.append(result)

            # print(f'self.current_odom_data: {self.current_odom_data}')
            # print(f'self.prev_odom_data: {self.prev_odom_data}')
            # print(f'self.odom_diff: {self.odom_diff}')

        except Exception as e:
            print(f'Exception: {e}')

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

    def establish_global_laser_points(self):
        global_laser_points = []
        if self.current_global_laser_markers:
            self.delete_all_markers(ns='global_laser_points')
                
        laser_markers = []

        # print(f'len(self.laser_ranges):', len(self.laser_ranges))
        # print(f'len(self.laser_angles):', len(self.laser_angles))

        for i in range(len(self.laser_ranges)):
            # print(f'current_index:', i)
            # print(f'len(self.laser_ranges):', len(self.laser_ranges))
            # print(f'len(self.laser_angles):', len(self.laser_angles))

            current_range = self.laser_ranges[i]
            current_angle= self.laser_angles[i]

            local_x = current_range * math.cos(current_angle)
            local_y = current_range * math.sin(current_angle)

            local_x = float('{:.3f}'.format(local_x))
            local_y = float('{:.3f}'.format(local_y))

            local_point_position = (local_x, local_y, 0)

            try:
                self.tf_helper.tf_listener.waitForTransform(target_frame='map',
                                            source_frame='base_range_sensor_link',
                                            time=rospy.Time(0),
                                            timeout=rospy.Duration(1))
                
                global_laser_point_position = self.tf_helper.transform_point(point_position=local_point_position, 
                                                        source_frame='base_range_sensor_link',
                                                        target_frame='map')
                
                # print(f'global_laser_point_position: {global_laser_point_position}')
                
                global_laser_points.append(global_laser_point_position)

                laser_marker = self.rviz_helper.generate_marker(ns='global_laser_points',
                                            frame_id='map',
                                            marker_type=Marker.CUBE,
                                            marker_action=Marker.ADD,
                                            marker_id=i,
                                            marker_scale=(0.02, 0.02, 0.02),
                                            marker_color=(1, 1, 1, 1),
                                            marker_position=global_laser_point_position)
                
                # print(f'laser_marker:\n {laser_marker}')

                laser_markers.append(laser_marker)
            except Exception as e:
                print(f'Exception: {e}')
                pass

        self.current_global_laser_points = global_laser_points

        self.current_global_laser_markers = MarkerArray(laser_markers)

        try:
            self.global_laser_marker_array_pub.publish(self.current_global_laser_markers)

            # print(f'Published global_laser_markers!')
        except Exception as e:
            pass

        return global_laser_points
    
    def calculate_dist_angle(self, base_coords, point_coords):
        displacement_x = point_coords[0] - base_coords[0]
        displacement_y = point_coords[1] - base_coords[1]
        dist = math.sqrt(displacement_x ** 2 + displacement_y ** 2)

        base_yaw = base_coords[2]

        angle_abs = math.atan2(displacement_y, displacement_x)

        angle_rel =  angle_abs - base_yaw

        return dist, angle_abs, angle_rel

    def process_laser_data(self):
        # self.global_laser_points = self.establish_global_laser_points(laser_data_angle_min=self.laser_angle_min, 
        #                                                               laser_data_angle_increment=self.laser_angle_increment, 
        #                                                               laser_data_ranges=self.laser_ranges)

        self.global_laser_points = self.establish_global_laser_points()

        # print(f'global_laser_points: {global_laser_points}')

        if not self.global_laser_points:
            print(self.general_helper.trans_sep)
            print(f'No global_laser_points! Waiting for next cycle...\n')
            print(self.general_helper.trans_sep)
            return
        
        det_data = []
        pf_sensor_data_markers = []

        if self.current_pf_sensor_data_markers:
            self.delete_all_markers(ns='pf_sensor_data')
        
        for i, global_laser_point in enumerate(self.global_laser_points):
            dist, angle_abs, angle_rel = self.calculate_dist_angle(base_coords=self.current_odom_data, point_coords=global_laser_point)

            det_data_tuple = (i, dist, angle_rel)

            # print(f'det_data_tuple: {det_data_tuple}')

            det_data.append(det_data_tuple)

            # print(f'self.current_odom_data: {self.current_odom_data}')

            origin_arrow_point = [self.current_odom_data[0], self.current_odom_data[1], 0]
            end_arrow_point = list(global_laser_point)

            # print(f'origin_arrow_point:', origin_arrow_point)
            # print(f'end_arrow_point:', end_arrow_point)

            pf_sensor_data_marker = self.rviz_helper.generate_marker(ns='pf_sensor_data',
                                            frame_id='map',
                                            marker_type=Marker.ARROW,
                                            marker_action=Marker.ADD,
                                            marker_id=i,
                                            marker_scale=(0.025, 0.05, 0.05),
                                            marker_color=(1, 0.5, 1, 1),
                                            marker_points=[origin_arrow_point, end_arrow_point])
            
            pf_sensor_data_markers.append(pf_sensor_data_marker)

        self.current_pf_sensor_data_markers = MarkerArray(pf_sensor_data_markers)

        try:
            self.pf_sensor_data_marker_array_pub.publish(self.current_pf_sensor_data_markers)

        except Exception as e:
            print(f'Exception: {e}')
            pass

    def add_anchor_particles(self, particles):
        self.anchor_particles = []

        # Odom Particle
        self.anchor_particles.append(self.odom_particle)

        self.anchor_particle_start_index = len(particles)
        self.num_anchor_particles = len(self.anchor_particles)

        self.anchor_particles = np.array(self.anchor_particles)

        print(f'self.anchor_particles:', self.anchor_particles)

        particles = np.vstack((particles, self.anchor_particles))

        return particles

    def init_particles(self, count, spawn_position, spawn_orientation, radius, range_orientation):
        # print(f'init_particles - spawn position:', spawn_position)

        field0, field1, field2 = spawn_position
        particles = []

        max_attempts = 10 * count  # Prevents infinite loops in dense maps
        attempts = 0

        cos_y = math.cos(self.occupancy_map_yaw)
        sin_y = math.sin(self.occupancy_map_yaw)

        while len(particles) < count and attempts < max_attempts:
            # Sample positions within the given radius
            x_sample = field0 + np.random.uniform(-radius, radius)
            y_sample = field1 + np.random.uniform(-radius, radius)
            theta_sample = spawn_orientation + np.random.uniform(0, range_orientation)

            # Convert sampled positions to occupancy grid coordinates
            # grid_x = ((x_sample - self.occupancy_map_origin[0]) / self.occupancy_map_resolution).astype(int)
            # grid_y = (-(y_sample - self.occupancy_map_origin[1]) / self.occupancy_map_resolution).astype(int)

            # grid_x = int((x_sample - self.occupancy_map_origin[0]) / self.occupancy_map_resolution)
            # grid_y = int((y_sample - self.occupancy_map_origin[1]) / self.occupancy_map_resolution)

            dx = x_sample - self.occupancy_map_origin[0]
            dy = y_sample - self.occupancy_map_origin[1]
            x_map = cos_y * dx + sin_y * dy
            y_map = -sin_y * dx + cos_y * dy

            grid_x = int(x_map / self.occupancy_map_resolution)
            grid_y = int(y_map / self.occupancy_map_resolution)

            # print(f'grid_x:', grid_x)
            # print(f'grid_y:', grid_y)

            # print(f'self.occupancy_grid max x:', self.occupancy_grid.shape[1])
            # print(f'self.occupancy_grid max y:', self.occupancy_grid.shape[0])

            # Check if the sampled position is within map bounds
            if 0 <= grid_x < self.occupancy_grid.shape[1] and 0 <= grid_y < self.occupancy_grid.shape[0]:
                # Check if the cell is white (free space)
                # print(f'occupancy value:', self.occupancy_grid[grid_y, grid_x])

                # print(f'self.occupancy_grid grid_y grid_xy value:', self.occupancy_grid[grid_y, grid_x])

                if self.occupancy_grid[grid_y, grid_x] <= self.occupied_threshold:  # White area in occupancy grid
                    particles.append([x_sample, y_sample, theta_sample])

            # print(f'attempts:', attempts)

            attempts += 1

        if len(particles) == 0:
            raise ValueError("No valid particles found in free space. Try increasing radius or modifying map.")

        # Convert to NumPy array
        particles = np.array(particles)

        # Add spawn_origin_particle
        spawn_origin_particle = np.array([spawn_position[0], spawn_position[1], spawn_orientation])
        particles = np.vstack((particles, spawn_origin_particle))
        
        # Add anchor particles
        particles = self.add_anchor_particles(particles=particles)

        return particles
    
    def init_particles_unoccupied_map(self, count, orientation, occ_xmin_world=None, occ_xmax_world=None, occ_ymin_world=None, occ_ymax_world=None):
        print('INIT PARTICLES UNOCCUPIED!!!')

        # Extract occupancy map parameters
        map_origin_x, map_origin_y = self.occupancy_map_origin
        resolution = self.occupancy_map_resolution
        grid_height, grid_width = self.occupancy_map_height, self.occupancy_map_width

        cos_y = math.cos(self.occupancy_map_yaw)
        sin_y = math.sin(self.occupancy_map_yaw)

        if occ_xmin_world is not None: self.occ_xmin_world = occ_xmin_world
        if occ_xmax_world is not None: self.occ_xmax_world = occ_xmax_world
        if occ_ymin_world is not None: self.occ_ymin_world = occ_ymin_world
        if occ_ymax_world is not None: self.occ_ymax_world = occ_ymax_world

        occ_xmin_world, occ_xmax_world, occ_ymin_world, occ_ymax_world = (
            self.occ_xmin_world, self.occ_xmax_world, self.occ_ymin_world, self.occ_ymax_world
        )

        # Find all free cells in the occupancy grid
        free_cells = np.argwhere(self.occupancy_grid <= self.occupied_threshold)

        # print(f'free_cells:', free_cells)

        if len(free_cells) == 0:
            raise ValueError("No free space found in the occupancy grid!")

        # Convert grid coordinates to world coordinates
        # free_world_x = free_cells[:, 1] * resolution + map_origin_x
        # free_world_y = free_cells[:, 0] * resolution + map_origin_y

        # free_world = []
        # for my, mx in free_cells:
        #     x_map = (mx + 0.5) * resolution
        #     y_map = (my + 0.5) * resolution
        #     wx = cos_y * x_map - sin_y * y_map + map_origin_x
        #     wy = sin_y * x_map + cos_y * y_map + map_origin_y 
        #     free_world.append([wx, wy])

        # free_world = np.array(free_world)

        free_world = free_cells_to_world(
            free_cells=free_cells,
            resolution=resolution,
            map_origin_x=map_origin_x,
            map_origin_y=map_origin_y,
            cos_y=cos_y,
            sin_y=sin_y
        )


        free_world_x = free_world[:, 0]
        free_world_y = free_world[:, 1]

        # **Filter out cells outside the room bounds**
        mask = (free_world_x >= occ_xmin_world) & (free_world_x <= occ_xmax_world) & \
            (free_world_y >= occ_ymin_world) & (free_world_y <= occ_ymax_world)
        
        free_world_x, free_world_y = free_world_x[mask], free_world_y[mask]

        # print(f'free_world_x:', free_world_x)
        # print(f'free_world_y:', free_world_y)

        if len(free_world_x) == 0:
            raise ValueError("No valid free space inside the room boundaries!")

        free_positions = np.column_stack((free_world_x, free_world_y))

        # print(f'len free_positions:', len(free_positions))

        # Compute an adaptive step size based on valid free space
        x_range = np.ptp(free_world_x)  
        y_range = np.ptp(free_world_y)  
        grid_spacing = np.sqrt((x_range * y_range) / count)  

        # Create a spatially even grid within the free space
        x_grid = np.arange(np.min(free_world_x), np.max(free_world_x), grid_spacing)
        y_grid = np.arange(np.min(free_world_y), np.max(free_world_y), grid_spacing)
        xx, yy = np.meshgrid(x_grid, y_grid)
        candidate_particles = np.column_stack((xx.ravel(), yy.ravel()))

        # print(f'len candidate_particles:', len(candidate_particles))

        # **Ensure candidates are inside free space**
        valid_particles = []
        for px, py in candidate_particles:
            grid_x = int((px - map_origin_x) / resolution)
            grid_y = int((py - map_origin_y) / resolution)

            # print(f'grid_x:', grid_x)
            # print(f'grid_y:', grid_y)
            # print(f'grid_width:', grid_width)
            # print(f'grid_height:', grid_height)
            # print(f'self.occupancy_grid[grid_y, grid_x]:', self.occupancy_grid[grid_y, grid_x])

            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height and self.occupancy_grid[grid_y, grid_x] <= self.occupied_threshold:
                if self.global_kidnapping_num_particle_angles is not None:
                    max_angles = self.global_kidnapping_num_particle_angles
                else:
                    max_angles = 16
                for i in range(max_angles):
                    valid_particles.append([px, py, (2 * math.pi) / max_angles * i])

        particles = np.array(valid_particles)

        # Add anchor particles
        particles = self.add_anchor_particles(particles=particles)

        # **Include odometry-based particle**
        # odom_particle = np.array(self.current_odom_data).reshape(1, -1)

        # print(f'particles:', particles)
        # print(f'odom_particle:', odom_particle)

        # particles = np.vstack((particles, odom_particle))

        return particles
    
    def diffangle(self, a1, a2):
        if a1 < math.inf and a2 < math.inf:
            a1 = self.normangle(a1,0)
            a2 = self.normangle(a2,0)
            
            delta = a1 - a2
            
            if a1 > a2:
                while delta > math.pi: delta = delta - 2*math.pi
            elif a2 > a1:
                while delta < -math.pi: delta = delta + 2*math.pi
        else: delta = math.inf
        return delta

    def mean_position(self, particles, weights):
        if len(particles) == 0:
            return np.zeros(3)  # Return a default zero position if no particles exist
        
        mean_x = np.average(particles[:, 0], weights=weights)
        mean_y = np.average(particles[:, 1], weights=weights)
        
        mean_sin = np.average(np.sin(particles[:, 2]), weights=weights)
        mean_cos = np.average(np.cos(particles[:, 2]), weights=weights)
        mean_theta = np.arctan2(mean_sin, mean_cos)



        mean_pos = np.array([mean_x, mean_y, mean_theta])

        return mean_pos
    
    def refresh_anchor_particles(self, particles):
        self.anchor_particles = []

        self.anchor_particles.append(self.odom_particle)

        particles[self.anchor_particle_start_index:] = self.anchor_particles

        # print(f'particles:', particles)

        return particles

    def resample(self, particles, weights):
        N = weights.shape[0]
        my_numbers = np.arange(N)
        my_random = np.random.rand()
        positions = (my_numbers + my_random) / N
        indexes = np.zeros(N, 'i')


        slices = np.cumsum(weights) # Determine probability boundaries of each particle
        indexes = np.searchsorted(slices, positions) # Find where each position lands on the sample slices
        new_particles = particles[indexes]

        new_particles = self.refresh_anchor_particles(particles=new_particles)

        # if self.odom_particle is not None:
        #     new_particles[0] = self.odom_particle
        
        return new_particles
    
    def sample_motion_model(self, a, x, add_noise=False):
        # Factors for scaling the odometry diff
        if ros_master_status == 'hm':
            if self.sample_motion_model_x_factor is not None:
                x_factor = float(self.sample_motion_model_x_factor)
            else:
                x_factor = 0.5
        
            if self.sample_motion_model_y_factor is not None:
                y_factor = float(self.sample_motion_model_y_factor)
            else:
                y_factor = 0.5

            if self.sample_motion_model_theta_factor is not None:
                theta_factor = float(self.sample_motion_model_theta_factor)
            else:
                theta_factor = 0.9
        else:
            # Gazebo
            # x_factor = 0.9
            # y_factor = 0.9
            # theta_factor = 0.9

            # Isaac Sim
            # x_factor = 2.1
            # y_factor = 2.1
            # theta_factor = 2.1

            if self.sample_motion_model_x_factor is not None:
                x_factor = float(self.sample_motion_model_x_factor)
            else:
                x_factor = 2.1
        
            if self.sample_motion_model_y_factor is not None:
                y_factor = float(self.sample_motion_model_y_factor)
            else:
                y_factor = 2.1

            if self.sample_motion_model_theta_factor is not None:
                theta_factor = float(self.sample_motion_model_theta_factor)
            else:
                theta_factor = 2.1

        # print("a:", a)
        # Scale odometry difference 'a' (Δx, Δy, Δθ)
        factors = np.array([x_factor, y_factor, theta_factor])
        a_scaled = np.multiply(a, factors)  # Scaling odometry diff

        # print(f'a_scaled:', a_scaled)

        # Shape of x
        x_row, _ = x.shape

        if add_noise:
            # Noise factors (adjust to reflect real-world noise levels)
            noise = [1, 1, 1]
            noise = np.array(noise)
            # noise = np.array([1, 1, 1.2])
            # noise = np.array([1.2, 1.2, 1.4])


            # Create noise-adjusted odometry difference
            S = np.tile(noise * a_scaled, (x_row, 1))  # Adjust odometry diff with noise

            # Generate random Gaussian noise
            base_random_factor = np.random.randn(x_row, 3)

            # Define Gaussian noise parameters (adjust for realism)
            
            if ros_master_status == 'hm':
                if self.sample_motion_model_mu is not None and self.sample_motion_model_sigma is not None:
                    mu = float(self.sample_motion_model_mu)
                    sigma = float(self.sample_motion_model_sigma)
                
                else:
                    mu = 14
                    sigma= 1

            if ros_master_status == 'sm':
                if self.sample_motion_model_mu is not None and self.sample_motion_model_sigma is not None:
                    mu = float(self.sample_motion_model_mu)
                    sigma = float(self.sample_motion_model_sigma)
                else:
                    mu = 2
                    sigma= 1

            random_factor = base_random_factor * mu + sigma

            # Calculate noise-affected motion
            N = np.multiply(random_factor, S)

            # N = S

            # Extract individual components (Δx, Δy, Δθ) from noise-adjusted values
            first_col = N[:, 0]
            second_col = N[:, 1]
            third_col = N[:, 2]

            # Combine noise-affected updates
            chosen_array = np.array([first_col, second_col, third_col])

            # Update particles with new positions based on odometry difference
            x_new = np.add(x, chosen_array.T)

        else:
            # Deterministic motion update without noise
            S = np.tile(a_scaled, (x_row, 1))  # Pure odometry update
            x_new = np.add(x, S)

        return x_new
    
    def check_for_convergence(self, particles, normalized_weights, mean, threshold=0.2):
        """Reset kidnap trigger once particle cloud is tight."""
        weights = normalized_weights

        if not self.pf_kidnap_trigger:
            return particles, weights
        
        # print(f'check for convergence call!')

        std_x = np.sqrt(np.average((particles[:, 0] - mean[0])**2, weights=normalized_weights))
        std_y = np.sqrt(np.average((particles[:, 1] - mean[1])**2, weights=normalized_weights))
        std_total = np.sqrt(std_x**2 + std_y**2)

        # print(f'std_total:', std_total)
        # print(f'threshold:', threshold)

        mean_pos = [mean[0], mean[1], 0.0]

        if std_total < threshold:
            print(f"[PF] Converged after kidnapping! Time = {round(time.time() - self.kidnap_start_time, 2)} s")
            particles = self.init_particles(count=self.num_tracking_particles, spawn_position=mean_pos, spawn_orientation=mean[2], radius=0.5, range_orientation=math.pi)

            # particles = self.init_particles(count=500, spawn_position=mean_pos, spawn_orientation=mean[2], radius=0.5, range_orientation=math.pi)

            weights = []
            weights = compute_weights_numba(X_new=particles,
                                        laser_ranges=self.laser_ranges,
                                        raycast_particle_step_size=self.raycast_particle_step_size,
                                        raycast_particle_max_raycast_dist=self.raycast_particle_max_raycast_dist,
                                        occupancy_map_resolution=self.occupancy_map_resolution,
                                        laser_angles=self.laser_angles,
                                        laser_max_range=self.laser_max_range,
                                        occupancy_map_origin=self.occupancy_map_origin,
                                        occupancy_map_yaw=self.occupancy_map_yaw,
                                        occupancy_grid=self.occupancy_grid,
                                        measurement_weight_sigma=self.measurement_weight_sigma,
                                        measurement_weight_epsilon=self.measurement_weight_epsilon)
            
            weights = np.divide(weights, np.sum(weights))

            self.delete_namespace_markers()

            # print(f'convergence self.X:', len(self.X))
            self.pf_kidnap_trigger = False
            self.converged = True
            self.convergence_time = time.time() - self.kidnap_start_time

        return particles, weights

    def particle_filter(self, z, u, X):
        try:
            self.tf_helper.tf_listener.waitForTransform(target_frame='base_link',
                                        source_frame='base_range_sensor_link',
                                        time=rospy.Time(0),
                                        timeout=rospy.Duration(1))

            (trans, rot) = self.tf_helper.tf_listener.lookupTransform(target_frame='base_link', source_frame='base_range_sensor_link', time=rospy.Time(0))

            self.sensor_offset_x, self.sensor_offset_y, sensor_offset_z = trans

        except Exception as e:
            rospy.logwarn(f'TF lookup exception: {e}')

        if X is not None and X.size == 0:
            rospy.loginfo('Initializing particles')

            X = self.init_particles(count=self.num_tracking_particles, spawn_position=[self.current_odom_data[0] + self.sensor_offset_x, self.current_odom_data[1], 0], spawn_orientation=self.current_odom_data[2], radius=1, range_orientation=math.pi/2)

            if self.pf_kidnap_trigger:
                rospy.loginfo('Initializing particles (kidnap recovery)')

                if ros_master_status == 'hm':
                    if self.global_kidnapping_num_particles is not None and self.global_kidnapping_max_orientation is not None and self.global_kidnapping_occ_map_xmin is not None and self.global_kidnapping_occ_map_ymin is not None and self.global_kidnapping_occ_map_ymax is not None:
                        rospy.loginfo('Global kidnapping parameters provided')
                        X = self.init_particles_unoccupied_map(count=float(self.global_kidnapping_num_particles), orientation=float(self.global_kidnapping_max_orientation), occ_xmin_world=float(self.global_kidnapping_occ_map_xmin), occ_xmax_world=float(self.global_kidnapping_occ_map_xmax), occ_ymin_world=float(self.global_kidnapping_occ_map_ymin), occ_ymax_world=float(self.global_kidnapping_occ_map_ymax))
                    else:
                        X = self.init_particles_unoccupied_map(count=400, orientation=math.pi, occ_xmin_world=-0.4, occ_xmax_world=11.3, occ_ymin_world=-0.7, occ_ymax_world=6.6)

                if ros_master_status == 'sm':
                    if self.global_kidnapping_num_particles is not None and self.global_kidnapping_max_orientation is not None and self.global_kidnapping_occ_map_xmin is not None and self.global_kidnapping_occ_map_ymin is not None and self.global_kidnapping_occ_map_ymax is not None:
                        X = self.init_particles_unoccupied_map(count=float(self.global_kidnapping_num_particles), orientation=float(self.global_kidnapping_max_orientation), occ_xmin_world=float(self.global_kidnapping_occ_map_xmin), occ_xmax_world=float(self.global_kidnapping_occ_map_xmax), occ_ymin_world=float(self.global_kidnapping_occ_map_ymin), occ_ymax_world=float(self.global_kidnapping_occ_map_ymax))
                    else:
                        X = self.init_particles_unoccupied_map(count=350, orientation=math.pi, occ_xmin_world=0.0, occ_xmax_world=9.7, occ_ymin_world=-0.5, occ_ymax_world=6.9)

                self.kidnap_start_time = time.time()

            X_new = X
            mean = self.current_odom_data

            return X_new, mean

        # Placeholder values
        # X_new = X
        # mean = self.current_odom_data
        # weights = np.ones((X.shape[0], 1))

        # print(f'X:', X)

        X_new = self.sample_motion_model(u, X, add_noise=True)
        # X_new = self.sample_motion_model(u, X, add_noise=False)

        # print(f'X_new sample motion:', X_new)

        pf_endpoints = []        

        # X_new = X_new.tolist()

        # print(f'X_new:', X_new)
        # print(f'type X_new:', type(X_new))
        # print(f'type X_new[0]:', type(X_new[0]))

        # print(f'len(laser_ranges):', len(self.laser_ranges))

        self.pf_sensor_data_num_landmarks_seen_pub.publish(str(len(self.laser_ranges)))

        # print(f'Before compute_weights_numba!')
        # print(f'measurement_weight_sigma:', self.measurement_weight_sigma)
        # print(f'measurement_weight_epsilon:', self.measurement_weight_epsilon)

        weights = []
        weights = compute_weights_numba(X_new=X_new,
                                        laser_ranges=self.laser_ranges,
                                        raycast_particle_step_size=self.raycast_particle_step_size,
                                        raycast_particle_max_raycast_dist=self.raycast_particle_max_raycast_dist,
                                        occupancy_map_resolution=self.occupancy_map_resolution,
                                        laser_angles=self.laser_angles,
                                        laser_max_range=self.laser_max_range,
                                        occupancy_map_origin=self.occupancy_map_origin,
                                        occupancy_map_yaw=self.occupancy_map_yaw,
                                        occupancy_grid=self.occupancy_grid,
                                        measurement_weight_sigma=self.measurement_weight_sigma,
                                        measurement_weight_epsilon=self.measurement_weight_epsilon)
        
        # print(f'weights:', weights)

        weights = np.divide(weights, np.sum(weights))
        # print(f'X_new:', X_new)
        # print(f'normalized weights:', weights)

        # print('average_particle_weight:', np.mean(weights))

        theta_array = X_new[:, 2]

        mean = self.mean_position(X_new, weights)

        # mean = [0, 0, 0]

        # Assuming the mean pose is in the map frame: [mean_x, mean_y, mean_theta]
        mean_x, mean_y, mean_theta = mean  # Robot position in map frame

        # try:
        #     self.tf_helper.tf_listener.waitForTransform(target_frame='base_link',
        #                                 source_frame='base_range_sensor_link',
        #                                 time=rospy.Time(0),
        #                                 timeout=rospy.Duration(1))

        #     (trans, rot) = self.tf_helper.tf_listener.lookupTransform(target_frame='base_link', source_frame='base_range_sensor_link', time=rospy.Time(0))
        #     # print(f'(trans, rot):', (trans, rot))

        #     sensor_offset_x, sensor_offset_y, sensor_offset_z = trans

        # except Exception as e:
        #     print(f'Exception:', e)


        # Sensor offset in the base_link frame
        # sensor_offset_x, sensor_offset_y = sensor_offset_x, sensor_offset_y

        # Rotation matrix based on robot's orientation in the map frame (mean_theta)
        rot_matrix = np.array([
            [np.cos(mean_theta), -np.sin(mean_theta), 0],
            [np.sin(mean_theta), np.cos(mean_theta), 0],
            [0, 0, 1]
        ])

        # Apply the sensor offset (relative to base_link) to the robot's global position
        # First, apply rotation to the sensor offset
        self.adjusted_sensor_offset = np.dot(rot_matrix, np.array([self.sensor_offset_x, self.sensor_offset_y, 0]))

        # Now translate the mean pose by the rotated sensor offset
        adjusted_mean_x = mean_x - self.adjusted_sensor_offset[0]
        adjusted_mean_y = mean_y - self.adjusted_sensor_offset[1]

        # The orientation (mean_theta) remains the same in the map frame
        adjusted_mean_theta = mean_theta

        # Final adjusted mean pose
        mean = [adjusted_mean_x, adjusted_mean_y, adjusted_mean_theta]

        X_new, weights = self.check_for_convergence(X_new, normalized_weights=weights, mean=mean)

        X_new = self.resample(X_new, weights)

        # print(f'X_new resample:', X_new)
        # print(f'resample len(X_new):', len(X_new))

        return X_new, mean
    
    def normangle(self, a, mina):
        if a < math.inf:
            while a >= mina + 2 * math.pi: a = a - 2*math.pi
            while a < mina: a = a + 2*math.pi
            ar = a
        else: ar = math.inf
        return ar

    def publish_pf_state(self):
        # print(f'publish_pf_state!')
        pf_particles_markers = []

        for i, particle in enumerate(self.X):
            pf_particle_marker = self.rviz_helper.generate_marker(ns='pf_particles',
                                 frame_id='map',
                                 marker_type=Marker.ARROW,
                                 marker_action=Marker.ADD,
                                 marker_id=i,

                                 marker_scale=(0.25, 0.05, 0.05),
                                 marker_color=(1,0,0,1),
                                 marker_position=(particle[0], particle[1], 0),
                                 marker_orientation=(0, 0, particle[2]))
            
            pf_particles_markers.append(pf_particle_marker)

        self.pf_particles_marker_array_pub.publish(pf_particles_markers)

        mean_position = (self.mean[0], self.mean[1], 0)
        mean_orientation = (0, 0, self.mean[2])

        # mean_orientation = (0, 0, self.normangle(self.mean[2], mina=0))

        # print(f'marker mean_position: {mean_position}')
        # print(f'marker mean_orientation: {mean_orientation}')

        pf_particles_mean_marker = self.rviz_helper.generate_marker(ns='pf_particles_mean',
                                                        frame_id='map',
                                                        marker_type=Marker.ARROW,
                                                        marker_action=Marker.ADD,
                                                        marker_id=0,
                                                        marker_scale=(0.25, 0.05, 0.05),
                                                        marker_color=(1,1,0,1),
                                                        marker_position=mean_position,
                                                        marker_orientation=mean_orientation)
        
        self.pf_particles_mean_marker_pub.publish(pf_particles_mean_marker)

        self.pf_particles_mean_raycast_occ_coverage_pub.publish(str(self.mean_particle_raycast_coverage))

        self.pf_particles_mean_live_scan_occ_coverage_pub.publish(str(self.mean_particle_live_scan_coverage))

        if self.convergence_time is not None:
            pf_convergence_time_str = str(self.convergence_time)
            self.pf_particles_convergence_time_pub.publish(pf_convergence_time_str)

        pf_variance_str = str([str(elem) for elem in self.pf_variance])

        self.pf_particles_variance_pub.publish(pf_variance_str)

        pf_std_str = str([str(elem) for elem in self.pf_std])

        self.pf_particles_std_pub.publish(pf_std_str)

    def move_base_pose(self, desired_pose_position, desired_pose_orientation):
        pose_cov = PoseWithCovarianceStamped()
        pose_cov.header.frame_id = 'map'  # Ensure this matches your coordinate frame
        pose_cov.header.stamp = rospy.Time.now()

        # print(f'header.stamp: {pose_cov.header.stamp}')

        pose_cov.pose.pose.position.x = desired_pose_position[0]  # Set your desired x position
        pose_cov.pose.pose.position.y = desired_pose_position[1]  # Set your desired y position
        pose_cov.pose.pose.position.z = desired_pose_position[2]  # Typically 0 for 2D navigation

        # print(f'move_base_pose - desired_pose_position:', desired_pose_position)
        # print(f'move_base_pose - desired_pose_orientation:', desired_pose_orientation)
        
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
    
    def compute_particle_raycast_occupancy_map_coverage(self, particle_pose):
        raycasted_ranges = raycast_particle_numba(particle_pose=particle_pose, 
                                                    raycast_particle_step_size=self.raycast_particle_step_size, 
                                                    raycast_particle_max_raycast_dist=self.raycast_particle_max_raycast_dist, 
                                                    occupancy_map_resolution=self.occupancy_map_resolution,
                                                    laser_angles=self.laser_angles,
                                                    laser_max_range=self.laser_max_range,
                                                    occupancy_map_origin=self.occupancy_map_origin,
                                                    occupancy_map_yaw=self.occupancy_map_yaw,
                                                    occupancy_grid=self.occupancy_grid)
        hits = 0
        total = len(raycasted_ranges)

        # print(f'total:', total)

        if not all(map(math.isfinite, particle_pose)):
            return 0.0

        for i, r in enumerate(raycasted_ranges):
            # Compute global coordinates of ray endpoint
            x_end = particle_pose[0] + r * math.cos(particle_pose[2] + self.laser_angles[i])
            y_end = particle_pose[1] + r * math.sin(particle_pose[2] + self.laser_angles[i])

            # print(f'(x_end, y_end):', (x_end, y_end))

            # Transform to map coordinates
            dx = x_end - self.occupancy_map_origin[0]
            dy = y_end - self.occupancy_map_origin[1]
            x_map = int((math.cos(self.occupancy_map_yaw) * dx + math.sin(self.occupancy_map_yaw) * dy) / self.occupancy_map_resolution)
            y_map = int((-math.sin(self.occupancy_map_yaw) * dx + math.cos(self.occupancy_map_yaw) * dy) / self.occupancy_map_resolution)

            # print(f'(x_map, y_map):', (x_map, y_map))

            if (0 <= x_map < self.occupancy_grid.shape[1]) and (0 <= y_map < self.occupancy_grid.shape[0]):
                # print(f'occupancy_value:', self.occupancy_grid[y_map, x_map])
                if self.occupancy_grid[y_map, x_map] >= self.occupied_threshold:
                    hits += 1

        # print("hits:", hits)
        # print("total:", total)

        return hits / total if total > 0 else 0.0

    # Note: This doesn't work well with the sim occupancy map due non-gradual cutoff of the occupancy regions. In other words, use the raycast occupancy if no static obstacles for sim and live scan occupancy if there are dynamic obstacles on the real robot.
    def compute_particle_live_scan_occupancy_map_coverage(self, particle_pose):
        """
        Compare live laser scan endpoints (from the mean particle perspective)
        against the static occupancy map.
        """
        observed = self.laser_ranges
        total = len(observed)
        hits = 0

        if not all(map(math.isfinite, particle_pose)):
            return 0.0

        for i, r in enumerate(observed):
            # Compute global coordinates of live scan endpoint
            x_end = particle_pose[0] + r * math.cos(particle_pose[2] + self.laser_angles[i])
            y_end = particle_pose[1] + r * math.sin(particle_pose[2] + self.laser_angles[i])

            # Transform into map coordinates
            dx = x_end - self.occupancy_map_origin[0]
            dy = y_end - self.occupancy_map_origin[1]
            x_map = int((math.cos(self.occupancy_map_yaw) * dx + math.sin(self.occupancy_map_yaw) * dy) / self.occupancy_map_resolution)
            y_map = int((-math.sin(self.occupancy_map_yaw) * dx + math.cos(self.occupancy_map_yaw) * dy) / self.occupancy_map_resolution)

            # Check map bounds + occupancy
            if (0 <= x_map < self.occupancy_grid.shape[1]) and (0 <= y_map < self.occupancy_grid.shape[0]):
                cell_value = self.occupancy_grid[y_map, x_map]

                # print(f"Ray {i}: range={r:.2f}, map_coord=({x_map},{y_map}), occupancy={cell_value}")

                if cell_value >= self.occupied_threshold:
                    hits += 1
                    # print(f"  → HIT! (value {cell_value} >= threshold {self.occupied_threshold})")
                else:
                    # print(f"  → MISS (value {cell_value} < threshold {self.occupied_threshold})")
                    pass

        return hits / total if total > 0 else 0.0

    def pf_step(self, landmarks_num=3):
        # print(f'start pf_step!')

        z = np.array([[]])
        u = np.array([[]])

        self.sensor_data = []

        if self.sensor_data is not None and len(self.sensor_data) >= landmarks_num and not self.pf_first_time:
            z = np.array(self.sensor_data)
        else:
            self.pf_first_time = False

        if self.odom_diff is not None and len(self.odom_diff) != 0:
            u = np.array([self.odom_diff])

            # print(f'u:', u)
        else:
            u = np.array([[0.0, 0.0, 0.0]])

        # print(f'z: {z}')
        # print(f'u: {u}')
        # print(f'before self.X: {self.X}')

        self.X, self.mean = self.particle_filter(z, u, self.X)

        # Exclude anchor particles from std and variance calculations.
        regular_particles = self.X[:self.anchor_particle_start_index]

        x_variance = np.var(regular_particles[:, 0])
        y_variance = np.var(regular_particles[:, 1])
        theta_variance = np.var(regular_particles[:, 2])

        # x_variance = np.var(self.X[:, 0])
        # y_variance = np.var(self.X[:, 1])
        # theta_variance = np.var(self.X[:, 2])

        self.pf_variance = [x_variance, y_variance, theta_variance]

        # print(f'(x_variance, y_variance, theta_variance):', (x_variance, y_variance, theta_variance))

        x_std = np.std(regular_particles[:, 0])
        y_std = np.std(regular_particles[:, 1])
        theta_std = np.std(regular_particles[:, 2])
       
        # x_std = np.std(self.X[:, 0])
        # y_std = np.std(self.X[:, 1])
        # theta_std = np.std(self.X[:, 2])

        self.pf_std = [x_std, y_std, theta_std]

        # print(f'(x_std, y_std, theta_std):', (x_std, y_std, theta_std))

        # unique_particles = np.unique(self.X, axis=0)
        # print(f'len(unique_particles):', len(unique_particles))

        # print(f'self.pf_first_time:', self.pf_first_time)

        # print(f'after self.X:', self.X)

        # print(f'self.mean: {self.mean}')
        # print(f'self.mean type: {type(self.mean)}')

        mean_position = (self.mean[0], self.mean[1], 0)
        mean_orientation = (0, 0, self.normangle(self.mean[2], mina=0))

        mean_particle = np.array([mean_position[0], mean_position[1], mean_orientation[2]])

        # print(f'mean_position:', mean_position)
        # print(f'mean_orientation:', mean_orientation)
        # print(f'mean_particle:', mean_particle)

        self.mean_particle_raycast_coverage = self.compute_particle_raycast_occupancy_map_coverage(particle_pose=mean_particle)

        # print(f'mean_particle_raycast_coverage:', self.mean_particle_raycast_coverage)

        self.mean_particle_live_scan_coverage = self.compute_particle_live_scan_occupancy_map_coverage(particle_pose=mean_particle)

        # print(f'self.mean_particle_live_scan_coverage:', self.mean_particle_live_scan_coverage)

        # The meaning of "convergence" is that the particle come together or cluster at the correct x, y theta robot pose. In this case, I check this by saying that the particle mean satisfies a position + angular distance threshold set as the original target criteria for the thesis. For simulator, this works successfully. For real world, this doesn't work as successfully due camera groundtruth being inexact. This was resolved by doing hand measurements but doesn't provide a good sense for convergence time. Thus, for real world values, the convergence criteria was relaxed simply to get a convergence time estimate near the actual position.

        # This was commented out as convergence was mainly already dominated by prior implementation in the form pf std_threshold. Correctness of position is evaluated through raycast coverage and/or pose error from groundtruth. 
        # if not self.converged:
        #     if ros_master_status == 'sm':
        #         pos_error = np.linalg.norm(np.array(self.mean[:2]) - np.array(self.current_isaac_robot_pose[:2]))
        #         ang_error = abs(self.diffangle(self.mean[2], self.current_isaac_robot_pose[2]))

        #         if pos_error < 0.05 and ang_error < 0.052:  # Your thresholds
        #             self.converged = True
        #             print(f'self.kidnap_start_time:', self.kidnap_start_time)
        #             self.convergence_time = time.time() - self.kidnap_start_time
        #             print(f"Converged in {self.convergence_time:.2f} seconds")

        #     if ros_master_status == 'hm':
        #         if self.current_robot_center_camera_groundtruth_pose is not None:
        #             pos_error = np.linalg.norm(np.array(self.mean[:2]) - np.array(self.current_robot_center_camera_groundtruth_pose[:2]))
        #             ang_error = abs(self.diffangle(self.mean[2], self.current_robot_center_camera_groundtruth_pose[2]))

        #             if pos_error < 0.2:  # Your thresholds
        #                 self.converged = True
        #                 self.convergence_time = time.time() - self.kidnap_start_time
        #                 print(f"Converged in {self.convergence_time:.2f} seconds")

        try:
            self.publish_pf_state()

        except Exception as e:
            print(f'Exception:', e)

        # print(f'after publish_pf_state!')

        ### Robot Pose Correction
        if self.enable_odom_correction:
            ## Always correct...
            # if all(map(math.isfinite, mean_position)) and all(map(math.isfinite, mean_orientation)):
            #     self.move_base_pose(desired_pose_position=mean_position, desired_pose_orientation=mean_orientation)

            # print(f'mean_position:', mean_position)
            # print(f'mean_orientation:', mean_orientation)

            ## Correct only once there is a pf mean and pf has converged.
            if mean_position is not None and mean_orientation is not None and self.converged is True:
                if self.kidnap_correct_position is None and self.kidnap_correct_orientation is None:  
                    self.kidnap_correct_position = mean_position
                    self.kidnap_correct_orientation = mean_orientation

                if self.kidnap_correct_position is not None and self.kidnap_correct_orientation is not None and self.reached_kidnap_correct_pose is False:
                    print(f'desired_pose_position:', self.kidnap_correct_position)
                    print()

                    print(f'desired_pose_orientation:', self.kidnap_correct_orientation)
                    print()

                    if all(map(math.isfinite, self.kidnap_correct_position)) and all(map(math.isfinite, self.kidnap_correct_orientation)):
                        self.move_base_pose(desired_pose_position=self.kidnap_correct_position, desired_pose_orientation=self.kidnap_correct_orientation)

                    dx = self.current_odom_data[0] - self.kidnap_correct_position[0]
                    dy = self.current_odom_data[1] - self.kidnap_correct_position[1]
                    pos_error = math.hypot(dx, dy)

                    dtheta = self.current_odom_data[2] - self.kidnap_correct_orientation[2]
                    dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

                    print(f'pos_error:', pos_error)
                    print(f'orient_error:', dtheta)

                    if pos_error < 0.2 and dtheta < 0.052:
                        self.reached_kidnap_correct_pose = True

                if self.reached_kidnap_correct_pose:
                    if all(map(math.isfinite, mean_position)) and all(map(math.isfinite, mean_orientation)):
                        self.move_base_pose(desired_pose_position=mean_position, desired_pose_orientation=mean_orientation)

        # print(f'self.particle_filter_iter:', self.particle_filter_iter)

        self.particle_filter_iter += 1

    def pf_timed_cb(self, event):
        # print(f'In timer callback! count: {self.timer_callback_count}')
        self.timer_callback_count += 1

        # print(f'pf_timed_callback event: {event}')
        # print(f'pf_timed_callback event last duration: {event.last_duration}')

        self.get_odom_data()

        if ros_master_status == 'sm':
            self.establish_isaac_sim_robot_pose_marker()

        if self.current_laser_data:
            self.get_laser_data()
            self.process_laser_data()

        else:
            print(self.general_helper.trans_sep)
            print(f'No laser_data! Waiting for next cycle...\n')
            print(self.general_helper.trans_sep)
            return

        if not self.processed_occupancy_data: 
            print('occupancy data not processed! Continue...')
            return

        self.pf_step()
  
    def delete_all_markers(self, ns):
        # print(f'\nDeleting all markers: {ns}\n')

        markers = []

        marker = Marker()
        marker.ns = ns
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.DELETEALL

        markers.append(marker)

        marker_array = MarkerArray(markers)

        if ns == 'odom':
            self.odom_marker_pub.publish(marker)

        if ns == 'isaac_robot_pose':
            self.isaac_sim_robot_pose_pub.publish(marker)

        if ns == "global_laser_points":
            self.global_laser_marker_array_pub.publish(marker_array)
        
        if ns == "pf_sensor_data":
            self.pf_sensor_data_marker_array_pub.publish(marker_array)

        if ns == "pf_particles":
            self.pf_particles_marker_array_pub.publish(marker_array)

        if ns == "pf_particles_mean":
            self.pf_particles_mean_marker_pub.publish(marker)

    def delete_namespace_markers(self):
        self.delete_all_markers(ns='odom')
        self.delete_all_markers(ns='isaac_robot_pose')
        self.delete_all_markers(ns='global_laser_points')
        self.delete_all_markers(ns='pf_sensor_data')
        self.delete_all_markers(ns='pf_particles')
        self.delete_all_markers(ns='pf_particles_mean')

    def shutdown_cb(self):
        self.delete_namespace_markers()

    def run_nodes(self):
        print('In run_nodes!')

        # Subscribed topics
        rospy.Subscriber("/clock", Clock, self.clock_cb)

        # Occupancy Grid from map server
        if ros_master_status == 'hm':
            rospy.Subscriber("/static_obstacle_ros_map", OccupancyGrid, self.occupancy_grid_cb)
        else:
            rospy.Subscriber("/map", OccupancyGrid, self.occupancy_grid_cb)

        # Laser Scan
        rospy.Subscriber("/hsrb/base_scan", LaserScan, self.laser_scan_cb)

        # Lab Camera Robot Pose
        if ros_master_status == 'hm':
            rospy.Timer(rospy.Duration(self.node_time_interval), self.get_robot_center_camera_groundtruth_timed_cb)

        # Isaac Sim Robot Pose
        if ros_master_status == 'sm':
            rospy.Subscriber("/isaac_sim/robot_pose", String, self.isaac_robot_pose_cb)

        rospy.Timer(rospy.Duration(self.node_time_interval), self.pf_timed_cb)

        rospy.on_shutdown(self.shutdown_cb)

        rospy.spin()

if __name__ == '__main__':
    kidnapper = hsr_pf_localization()
    # kidnapper = hsr_pf_localization(record_logging_data=True)

    if ros_master_status == 'hm':
        kidnapper.run_nodes()
