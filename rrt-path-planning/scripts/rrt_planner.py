#! /usr/bin/env python3
"""RRT path planner and navigation controller for the Toyota HSR robot.

Implements Rapidly-exploring Random Tree (RRT) path planning with obstacle
inflation, dynamic replanning, and RViz visualization. Includes a robot
controller that executes planned paths with real-time obstacle monitoring.
"""
import numpy as np
import random
import math
import rospy, tf
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.interpolate import splprep, splev
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from tf import TransformListener
from tf.transformations import *
from sensor_msgs.msg import LaserScan

MARKER_ID_TREE = 0
MARKER_ID_NODE_START = 1000  # Nodes: 1000-1999
MARKER_ID_PATH = 2000
MARKER_ID_START = 3000
MARKER_ID_GOAL = 3001
MARKER_ID_UPDATED_PATH = 4000  # For dynamic path updates

# Module-level ROS resources (initialized in __main__)
tf_listener = None
marker_pub = None


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, map_size, occupancy_grid, map_data, step_size=0.3, max_iter=1000000, x_min=-1, x_max=11, y_min=-1, y_max=8, inflation_radius=5.0):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_size = map_size
        self.map_data = map_data
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.original_occupancy_grid = occupancy_grid
        self.occupancy_grid = self.inflate_obstacles(inflation_radius)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


    def get_random_node(self, goal_bias=0.05):
        if random.uniform(0.0, 1.0) <= goal_bias:
            return self.goal

        x_rand = random.uniform(self.x_min, self.x_max)
        y_rand = random.uniform(self.y_min, self.y_max)

        return Node(x_rand, y_rand)

    def get_nearest_node(self, node):
        best_node = None
        best_dist = float('inf')
        for nei in self.node_list:
           dist = self.get_euclidean_distance(node, nei)
           if dist < best_dist:
               best_node = nei
               best_dist = dist

        return best_node

    def get_euclidean_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def is_within_world(self, p):
        return (self.x_min < p.x < self.x_max) and (self.y_min < p.y < self.y_max)

    def obstacle_free(self, xnear, xnew):
        rez = float(self.map_data.info.resolution) * 0.2
        dist = self.get_euclidean_distance(xnear, xnew)
        steps = int(dist / rez)
        for i in range(steps):
            frac = i / steps
            x = xnear.x + frac * (xnew.x - xnear.x)
            y = xnear.y + frac * (xnew.y - xnear.y)
            if not self.is_safe(Node(x, y),self.occupancy_grid ,self.map_data):
                return 0
        return 1

    def steer(self, x_nearest, x_rand, eta):
        x_new = Point()
        if self.get_euclidean_distance(x_nearest, x_rand) <= eta:
            x_new = x_rand
        else:
            m = (x_rand.x - x_nearest.x) / (x_rand.y - x_nearest.y) if x_rand.y != x_nearest.y else float('inf')
            x_new.y = (math.copysign(1, x_rand.y - x_nearest.y)) * (math.sqrt(eta ** 2 / (m ** 2 + 1))) + x_nearest.y
            x_new.x = m * (x_new.y - x_nearest.y) + x_nearest.x
            if x_rand.y == x_nearest.y:
                x_new.y = x_nearest.y
                x_new.x = x_nearest.x + eta
        return x_new

    def grid_value(self, xp):
        resolution = self.map_data.info.resolution
        x_start_x = self.map_data.info.origin.position.x
        x_start_y = self.map_data.info.origin.position.y
        width = self.map_data.info.width
        data = self.map_data.data

        indx = (math.floor((xp.y - x_start_y) / resolution) * width + math.floor((xp.x - x_start_x) / resolution))
        return data[int(indx)]

    def quick_path(self, p1, p2):
        """
        Check if there's a direct, obstacle-free path from p1 to p2.
        p1 and p2 are Node objects.
        Returns a path (list of tuples) if clear, None otherwise.
        """
        # First, check if the straight line is obstacle-free
        if not self.obstacle_free(p1, p2):
            return None
        
        # If it's clear, create a path with intermediate waypoints
        distance = self.get_euclidean_distance(p1, p2)
        num_steps = max(2, int(distance / self.step_size))
        
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            x = p1.x + t * (p2.x - p1.x)
            y = p1.y + t * (p2.y - p1.y)
            path.append((x, y))
        
        rospy.loginfo(f"Found direct path to goal with {len(path)} waypoints!")
        
        return path

    def extend_tree(self, nearest_node, random_node):

        theta = math.atan2(random_node.y - nearest_node.y, random_node.x - nearest_node.x)
        new_node = Node(nearest_node.x + self.step_size * math.cos(theta),
                        nearest_node.y + self.step_size * math.sin(theta))
        new_node.parent = nearest_node
        
        if (self.obstacle_free(nearest_node, new_node) == 1): 
            return new_node
        return None  # Invalid node

    def plan(self):

        path = self.quick_path(self.start, self.goal)
        if path:
            return path

        i = 0
        x_new = self.start
        while i < self.max_iter and self.get_euclidean_distance(x_new, self.goal) > self.step_size:
            x_rand = self.get_random_node()
            x_near = self.get_nearest_node(x_rand)
            x_new_candidate = self.extend_tree(x_near, x_rand)
            if x_new_candidate:
                x_new = x_new_candidate
                self.node_list.append(x_new)

            i += 1
        if self.get_euclidean_distance(x_new, self.goal) <= self.step_size:
            return self.get_path(x_new)
        
        return None


    def inflate_obstacles(self, radius):
        """ Inflate obstacles in the occupancy grid by the specified radius. """
        inflated_grid = self.original_occupancy_grid.copy()
        radius_cells = int(np.ceil(radius))  # Convert radius to cell count

        for y in range(self.map_size[1]):
            for x in range(self.map_size[0]):
                if self.original_occupancy_grid[y, x] > 0:  # If there's an obstacle
                    # Inflate the area around this obstacle
                    for dy in range(-radius_cells, radius_cells + 1):
                        for dx in range(-radius_cells, radius_cells + 1):
                            if np.sqrt(dx**2 + dy**2) <= radius:
                                if 0 <= y + dy < self.map_size[1] and 0 <= x + dx < self.map_size[0]:
                                    inflated_grid[y + dy, x + dx] = 1  # Mark as occupied

        return inflated_grid

    def smooth_path(self, path, occupancy_grid, map_data, smoothing_factor=0.7, prune_threshold=0.1):
        if len(path) < 3:
            return path
        
        path_x, path_y = [], []
        for x, y in path:
            path_x.append(x)
            path_y.append(y)
        
        tck, u = splprep([path_x, path_y], s=smoothing_factor, k=min(3, len(path) - 1))
        
        # Generate more points along the spline for a smoother path
        u_new = np.linspace(0, 1, len(path) * 3)
        x_smooth, y_smooth = splev(u_new, tck)

        pruned_path = [(x_smooth[0], y_smooth[0])]
        for i in range(1, len(x_smooth)):
            dist = np.hypot(x_smooth[i] - pruned_path[-1][0], y_smooth[i] - pruned_path[-1][1])
            
            # Check if point is clear of obstacles
            candidate_point = Node(x_smooth[i], y_smooth[i])

            if dist > prune_threshold and self.is_safe(candidate_point, occupancy_grid, map_data):
                pruned_path.append((x_smooth[i], y_smooth[i]))
                
        if len(pruned_path) > 0:
            last_point = Node(x_smooth[-1], y_smooth[-1])
            if self.is_safe(last_point, occupancy_grid, map_data):
                if pruned_path[-1] != (x_smooth[-1], y_smooth[-1]):
                    pruned_path.append((x_smooth[-1], y_smooth[-1]))

        return pruned_path

    def is_safe(self, point, inflated_grid, map_data):
        """
        Check if a point is in a safe, obstacle-free area using the inflated grid.
        Returns True if the point is safe, False otherwise.
        """
        resolution = map_data.info.resolution
        x_start_x = map_data.info.origin.position.x
        x_start_y = map_data.info.origin.position.y
        width = map_data.info.width

        # Convert the point coordinates to grid indices
        grid_x = int((point.x - x_start_x) / resolution)
        grid_y = int((point.y - x_start_y) / resolution)

        # Ensure point is within map bounds and not in an inflated obstacle
        if 0 <= grid_x < width and 0 <= grid_y < map_data.info.height:
            return inflated_grid[grid_y, grid_x] == 0  # 0 indicates free space in the grid
        return False



    def get_path(self, goal_node):
        path = []
        cur_node = goal_node
        while cur_node:
            path.append((cur_node.x, cur_node.y))
            cur_node = cur_node.parent

        path = path[::-1]  # Reverse to get path from start to goal

        smoothed_path = self.smooth_path(path, self.occupancy_grid, self.map_data)
        return smoothed_path

class RobotController:
    LOOKAHEAD_DISTANCE = 0.4
    DANGER_ANGLE_RANGE = 0.5
    DANGER_THRESHOLD = 0.7

    def __init__(self, path, goal, occupancy_grid, map_data, map_size, marker_pub, distance_threshold=0.4, speed=0.2, angular_speed=1.0):
        self.path = path
        self.goal = goal
        self.occupancy_grid = occupancy_grid
        self.map_data = map_data
        self.map_size = map_size
        self.marker_pub = marker_pub
        self.distance_threshold = distance_threshold
        self.speed = speed
        self.angular_speed = angular_speed
        self.waypoint_index = 0
        self.cmd_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber('/hsrb/base_scan', LaserScan, self.laser_callback)
        self.current_position = None
        self.current_orientation = None
        self.robot_position()

    def laser_callback(self, scan_data):
        if self.path and self.detect_obstacle(scan_data):
            rospy.loginfo("Obstacle detected, updating path.")
            self.update_path()

    def update_goal(self, new_goal):
        self.goal = new_goal

    def detect_obstacle(self, scan_data):
        # Calculate indices for forward cone
        angles_per_reading = scan_data.angle_increment
        center_index = len(scan_data.ranges) // 2
        angle_indices = int(self.DANGER_ANGLE_RANGE / angles_per_reading)
        start_idx = max(0, center_index - angle_indices)
        end_idx = min(len(scan_data.ranges), center_index + angle_indices + 1)

        # Count valid dangerous readings
        danger_count = 0
        valid_count = 0

        for i in range(start_idx, end_idx):
            range_reading = scan_data.ranges[i]
            
            if not math.isnan(range_reading) and not math.isinf(range_reading):
                valid_count += 1
                if range_reading < self.LOOKAHEAD_DISTANCE:
                    danger_count += 1

        # Only trigger if we have enough valid readings and high danger ratio
        if valid_count > 0:
            danger_ratio = danger_count / valid_count
            return danger_ratio > self.DANGER_THRESHOLD and valid_count > 5

        return False

    def update_path(self):
        cur_position = self.robot_position()
        rrt = RRT(cur_position, self.goal, self.map_size, self.occupancy_grid, self.map_data)
        new_path = rrt.plan()

        if new_path:
            self.path = new_path
            self.waypoint_index = 0
            clear_rrt_tree(self.marker_pub)
            publish_rrt_tree(self.marker_pub, rrt.node_list)
            publish_updated_path(self.marker_pub, new_path)
        else:
            rospy.logwarn("Failed to find an alternative path")

    def detect_obstacle_near_waypoint(self, waypoint):
        """Detect if there's an obstacle near a given waypoint."""
        x, y = waypoint
        grid_x = int((x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        grid_y = int((y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        if 0 <= grid_x < self.map_data.info.width and 0 <= grid_y < self.map_data.info.height:
            # Check a small radius around the waypoint
            return self.occupancy_grid[grid_y, grid_x] > 95  # Indicate obstacle
        return False


    def robot_position(self):
        tf_listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
        (trans,rot) = tf_listener.lookupTransform("map", "base_link", rospy.Time(0))

        x = trans[0]
        y = trans[1]
        z = 0.0
        (roll, pitch, t) = euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
        self.current_position = (x, y)
        self.current_orientation = t
        return (x, y)

    def move_along_path(self):
        #Move robot along the planned path using hsrb/command_velocity topic.
        while self.waypoint_index < len(self.path):
            waypoint = self.path[self.waypoint_index]
            while not self.is_at_waypoint(waypoint):
                theta = self.calculate_angle(waypoint)

                if abs(theta) > 0.05:
                    rotation = Twist()
                    rotation.angular.z = self.angular_speed if theta > 0 else -self.angular_speed
                    self.cmd_pub.publish(rotation)
                else:
                    move = Twist()
                    move.linear.x = self.speed
                    self.cmd_pub.publish(move)
                
                rospy.sleep(0.1)
            
            self.waypoint_index += 1


        # Stop the robot at the end
        self.stop_robot()

    def calculate_distance(self, waypoint):
        """Calculate distance from current position to a waypoint."""
        self.robot_position()
        return math.sqrt((waypoint[0] - self.current_position[0])**2 + (waypoint[1] - self.current_position[1]) ** 2)

    def calculate_angle(self, waypoint):
        """Calculate angle difference to face the waypoint."""
        self.robot_position()
        target_angle = math.atan2(waypoint[1] - self.current_position[1],
                                  waypoint[0] - self.current_position[0])
        angle_difference = target_angle - self.current_orientation
        return math.atan2(math.sin(angle_difference), math.cos(angle_difference))  # Normalize angle

    def is_at_waypoint(self, waypoint):
        """Check if the robot is within threshold distance to the waypoint."""
        return self.calculate_distance(waypoint) < self.distance_threshold

    def stop_robot(self):
        """Stop the robot by publishing zero velocities."""
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.cmd_pub.publish(stop_twist)
        rospy.loginfo("Robot has reached the final waypoint and stopped.")

def publish_rrt_tree(marker_pub, node_list):
    """Publish both RRT nodes and their connections (tree structure)"""
    if not node_list or len(node_list) == 0:
        rospy.logwarn("No nodes to publish")
        return
    
    # Publish tree connections first (if there are enough nodes)
    if len(node_list) >= 2:
        tree_marker = Marker()
        tree_marker.header.frame_id = 'map'
        tree_marker.header.stamp = rospy.Time.now()
        tree_marker.ns = 'tree'
        tree_marker.id = MARKER_ID_TREE  # ID 0
        tree_marker.type = Marker.LINE_LIST
        tree_marker.action = Marker.ADD
        tree_marker.lifetime = rospy.Duration(0)
        tree_marker.scale.x = 0.02
        tree_marker.color.r = 0.5
        tree_marker.color.g = 0.5
        tree_marker.color.b = 0.8
        tree_marker.color.a = 0.7
        tree_marker.pose.orientation.w = 1.0
        
        connection_count = 0
        for node in node_list:
            if node.parent is not None:
                # Add parent point
                p1 = Point()
                p1.x = node.parent.x
                p1.y = node.parent.y
                p1.z = 0.0
                tree_marker.points.append(p1)
                
                # Add child point
                p2 = Point()
                p2.x = node.x
                p2.y = node.y
                p2.z = 0.0
                tree_marker.points.append(p2)
                connection_count += 1
        
        if connection_count > 0:
            marker_pub.publish(tree_marker)
            rospy.loginfo(f"Published tree with {connection_count} connections")
    
    # Publish nodes as spheres
    for i, node in enumerate(node_list):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'node'
        marker.id = MARKER_ID_NODE_START + i  # IDs 1000+
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        # Color code nodes: start node green, goal area nodes blue, others red
        if i == 0:  # Start node
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:  # Regular nodes
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
        marker.color.a = 1.0
        marker.pose.position.x = node.x
        marker.pose.position.y = node.y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker_pub.publish(marker)
    
    rospy.loginfo(f"Published {len(node_list)} nodes with tree structure")

def clear_rrt_tree(marker_pub, max_nodes=1000):
    """Clear all RRT tree visualization (both nodes and connections)"""
    # Clear tree connections
    marker = Marker()
    marker.header.frame_id = 'map'
    marker.header.stamp = rospy.Time.now()
    marker.ns = 'tree'
    marker.id = MARKER_ID_TREE
    marker.action = Marker.DELETE
    marker_pub.publish(marker)
    
    # Clear all nodes
    for i in range(max_nodes):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'node'
        marker.id = MARKER_ID_NODE_START + i
        marker.action = Marker.DELETE
        marker_pub.publish(marker)
 
def publish_path(marker_pub, path):
    """Publish the initial planned path"""
    if not path:
        rospy.logwarn("No path to publish")
        return
    
    path_marker = Marker()
    path_marker.header.frame_id = 'map'
    path_marker.header.stamp = rospy.Time.now()
    path_marker.ns = 'path'
    path_marker.id = MARKER_ID_PATH  # ID 2000
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.lifetime = rospy.Duration(0)
    path_marker.scale.x = 0.05
    path_marker.color.r = 0.0
    path_marker.color.g = 1.0
    path_marker.color.b = 0.0
    path_marker.color.a = 1.0
    path_marker.pose.orientation.w = 1.0

    for waypoint in path:
        p = Point()
        p.x = waypoint[0]
        p.y = waypoint[1]
        p.z = 0.0
        path_marker.points.append(p)

    marker_pub.publish(path_marker)

def publish_updated_path(marker_pub, path):
    """Publish an updated/replanned path (different color and ID from initial path)"""
    if not path:
        rospy.logwarn("No updated path to publish")
        return

    path_marker = Marker()
    path_marker.header.frame_id = 'map'
    path_marker.header.stamp = rospy.Time.now()
    path_marker.ns = 'path'
    path_marker.id = MARKER_ID_UPDATED_PATH  # ID 4000
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.lifetime = rospy.Duration(0)
    path_marker.scale.x = 0.06
    path_marker.color.r = 1.0  # Orange for updated path
    path_marker.color.g = 0.5
    path_marker.color.b = 0.0
    path_marker.color.a = 1.0
    path_marker.pose.orientation.w = 1.0

    for waypoint in path:
        p = Point()
        p.x = waypoint[0]
        p.y = waypoint[1]
        p.z = 0.0
        path_marker.points.append(p)

    marker_pub.publish(path_marker)

def publish_start_goal(marker_pub, start, goal):
    """Publish start and goal markers"""
    # Start marker (green sphere)
    start_marker = Marker()
    start_marker.header.frame_id = "map"
    start_marker.header.stamp = rospy.Time.now()
    start_marker.ns = "start"
    start_marker.id = MARKER_ID_START  # ID 3000
    start_marker.type = Marker.SPHERE
    start_marker.action = Marker.ADD
    start_marker.lifetime = rospy.Duration(0)
    start_marker.scale.x = 0.3
    start_marker.scale.y = 0.3
    start_marker.scale.z = 0.3
    start_marker.color.r = 0.0
    start_marker.color.g = 1.0
    start_marker.color.b = 0.0
    start_marker.color.a = 1.0
    start_marker.pose.position.x = start[0]
    start_marker.pose.position.y = start[1]
    start_marker.pose.position.z = 0.0
    start_marker.pose.orientation.w = 1.0
    
    marker_pub.publish(start_marker)
    
    # Goal marker (blue sphere)
    goal_marker = Marker()
    goal_marker.header.frame_id = "map"
    goal_marker.header.stamp = rospy.Time.now()
    goal_marker.ns = "goal"
    goal_marker.id = MARKER_ID_GOAL  # ID 3001
    goal_marker.type = Marker.SPHERE
    goal_marker.action = Marker.ADD
    goal_marker.lifetime = rospy.Duration(0)
    goal_marker.scale.x = 0.3
    goal_marker.scale.y = 0.3
    goal_marker.scale.z = 0.3
    goal_marker.color.r = 0.0
    goal_marker.color.g = 0.0
    goal_marker.color.b = 1.0
    goal_marker.color.a = 1.0
    goal_marker.pose.position.x = goal[0]
    goal_marker.pose.position.y = goal[1]
    goal_marker.pose.position.z = 0.0
    goal_marker.pose.orientation.w = 1.0
    
    marker_pub.publish(goal_marker)

def map_callback(msg):
    map_width = msg.info.width
    map_height = msg.info.height
    map_size = (map_width, map_height)
    rospy.loginfo(f"Map received: {map_width}x{map_height}")

    # Convert OccupancyGrid data to a numpy array
    occupancy_grid = np.array(msg.data).reshape((map_height, map_width))
    map_data = msg

    # Get initial robot position
    tf_listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
    (trans,rot) = tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
    start = (trans[0], trans[1])
    
    # Define multiple waypoints to visit sequentially
    waypoints = [
        (1.0, 0.0),
        (6.0, 4.0),
        (3.0, 3.0),
        (9.0, 3.0)
    ]

    # Wait for rviz to begin
    rospy.sleep(5)
    
    # Process each waypoint
    current_start = start

    for i, goal in enumerate(waypoints):
        rospy.loginfo(f"Planning to waypoint {i+1}/{len(waypoints)}: {goal}")
        
        # Clear previous RRT tree visualization before planning new path
        if i > 0:  # Don't clear on first waypoint
            clear_rrt_tree(marker_pub)
            rospy.sleep(0.1)  # Small delay to ensure clearing happens
        
        # Publish start and goal markers for current segment
        publish_start_goal(marker_pub, current_start, goal)
        
        # Run RRT for this segment
        rrt = RRT(current_start, goal, map_size, occupancy_grid, map_data)
        path = rrt.plan()
        rospy.loginfo(f"Path to waypoint {i+1}: {path}")
        
        if path:
            # Publish RRT tree visualization
            publish_rrt_tree(marker_pub, rrt.node_list)
            rospy.sleep(0.05)
            
            # Publish the planned path
            publish_path(marker_pub, path)
            rospy.sleep(0.05)
            
            # Move the robot along the path
            controller = RobotController(path, goal, occupancy_grid, map_data, map_size, marker_pub)
            controller.move_along_path()
            
            # Update start position for next waypoint
            current_start = goal
            
            rospy.loginfo(f"Reached waypoint {i+1}")
            rospy.sleep(1.0)  # Brief pause at each waypoint
        else:
            rospy.logwarn(f"No valid path found to waypoint {i+1}")
            break  # Stop if we can't reach a waypoint
    
    rospy.loginfo("Completed all waypoints!")


if __name__ == '__main__':
    rospy.init_node('rrt_path_planning')
    tf_listener = TransformListener()
    marker_pub = rospy.Publisher('rrt_visualization_marker', Marker, queue_size=10)
    rospy.Subscriber('map', OccupancyGrid, map_callback)
    rospy.loginfo("RRT path planner initialized, waiting for map...")
    rospy.spin()
