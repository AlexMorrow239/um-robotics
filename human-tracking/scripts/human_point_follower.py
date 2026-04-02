#!/usr/bin/env python3
"""
Human Point Follower - Unified State Machine

Combines human tracking and pointing detection into a single node
with consistent state management to prevent race conditions.

States:
- SEARCHING: Rotate base 360° scanning for humans
- TRACKING: Keep detected human centered in frame
- DETECTED: Validate 3-second pointing hold (base stopped)
- NAVIGATING: Move to pointed location via move_base
"""

import rospy
import numpy as np
import math
from enum import Enum
import tf2_ros
import tf2_geometry_msgs

from std_msgs.msg import Bool, Float64, String
from geometry_msgs.msg import PoseArray, Twist, PointStamped, Vector3Stamped
from sensor_msgs.msg import CameraInfo, JointState, Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge


class State(Enum):
    IDLE = 0       # Waiting for start signal
    SEARCHING = 1
    TRACKING = 2
    DETECTED = 3
    NAVIGATING = 4
    COMPLETED = 5  # Demo finished, waiting for reset


class HumanPointFollower:
    def __init__(self):
        rospy.init_node('human_point_follower')

        # Current state - start in IDLE, waiting for start signal
        self.state = State.IDLE

        # ========== Parameters from both scripts ==========
        # Scanning parameters (from human_tracker.py)
        self.scan_angular_speed = rospy.get_param('/scanning/angular_speed', 0.3)
        self.head_tilt_angle = rospy.get_param('/scanning/head_tilt_angle', 0.0)
        self.head_pan_angle = rospy.get_param('/scanning/head_pan_angle', 0.0)

        # Tracking parameters (from human_tracker.py)
        self.control_rate = rospy.get_param('/tracking/control_rate', 20)
        self.target_x = rospy.get_param('/tracking/target_x', 320)
        self.target_y = rospy.get_param('/tracking/target_y', 240)
        self.deadband_x = rospy.get_param('/tracking/deadband_x', 10)
        self.min_confidence = rospy.get_param('/tracking/min_confidence', 0.3)

        # Initial centering parameters
        self.centering_timeout = rospy.get_param('/tracking/centering_timeout', 5.0)

        # Pointing parameters (from pointing_detector.py)
        self.pointing_min_confidence = rospy.get_param('/pointing/min_confidence', 0.1)
        self.straightness_threshold = rospy.get_param('/pointing/straightness_threshold', 35.0)
        self.min_extension_ratio = rospy.get_param('/pointing/min_extension_ratio', 0.5)
        self.hold_duration = rospy.get_param('/pointing/hold_duration', 3.0)
        self.direction_stability_threshold = rospy.get_param('/pointing/direction_stability_deg', 10.0)
        self.direction_history_size = rospy.get_param('/pointing/direction_history_size', 10)
        self.keypoint_smoothing_alpha = rospy.get_param('/pointing/keypoint_smoothing_alpha', 0.2)
        self.max_wrist_below_elbow_px = rospy.get_param('/pointing/max_wrist_below_elbow_px', 30)

        # ========== State tracking variables ==========
        # Scanning state
        self.total_rotation = 0.0
        self.last_time = None
        self.full_rotation_threshold = 2 * math.pi

        # Detection state
        self.human_detected = False
        self.detection_count = 0
        self.detection_threshold = 5
        self.lost_human_time = None
        self.lost_human_timeout = 2.0

        # Tracking state
        self.current_keypoints = None

        # Initial centering state (for TRACKING entry)
        self.initial_centering_complete = False
        self.centering_history = []  # Sliding window of centering results
        self.centering_history_size = 15  # Window size
        self.centering_success_ratio = 0.7  # Need 70% centered
        self.centering_start_time = None

        # Camera state
        self.camera_width = 640
        self.camera_height = 480
        self.camera_info_received = False
        self.fx = 554.383
        self.fy = 554.383
        self.cx = 320.0
        self.cy = 240.0

        # Joint state
        self.current_head_pan = None
        self.current_head_tilt = None
        self.joint_states_received = False

        # Head joint limits
        self.head_pan_min = -3.839
        self.head_pan_max = 1.745
        self.head_tilt_min = -0.524
        self.head_tilt_max = 1.047

        # Pointing detection state
        self.current_arm_keypoints = None
        self.smoothed_arm_keypoints = None  # EMA-smoothed keypoints
        self.pointing_start_time = None
        self.last_pointing_direction = None
        self.direction_history = []  # Sliding window of pointing directions
        self.pointing_confirmed = False
        self.active_arm = None

        # Depth image
        self.depth_image = None
        self.bridge = CvBridge()

        # Navigation state
        self.target_ground_point = None
        self.nav_phase = 0  # 0=rotating, 1=driving
        self.nav_start_time = None
        self.nav_rotate_start_time = None
        self.nav_drive_start_time = None
        self.nav_timeout = 30.0  # seconds
        self.initial_target_distance = 0.0
        self.target_angle = 0.0  # angle to rotate

        # TF2 for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ========== RViz Visualization ==========
        self.human_position_msg = None  # Cached human position from Isaac Sim
        self.ground_truth_point_msg = None  # Cached ground truth from arm controller
        self.target_ground_point_odom = None  # Robot computed target in odom frame (for fixed marker)

        # ========== Subscribers ==========
        # Start/Reset signal - publish True to start demo, any message resets from COMPLETED
        self.start_sub = rospy.Subscriber(
            '/human_point_follower/start',
            Bool,
            self.start_callback,
            queue_size=1
        )

        # Human detection (for SEARCHING/TRACKING)
        self.detection_sub = rospy.Subscriber(
            '/openpose/human_detected',
            Bool,
            self.detection_callback,
            queue_size=1
        )

        # Body keypoints (for TRACKING)
        self.keypoints_sub = rospy.Subscriber(
            '/openpose/keypoints',
            PoseArray,
            self.keypoints_callback,
            queue_size=1
        )

        # Arm keypoints (for DETECTED)
        self.arm_keypoints_sub = rospy.Subscriber(
            '/openpose/arm_keypoints',
            PoseArray,
            self.arm_keypoints_callback,
            queue_size=1
        )

        # Camera info
        self.camera_info_sub = rospy.Subscriber(
            '/hsrb/head_rgbd_sensor/rgb/camera_info',
            CameraInfo,
            self.camera_info_callback,
            queue_size=1
        )

        # Depth image (for pointing 3D calculation)
        self.depth_sub = rospy.Subscriber(
            rospy.get_param('/camera/depth_topic', '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'),
            Image,
            self.depth_callback,
            queue_size=1
        )

        # Joint states
        joint_state_topics = [
            '/hsrb/joint_states',
            '/joint_states',
            '/hsrb/robot_state/joint_states'
        ]
        for topic in joint_state_topics:
            try:
                rospy.wait_for_message(topic, JointState, timeout=1.0)
                self.joint_state_sub = rospy.Subscriber(topic, JointState, self.joint_state_callback, queue_size=1)
                rospy.loginfo(f"Subscribed to joint states on: {topic}")
                break
            except rospy.ROSException:
                continue

        # Visualization subscriptions (from Isaac Sim world)
        self.human_position_sub = rospy.Subscriber(
            '/human/position',
            PointStamped,
            self.human_position_callback,
            queue_size=1
        )
        self.ground_truth_sub = rospy.Subscriber(
            '/pointing/ground_truth',
            PointStamped,
            self.ground_truth_callback,
            queue_size=1
        )

        # ========== Publishers ==========
        # Base velocity (SEARCHING/TRACKING only)
        self.cmd_vel_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)

        # Head joint commands
        self.head_pan_pub = rospy.Publisher('/hsrb/head_pan_joint/command', Float64, queue_size=1)
        self.head_tilt_pub = rospy.Publisher('/hsrb/head_tilt_joint/command', Float64, queue_size=1)

        # Pointing status publishers
        self.pointing_detected_pub = rospy.Publisher('/pointing/detected', Bool, queue_size=1)
        self.pointing_hold_pub = rospy.Publisher('/pointing/hold_confirmed', Bool, queue_size=1)
        self.pointing_vector_pub = rospy.Publisher('/pointing/vector', Vector3Stamped, queue_size=1)
        self.pointing_wrist_3d_pub = rospy.Publisher('/pointing/wrist_3d', PointStamped, queue_size=1)
        self.pointing_ground_pub = rospy.Publisher('/pointing/ground_point', PointStamped, queue_size=1)
        self.active_arm_pub = rospy.Publisher('/pointing/active_arm', String, queue_size=1)

        # State publisher for monitoring
        self.state_pub = rospy.Publisher('/human_point_follower/state', String, queue_size=1)

        # RViz visualization markers
        self.marker_pub = rospy.Publisher('/pointing/markers', MarkerArray, queue_size=1)

        rospy.loginfo("Human Point Follower initialized")
        rospy.loginfo(f"  States: IDLE -> SEARCHING -> TRACKING -> DETECTED -> NAVIGATING -> COMPLETED")
        rospy.loginfo(f"  Hold duration: {self.hold_duration}s")
        rospy.loginfo(f"  Start topic: /human_point_follower/start (std_msgs/Bool)")
        rospy.loginfo(f"  State topic: /human_point_follower/state (std_msgs/String)")
        rospy.sleep(1.0)  # Give publishers time to connect

    # ========== Callbacks ==========

    def start_callback(self, msg):
        """Handle start/reset signal."""
        if self.state == State.IDLE:
            if msg.data:  # Only start on True
                rospy.loginfo("Received START signal - beginning demo")
                self.transition_to_searching()
        elif self.state == State.COMPLETED:
            # Any message resets from COMPLETED back to IDLE
            rospy.loginfo("Received RESET signal - ready for next demo")
            self.transition_to_idle()

    def detection_callback(self, msg):
        """Handle human detection status updates."""
        self.human_detected = msg.data

        if self.state == State.SEARCHING:
            if msg.data:
                self.detection_count += 1
                rospy.loginfo(f"Human detected! Count: {self.detection_count}/{self.detection_threshold}")
                if self.detection_count >= self.detection_threshold:
                    self.transition_to_tracking()
            else:
                if self.detection_count > 0:
                    rospy.loginfo("Lost human detection, resetting counter")
                self.detection_count = 0

        elif self.state == State.TRACKING:
            if not msg.data:
                if self.lost_human_time is None:
                    self.lost_human_time = rospy.Time.now()
                    rospy.logwarn("Lost human during tracking!")
                else:
                    elapsed = (rospy.Time.now() - self.lost_human_time).to_sec()
                    if elapsed > self.lost_human_timeout:
                        rospy.logerr(f"Human lost for {elapsed:.1f}s - returning to SEARCHING")
                        self.transition_to_searching()
            else:
                if self.lost_human_time is not None:
                    rospy.loginfo("Human reacquired!")
                    self.lost_human_time = None

        elif self.state == State.DETECTED:
            # During DETECTED state, if we lose the human entirely, go back to tracking
            if not msg.data:
                if self.lost_human_time is None:
                    self.lost_human_time = rospy.Time.now()
                else:
                    elapsed = (rospy.Time.now() - self.lost_human_time).to_sec()
                    if elapsed > self.lost_human_timeout:
                        rospy.logwarn("Human lost during pointing validation - returning to TRACKING")
                        self.transition_to_tracking()
            else:
                self.lost_human_time = None

    def keypoints_callback(self, msg):
        """Store body keypoints for tracking."""
        if len(msg.poses) > 0:
            self.current_keypoints = msg.poses
        else:
            self.current_keypoints = None

    def arm_keypoints_callback(self, msg):
        """Store arm keypoints for pointing detection with EMA smoothing.

        Expected keypoint order (8 total):
        - Index 0: Nose (head keypoint for pointing origin)
        - Index 1: Neck (fallback head keypoint)
        - Index 2-4: Left shoulder, elbow, wrist
        - Index 5-7: Right shoulder, elbow, wrist
        """
        if len(msg.poses) == 8:
            if self.smoothed_arm_keypoints is None:
                # First frame - initialize with current values
                self.smoothed_arm_keypoints = list(msg.poses)
            else:
                # Apply EMA smoothing to each keypoint
                alpha = self.keypoint_smoothing_alpha
                for i, pose in enumerate(msg.poses):
                    self.smoothed_arm_keypoints[i].position.x = (
                        alpha * pose.position.x +
                        (1 - alpha) * self.smoothed_arm_keypoints[i].position.x
                    )
                    self.smoothed_arm_keypoints[i].position.y = (
                        alpha * pose.position.y +
                        (1 - alpha) * self.smoothed_arm_keypoints[i].position.y
                    )
                    # Keep confidence as-is (no smoothing on confidence)
                    self.smoothed_arm_keypoints[i].position.z = pose.position.z
            self.current_arm_keypoints = self.smoothed_arm_keypoints

    def camera_info_callback(self, msg):
        """Update camera info."""
        if not self.camera_info_received:
            self.camera_width = msg.width
            self.camera_height = msg.height
            self.target_x = msg.width / 2
            self.target_y = msg.height / 2
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            self.camera_info_received = True
            rospy.loginfo(f"Camera info received: {msg.width}x{msg.height}")

    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            if msg.encoding == '16UC1':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            elif msg.encoding == '32FC1':
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1") * 1000
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            rospy.logerr_throttle(5.0, f"Depth conversion error: {e}")

    def joint_state_callback(self, msg):
        """Track current head joint positions."""
        try:
            if 'head_pan_joint' in msg.name:
                pan_idx = msg.name.index('head_pan_joint')
                self.current_head_pan = msg.position[pan_idx]
            if 'head_tilt_joint' in msg.name:
                tilt_idx = msg.name.index('head_tilt_joint')
                self.current_head_tilt = msg.position[tilt_idx]
            if not self.joint_states_received and self.current_head_pan is not None:
                rospy.loginfo(f"Current head position - Pan: {math.degrees(self.current_head_pan):.1f}°")
                self.joint_states_received = True
        except (ValueError, IndexError):
            pass

    def human_position_callback(self, msg):
        """Store human position from Isaac Sim world for visualization."""
        self.human_position_msg = msg

    def ground_truth_callback(self, msg):
        """Store ground truth intersection point from arm controller."""
        self.ground_truth_point_msg = msg

    # ========== State transitions ==========

    def transition_to_idle(self):
        """Transition to IDLE state - waiting for start signal."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("STATE: IDLE - Waiting for start signal")
        rospy.loginfo("  Publish: rostopic pub /human_point_follower/start std_msgs/Bool 'data: true'")
        rospy.loginfo("=" * 50)

        self.state = State.IDLE
        self.stop_base()
        self.reset_detection_state()
        self.reset_pointing_state()

    def transition_to_completed(self):
        """Transition to COMPLETED state - demo finished."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("STATE: COMPLETED - Demo finished!")
        rospy.loginfo("  Publish any message to /human_point_follower/start to reset")
        rospy.loginfo("=" * 50)

        self.state = State.COMPLETED
        self.stop_base()
        self.reset_detection_state()
        self.reset_pointing_state()

    def transition_to_searching(self):
        """Transition to SEARCHING state."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("STATE: SEARCHING")
        rospy.loginfo("=" * 50)

        self.state = State.SEARCHING
        self.stop_base()
        self.reset_detection_state()
        self.reset_pointing_state()
        self.total_rotation = 0.0
        self.last_time = rospy.Time.now()

        # Clear visualization markers for fresh demo
        self.clear_markers()

        # Position head for scanning
        self.set_head_position(self.head_pan_angle, self.head_tilt_angle)

    def transition_to_tracking(self):
        """Transition to TRACKING state."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("STATE: TRACKING - Human confirmed!")
        rospy.loginfo("=" * 50)

        self.state = State.TRACKING
        self.stop_base()
        self.lost_human_time = None
        self.reset_pointing_state()

        # Initialize centering sub-phase
        self.initial_centering_complete = False
        self.centering_history = []  # Reset sliding window
        self.centering_start_time = rospy.Time.now()
        rospy.loginfo("Starting initial centering phase - pointing detection disabled")

        # Center head for tracking
        self.set_head_position(0.0, 0.0)

    def return_to_tracking(self):
        """Return to TRACKING state without resetting centering (from DETECTED).

        Used when returning from DETECTED state - the human was already centered
        when pointing was first detected, so we skip re-centering.
        """
        rospy.loginfo("=" * 50)
        rospy.loginfo("STATE: TRACKING - Returning (centering preserved)")
        rospy.loginfo("=" * 50)

        self.state = State.TRACKING
        self.stop_base()
        self.lost_human_time = None
        self.reset_pointing_state()
        # Note: initial_centering_complete is NOT reset - we stay centered

    def transition_to_detected(self):
        """Transition to DETECTED state."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("STATE: DETECTED - Validating pointing gesture")
        rospy.loginfo("=" * 50)

        self.state = State.DETECTED
        self.stop_base()  # Stop base movement during pointing validation
        self.pointing_start_time = rospy.Time.now()
        self.pointing_confirmed = False

    def transition_to_navigating(self, target_point):
        """Transition to NAVIGATING state."""
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"STATE: NAVIGATING to ({target_point[0]:.2f}, {target_point[1]:.2f})")
        rospy.loginfo("=" * 50)

        self.state = State.NAVIGATING
        self.stop_base()
        self.target_ground_point = target_point
        self.nav_phase = 0  # Start with rotation phase
        self.nav_start_time = rospy.Time.now()
        self.nav_rotate_start_time = rospy.Time.now()

        # Transform target from base_link to odom for fixed RViz marker
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            # Create point in base_link frame
            pt_base = PointStamped()
            pt_base.header.frame_id = "base_link"
            pt_base.header.stamp = rospy.Time.now()
            pt_base.point.x = target_point[0]
            pt_base.point.y = target_point[1]
            pt_base.point.z = 0.0
            # Transform to odom
            pt_odom = tf2_geometry_msgs.do_transform_point(pt_base, transform)
            self.target_ground_point_odom = [pt_odom.point.x, pt_odom.point.y]
            rospy.loginfo(f"  Target in odom: ({pt_odom.point.x:.2f}, {pt_odom.point.y:.2f})")
        except tf2_ros.TransformException as e:
            rospy.logwarn(f"Could not transform target to odom: {e}")
            self.target_ground_point_odom = None

        # Calculate and store the angle to rotate and distance to drive
        # These are fixed at the start and we use time-based dead reckoning
        self.target_angle = math.atan2(target_point[1], target_point[0])
        self.initial_target_distance = math.sqrt(target_point[0]**2 + target_point[1]**2)

        rospy.loginfo(f"  Target angle: {math.degrees(self.target_angle):.1f}°")
        rospy.loginfo(f"  Target distance: {self.initial_target_distance:.2f}m")

    # ========== State reset helpers ==========

    def reset_detection_state(self):
        """Reset detection counters."""
        self.detection_count = 0
        self.lost_human_time = None

    def reset_pointing_state(self):
        """Reset pointing state variables."""
        self.pointing_start_time = None
        self.last_pointing_direction = None
        self.direction_history = []  # Clear direction history buffer
        self.smoothed_arm_keypoints = None  # Reset smoothed keypoints
        self.pointing_confirmed = False
        self.active_arm = None
        self.pointing_detected_pub.publish(Bool(False))
        self.pointing_hold_pub.publish(Bool(False))
        self.active_arm_pub.publish(String(data=''))  # Clear active arm for ground truth

    # ========== Robot control ==========

    def stop_base(self):
        """Stop base movement."""
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)

    def rotate_base(self, angular_velocity):
        """Send rotation command to base."""
        cmd_twist = Twist()
        cmd_twist.angular.z = angular_velocity
        self.cmd_vel_pub.publish(cmd_twist)

    def set_head_position(self, pan, tilt):
        """Set head position."""
        rospy.loginfo(f"Setting head to Pan: {math.degrees(pan):.1f}°, Tilt: {math.degrees(tilt):.1f}°")
        pan_msg = Float64(data=pan)
        tilt_msg = Float64(data=tilt)
        for _ in range(3):
            self.head_pan_pub.publish(pan_msg)
            self.head_tilt_pub.publish(tilt_msg)
            rospy.sleep(0.1)
        rospy.sleep(1.0)

    # ========== SEARCHING state logic ==========

    def run_searching(self, current_time):
        """Execute SEARCHING state - rotate base looking for humans."""
        if self.last_time:
            dt = (current_time - self.last_time).to_sec()
            self.total_rotation += abs(self.scan_angular_speed * dt)

            if self.total_rotation >= self.full_rotation_threshold:
                rospy.loginfo(f"Full rotation completed ({math.degrees(self.total_rotation):.1f}°) - no human found")
                # Reset and continue searching
                self.total_rotation = 0.0

        self.rotate_base(self.scan_angular_speed)
        rospy.loginfo_throttle(2.0, f"Scanning... {math.degrees(self.total_rotation):.1f}° / 360°")
        self.last_time = current_time

    # ========== TRACKING state logic ==========

    def get_human_center(self):
        """Get center position of human from keypoints."""
        if self.current_keypoints is None or len(self.current_keypoints) == 0:
            return None, None

        x_sum, y_sum, count = 0, 0, 0
        for pose in self.current_keypoints:
            if pose.position.z > self.min_confidence:
                x_sum += pose.position.x
                y_sum += pose.position.y
                count += 1

        if count > 0:
            return x_sum / count, y_sum / count
        return None, None

    def run_initial_centering(self, current_time):
        """Execute initial centering sub-phase of TRACKING state.

        Uses a sliding window approach: centering completes when 70% of the last
        15 frames have the human centered (within deadband).

        Returns True when centering is complete (or timed out), False if still centering.
        """
        # Check for timeout
        elapsed = (current_time - self.centering_start_time).to_sec()
        if elapsed >= self.centering_timeout:
            rospy.logwarn(f"Centering timeout after {elapsed:.1f}s - proceeding with tracking")
            self.initial_centering_complete = True
            self.stop_base()
            return True

        # Get human position
        center_x, center_y = self.get_human_center()
        if center_x is None:
            # Lost human during centering - record as not centered
            self.centering_history.append(False)
            if len(self.centering_history) > self.centering_history_size:
                self.centering_history.pop(0)
            rospy.loginfo_throttle(1.0, "Centering: waiting for human keypoints...")
            return False

        error_x = center_x - self.target_x
        is_centered = abs(error_x) < self.deadband_x

        # Add result to sliding window
        self.centering_history.append(is_centered)
        if len(self.centering_history) > self.centering_history_size:
            self.centering_history.pop(0)

        # Check if enough frames are centered (once window is full)
        if len(self.centering_history) >= self.centering_history_size:
            centered_count = sum(self.centering_history)
            ratio = centered_count / len(self.centering_history)

            if ratio >= self.centering_success_ratio:
                rospy.loginfo("=" * 40)
                rospy.loginfo("Initial centering complete - pointing detection enabled")
                rospy.loginfo("=" * 40)
                self.initial_centering_complete = True
                self.stop_base()
                return True

        # Apply correction if not currently centered
        if not is_centered:
            base_gain = 0.003
            base_angular_vel = -error_x * base_gain
            max_base_velocity = 0.5
            base_angular_vel = max(-max_base_velocity, min(max_base_velocity, base_angular_vel))
            self.rotate_base(base_angular_vel)
        else:
            self.stop_base()

        # Log progress
        centered_count = sum(self.centering_history)
        needed = int(self.centering_history_size * self.centering_success_ratio)
        rospy.loginfo_throttle(0.5,
            f"Centering: {centered_count}/{len(self.centering_history)} frames centered (need {needed})")

        return False

    def run_tracking(self, current_time):
        """Execute TRACKING state - keep human centered, check for pointing."""
        # Handle initial centering sub-phase
        if not self.initial_centering_complete:
            self.run_initial_centering(current_time)
            return  # Don't check for pointing until centered

        # Normal tracking behavior after centering is complete
        # First check for pointing gesture
        pointing_arm = self.check_pointing_gesture()
        if pointing_arm is not None:
            self.active_arm = pointing_arm  # Set active arm before transitioning
            self.transition_to_detected()
            return

        # Track human with base rotation
        center_x, center_y = self.get_human_center()
        if center_x is None:
            return

        error_x = center_x - self.target_x
        if abs(error_x) < self.deadband_x:
            self.stop_base()
            rospy.loginfo_throttle(2.0, f"Human centered at x={center_x:.0f}")
            return

        # Proportional base control
        base_gain = 0.003
        base_angular_vel = -error_x * base_gain
        max_base_velocity = 0.5
        base_angular_vel = max(-max_base_velocity, min(max_base_velocity, base_angular_vel))

        self.rotate_base(base_angular_vel)
        rospy.loginfo_throttle(0.5, f"Tracking - Error: {error_x:.0f}px | Base vel: {base_angular_vel:.3f}")

    # ========== DETECTED state logic (pointing validation) ==========

    def check_pointing_gesture(self):
        """Check if arm is in pointing pose. Returns arm name ('left'/'right') or None.

        Always prefers left arm when both are detected as pointing.

        Keypoint indices (8 total):
        - 0: Nose, 1: Neck (head keypoints)
        - 2-4: Left shoulder, elbow, wrist
        - 5-7: Right shoulder, elbow, wrist
        """
        if self.current_arm_keypoints is None or len(self.current_arm_keypoints) != 8:
            return None

        # Extract arm data (indices offset by 2 for nose/neck)
        left_arm = {
            'shoulder': (self.current_arm_keypoints[2].position.x,
                        self.current_arm_keypoints[2].position.y,
                        self.current_arm_keypoints[2].position.z),
            'elbow': (self.current_arm_keypoints[3].position.x,
                     self.current_arm_keypoints[3].position.y,
                     self.current_arm_keypoints[3].position.z),
            'wrist': (self.current_arm_keypoints[4].position.x,
                     self.current_arm_keypoints[4].position.y,
                     self.current_arm_keypoints[4].position.z)
        }
        right_arm = {
            'shoulder': (self.current_arm_keypoints[5].position.x,
                        self.current_arm_keypoints[5].position.y,
                        self.current_arm_keypoints[5].position.z),
            'elbow': (self.current_arm_keypoints[6].position.x,
                     self.current_arm_keypoints[6].position.y,
                     self.current_arm_keypoints[6].position.z),
            'wrist': (self.current_arm_keypoints[7].position.x,
                     self.current_arm_keypoints[7].position.y,
                     self.current_arm_keypoints[7].position.z)
        }

        left_pointing, _ = self.is_arm_pointing(left_arm, "Left")
        right_pointing, _ = self.is_arm_pointing(right_arm, "Right")

        # Left arm preference when both are pointing
        if left_pointing:
            return 'left'
        elif right_pointing:
            return 'right'
        return None

    def is_arm_pointing(self, arm, arm_name="unknown"):
        """Check if arm is in pointing pose."""
        shoulder, elbow, wrist = arm['shoulder'], arm['elbow'], arm['wrist']

        # Check confidence
        if shoulder[2] < self.pointing_min_confidence or \
           elbow[2] < self.pointing_min_confidence or \
           wrist[2] < self.pointing_min_confidence:
            return False, None

        # Calculate vectors
        upper_arm = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        lower_arm = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])

        upper_len = np.linalg.norm(upper_arm)
        lower_len = np.linalg.norm(lower_arm)

        if upper_len < 1 or lower_len < 1:
            return False, None

        # Check straightness
        upper_norm = upper_arm / upper_len
        lower_norm = lower_arm / lower_len
        dot = np.clip(np.dot(upper_norm, lower_norm), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(dot))

        if angle_deg > self.straightness_threshold:
            return False, None

        # Check extension
        full_arm = np.array([wrist[0] - shoulder[0], wrist[1] - shoulder[1]])
        extension = np.linalg.norm(full_arm)
        max_ext = upper_len + lower_len
        extension_ratio = extension / max_ext if max_ext > 0 else 0

        if extension_ratio < self.min_extension_ratio:
            return False, None

        # === Neutral arm filter 1: Wrist-above-elbow check ===
        # In image coordinates, Y increases downward
        # A neutral hanging arm has wrist significantly below elbow
        wrist_below_elbow = wrist[1] - elbow[1]
        if wrist_below_elbow > self.max_wrist_below_elbow_px:
            rospy.loginfo_throttle(2.0, f"{arm_name} arm rejected: wrist {wrist_below_elbow:.0f}px below elbow (max {self.max_wrist_below_elbow_px})")
            return False, None

        rospy.loginfo_throttle(1.0, f"{arm_name} arm: angle={angle_deg:.1f}°, ext={extension_ratio:.2f}, wrist_below={wrist_below_elbow:.0f}px")
        return True, tuple(lower_norm)

    def run_detected(self, current_time):
        """Execute DETECTED state - validate pointing hold.

        Keypoint indices (8 total):
        - 0: Nose, 1: Neck (head keypoints for face-to-hand vector)
        - 2-4: Left shoulder, elbow, wrist
        - 5-7: Right shoulder, elbow, wrist
        """
        # Base is stopped during this state

        if self.current_arm_keypoints is None or len(self.current_arm_keypoints) != 8:
            rospy.loginfo("Lost arm keypoints, returning to TRACKING")
            self.return_to_tracking()
            return

        # Extract head keypoints for face-to-hand pointing vector
        head_data = {
            'nose': (self.current_arm_keypoints[0].position.x,
                    self.current_arm_keypoints[0].position.y,
                    self.current_arm_keypoints[0].position.z),
            'neck': (self.current_arm_keypoints[1].position.x,
                    self.current_arm_keypoints[1].position.y,
                    self.current_arm_keypoints[1].position.z)
        }

        # Check which arm is pointing (indices offset by 2 for nose/neck)
        left_arm = {
            'shoulder': (self.current_arm_keypoints[2].position.x,
                        self.current_arm_keypoints[2].position.y,
                        self.current_arm_keypoints[2].position.z),
            'elbow': (self.current_arm_keypoints[3].position.x,
                     self.current_arm_keypoints[3].position.y,
                     self.current_arm_keypoints[3].position.z),
            'wrist': (self.current_arm_keypoints[4].position.x,
                     self.current_arm_keypoints[4].position.y,
                     self.current_arm_keypoints[4].position.z)
        }
        right_arm = {
            'shoulder': (self.current_arm_keypoints[5].position.x,
                        self.current_arm_keypoints[5].position.y,
                        self.current_arm_keypoints[5].position.z),
            'elbow': (self.current_arm_keypoints[6].position.x,
                     self.current_arm_keypoints[6].position.y,
                     self.current_arm_keypoints[6].position.z),
            'wrist': (self.current_arm_keypoints[7].position.x,
                     self.current_arm_keypoints[7].position.y,
                     self.current_arm_keypoints[7].position.z)
        }

        # Check both arms for pointing
        left_pointing, left_dir = self.is_arm_pointing(left_arm, "Left")
        right_pointing, right_dir = self.is_arm_pointing(right_arm, "Right")

        # Validate the active arm (set during transition from TRACKING) is still pointing
        pointing_detected = False
        current_direction = None
        active_arm_data = None

        if self.active_arm == 'left' and left_pointing:
            pointing_detected = True
            current_direction = left_dir
            active_arm_data = left_arm
        elif self.active_arm == 'right' and right_pointing:
            pointing_detected = True
            current_direction = right_dir
            active_arm_data = right_arm

        # If active arm stopped pointing, check if we should switch to the other arm
        # Always prefer left arm when switching
        if not pointing_detected:
            if left_pointing:
                self.active_arm = 'left'
                pointing_detected = True
                current_direction = left_dir
                active_arm_data = left_arm
            elif right_pointing:
                self.active_arm = 'right'
                pointing_detected = True
                current_direction = right_dir
                active_arm_data = right_arm

        # Publish active arm for ground truth marker
        if pointing_detected:
            self.active_arm_pub.publish(String(data=self.active_arm))

        self.pointing_detected_pub.publish(Bool(pointing_detected))

        if not pointing_detected:
            rospy.loginfo("Pointing ended, returning to TRACKING")
            self.reset_pointing_state()
            self.return_to_tracking()
            return

        # Check direction stability using sliding window
        self.direction_history.append(current_direction)
        if len(self.direction_history) > self.direction_history_size:
            self.direction_history.pop(0)

        # Only check stability once buffer has enough samples
        if len(self.direction_history) >= self.direction_history_size:
            # Calculate mean direction from buffer
            dirs = np.array(self.direction_history)
            mean_dir = np.mean(dirs, axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm > 0:
                mean_dir = mean_dir / mean_norm

                # Check if current direction deviates significantly from mean
                dot = np.clip(np.dot(current_direction, mean_dir), -1.0, 1.0)
                angle_from_mean = math.degrees(math.acos(dot))

                if angle_from_mean > self.direction_stability_threshold:
                    rospy.loginfo_throttle(1.0, f"Direction unstable ({angle_from_mean:.1f}° from mean), resetting timer")
                    self.pointing_start_time = current_time
                    self.direction_history = []  # Clear buffer on reset

        self.last_pointing_direction = current_direction

        # Check hold duration
        elapsed = (current_time - self.pointing_start_time).to_sec()

        if elapsed >= self.hold_duration and not self.pointing_confirmed:
            self.pointing_confirmed = True
            rospy.loginfo("=" * 50)
            rospy.loginfo(f"POINTING CONFIRMED after {elapsed:.1f}s hold!")
            rospy.loginfo(f"Active arm: {self.active_arm.upper()}")
            rospy.loginfo("=" * 50)

            # Calculate 3D pointing and ground intersection using face-to-hand vector
            ground_point = self.calculate_3d_pointing(active_arm_data, head_data, current_time)
            if ground_point is not None:
                self.transition_to_navigating(ground_point)
            else:
                rospy.logwarn("Could not calculate ground point, returning to TRACKING")
                self.return_to_tracking()
            return

        self.pointing_hold_pub.publish(Bool(self.pointing_confirmed))

        if not self.pointing_confirmed:
            rospy.loginfo_throttle(0.5, f"Holding point... {elapsed:.1f}s / {self.hold_duration}s")

    # ========== 3D pointing calculation ==========

    def get_depth_at_pixel(self, x, y):
        """Get depth value at pixel coordinates."""
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]
        x_int, y_int = int(round(x)), int(round(y))

        if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
            return None

        depth_val = self.depth_image[y_int, x_int]

        if depth_val <= 0 or np.isnan(depth_val) or np.isinf(depth_val):
            # Try averaging nearby pixels
            radius = 3
            x_min, x_max = max(0, x_int - radius), min(w, x_int + radius + 1)
            y_min, y_max = max(0, y_int - radius), min(h, y_int + radius + 1)
            region = self.depth_image[y_min:y_max, x_min:x_max]
            valid = region[(region > 0) & (~np.isnan(region)) & (~np.isinf(region))]
            if len(valid) == 0:
                return None
            depth_val = np.median(valid)

        if depth_val <= 0:
            return None

        return depth_val / 1000.0 if depth_val > 100 else depth_val

    def get_depth_at_region(self, cx, cy, region_radius=10, max_depth=None):
        """Get robust depth estimate for a keypoint region.

        Uses foreground-filtered mean depth within a circular region to
        reject background pixels that "bleed through" around small targets
        like wrists.

        Args:
            cx, cy: Center pixel coordinates
            region_radius: Radius of circular region to sample (default 10 pixels)
            max_depth: Maximum allowed depth in meters (for filtering background)

        Returns:
            Depth value in meters, or None if no valid depth found
        """
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]
        x_int, y_int = int(round(cx)), int(round(cy))

        # Bounds check
        if x_int < 0 or x_int >= w or y_int < 0 or y_int >= h:
            return None

        # Sample circular region around keypoint
        x_min = max(0, x_int - region_radius)
        x_max = min(w, x_int + region_radius + 1)
        y_min = max(0, y_int - region_radius)
        y_max = min(h, y_int + region_radius + 1)

        region = self.depth_image[y_min:y_max, x_min:x_max]

        # Create circular mask
        yy, xx = np.ogrid[:region.shape[0], :region.shape[1]]
        center_y, center_x = region.shape[0] // 2, region.shape[1] // 2
        mask = ((xx - center_x)**2 + (yy - center_y)**2) <= region_radius**2

        # Filter valid depth values within the circular mask
        valid = region[mask]
        valid = valid[(valid > 0) & (~np.isnan(valid)) & (~np.isinf(valid))]

        if len(valid) == 0:
            # Fallback to simple point sampling
            return self.get_depth_at_pixel(cx, cy)

        # Convert to meters for filtering
        valid_m = np.where(valid > 100, valid / 1000.0, valid)

        # Foreground filtering: reject background pixels
        # Use minimum depth + tolerance to find foreground cluster
        min_depth = np.min(valid_m)
        foreground_tolerance = 0.5  # 50cm tolerance for foreground cluster
        foreground_mask = valid_m <= (min_depth + foreground_tolerance)

        # Also apply max_depth constraint if provided
        if max_depth is not None:
            foreground_mask &= (valid_m <= max_depth)

        valid_foreground = valid_m[foreground_mask]

        if len(valid_foreground) == 0:
            # If no foreground pixels, use minimum (closest point)
            depth_val = min_depth
        else:
            # Use mean of foreground pixels
            depth_val = np.mean(valid_foreground)

        if depth_val <= 0:
            return None

        return depth_val

    def calculate_3d_pointing(self, arm_data, head_data, stamp):
        """Calculate 3D pointing using face-to-hand vector (Azari et al. paper method).

        The pointing vector originates at the person's face (nose or neck) and
        passes through the wrist. This matches human pointing intention better
        than using the forearm direction alone.

        Args:
            arm_data: Dict with 'wrist' (x, y, confidence) tuple
            head_data: Dict with 'nose' and 'neck' (x, y, confidence) tuples
            stamp: Current timestamp

        Returns:
            Ground intersection point as numpy array [x, y, z] in base_link frame,
            or None if calculation fails.
        """
        wrist = arm_data['wrist']
        nose = head_data['nose']
        neck = head_data['neck']

        # Select face keypoint: prefer nose, fallback to neck
        if nose[2] >= self.pointing_min_confidence:
            face_pixel = nose
            rospy.loginfo_throttle(2.0, "Using NOSE for pointing origin")
        elif neck[2] >= self.pointing_min_confidence:
            face_pixel = neck
            rospy.loginfo_throttle(2.0, "Using NECK for pointing origin (nose unavailable)")
        else:
            rospy.logwarn("No valid head keypoint for pointing origin")
            return None

        # Get face depth first (larger, more reliable target)
        face_depth = self.get_depth_at_region(face_pixel[0], face_pixel[1])
        if face_depth is None:
            rospy.logwarn("Could not get depth for face")
            return None

        # Get wrist depth with arm-length constraint
        # Human arm length is ~0.7-0.8m, so wrist cannot be more than ~1m further than face
        # This prevents background pixels from being mistaken for the wrist
        max_wrist_depth = face_depth + 1.0  # Allow up to 1m beyond face
        wrist_depth = self.get_depth_at_region(wrist[0], wrist[1], max_depth=max_wrist_depth)

        if wrist_depth is None:
            rospy.logwarn("Could not get depth for wrist")
            return None

        # Sanity check: log if depths seem problematic
        depth_diff = abs(wrist_depth - face_depth)
        if depth_diff > 0.8:
            rospy.logwarn_throttle(1.0, f"Large face-wrist depth difference: {depth_diff:.2f}m")

        # Project face to 3D (camera frame)
        face_3d = np.array([
            (face_pixel[0] - self.cx) * face_depth / self.fx,
            (face_pixel[1] - self.cy) * face_depth / self.fy,
            face_depth
        ])

        # Project wrist to 3D (camera frame)
        wrist_3d = np.array([
            (wrist[0] - self.cx) * wrist_depth / self.fx,
            (wrist[1] - self.cy) * wrist_depth / self.fy,
            wrist_depth
        ])

        # Publish wrist 3D for debugging
        wrist_msg = PointStamped()
        wrist_msg.header.stamp = rospy.Time.now()
        wrist_msg.header.frame_id = "head_rgbd_sensor_rgb_frame"
        wrist_msg.point.x, wrist_msg.point.y, wrist_msg.point.z = wrist_3d
        self.pointing_wrist_3d_pub.publish(wrist_msg)

        # Pointing vector: from face through wrist (paper equation 2: P = hand - face)
        pointing_dir = wrist_3d - face_3d
        dir_len = np.linalg.norm(pointing_dir)
        if dir_len < 0.01:
            rospy.logwarn("Face and wrist too close for valid pointing vector")
            return None
        pointing_dir = pointing_dir / dir_len

        # Publish pointing vector for debugging
        vector_msg = Vector3Stamped()
        vector_msg.header.stamp = rospy.Time.now()
        vector_msg.header.frame_id = "head_rgbd_sensor_rgb_frame"
        vector_msg.vector.x, vector_msg.vector.y, vector_msg.vector.z = pointing_dir
        self.pointing_vector_pub.publish(vector_msg)

        rospy.loginfo_throttle(1.0,
            f"Face-to-hand: face=({face_3d[0]:.2f},{face_3d[1]:.2f},{face_3d[2]:.2f}) "
            f"wrist=({wrist_3d[0]:.2f},{wrist_3d[1]:.2f},{wrist_3d[2]:.2f})")

        # Calculate ground intersection using face as origin (paper equations 1-4)
        return self.calculate_ground_intersection(face_3d, pointing_dir)

    def calculate_ground_intersection(self, origin_camera, direction_camera):
        """Calculate intersection with ground plane (z=0 in base_link)."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'head_rgbd_sensor_rgb_frame',
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            # Transform origin
            origin_msg = PointStamped()
            origin_msg.header.stamp = rospy.Time.now()
            origin_msg.header.frame_id = "head_rgbd_sensor_rgb_frame"
            origin_msg.point.x, origin_msg.point.y, origin_msg.point.z = origin_camera
            origin_base = tf2_geometry_msgs.do_transform_point(origin_msg, transform)

            # Transform direction (via second point)
            end_camera = origin_camera + direction_camera
            end_msg = PointStamped()
            end_msg.header.stamp = rospy.Time.now()
            end_msg.header.frame_id = "head_rgbd_sensor_rgb_frame"
            end_msg.point.x, end_msg.point.y, end_msg.point.z = end_camera
            end_base = tf2_geometry_msgs.do_transform_point(end_msg, transform)

            origin_np = np.array([origin_base.point.x, origin_base.point.y, origin_base.point.z])
            end_np = np.array([end_base.point.x, end_base.point.y, end_base.point.z])
            direction_base = end_np - origin_np
            direction_base = direction_base / np.linalg.norm(direction_base)

            # Find z=0 intersection
            if abs(direction_base[2]) < 0.01:
                rospy.logwarn("Pointing parallel to ground")
                return None

            t = -origin_np[2] / direction_base[2]
            if t < 0:
                rospy.logwarn("Ground intersection behind origin")
                return None

            ground_point = origin_np + t * direction_base

            # Publish ground point
            ground_msg = PointStamped()
            ground_msg.header.stamp = rospy.Time.now()
            ground_msg.header.frame_id = "base_link"
            ground_msg.point.x, ground_msg.point.y = ground_point[0], ground_point[1]
            ground_msg.point.z = 0.0
            self.pointing_ground_pub.publish(ground_msg)

            rospy.loginfo(f"GROUND TARGET: ({ground_point[0]:.2f}, {ground_point[1]:.2f}) m from robot")
            return ground_point

        except tf2_ros.TransformException as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None

    # ========== NAVIGATING state logic ==========

    def run_navigating(self, current_time):
        """Execute NAVIGATING state - move to target using time-based dead reckoning.

        Phase 0: Rotate for calculated time to face target
        Phase 1: Drive forward for calculated time to reach target
        """
        if self.target_ground_point is None:
            self.transition_to_searching()
            return

        # Check for timeout
        elapsed = (current_time - self.nav_start_time).to_sec()
        if elapsed > self.nav_timeout:
            rospy.logwarn(f"Navigation timeout after {elapsed:.1f}s")
            self.stop_base()
            self.transition_to_completed()
            return

        # Navigation parameters
        linear_speed = 0.2  # m/s
        angular_speed = 0.4  # rad/s

        # Phase 0: Rotate to face target
        if self.nav_phase == 0:
            # Calculate how long to rotate: time = angle / angular_speed
            rotate_time = abs(self.target_angle) / angular_speed
            rotate_elapsed = (current_time - self.nav_rotate_start_time).to_sec()

            if rotate_elapsed >= rotate_time:
                # Done rotating, transition to driving phase
                rospy.loginfo(f"Rotation complete ({math.degrees(self.target_angle):.1f}°), starting forward motion")
                self.stop_base()
                self.nav_phase = 1
                self.nav_drive_start_time = current_time
                return

            # Rotate in the correct direction
            cmd = Twist()
            cmd.angular.z = angular_speed if self.target_angle > 0 else -angular_speed
            self.cmd_vel_pub.publish(cmd)

            remaining_angle = abs(self.target_angle) - (rotate_elapsed * angular_speed)
            rospy.loginfo_throttle(0.5, f"Rotating... {math.degrees(remaining_angle):.1f}° remaining")
            return

        # Phase 1: Drive forward
        drive_time = self.initial_target_distance / linear_speed
        drive_elapsed = (current_time - self.nav_drive_start_time).to_sec()

        if drive_elapsed >= drive_time:
            # Navigation complete
            rospy.loginfo("=" * 50)
            rospy.loginfo(f"Navigation complete! Traveled ~{self.initial_target_distance:.2f}m")
            rospy.loginfo("=" * 50)
            self.stop_base()
            self.transition_to_completed()
            return

        # Drive forward
        cmd = Twist()
        cmd.linear.x = linear_speed
        self.cmd_vel_pub.publish(cmd)

        remaining = self.initial_target_distance - (drive_elapsed * linear_speed)
        rospy.loginfo_throttle(0.5, f"Driving... ~{remaining:.2f}m remaining")

    # ========== Main loop ==========

    def publish_state(self):
        """Publish current state for monitoring."""
        self.state_pub.publish(String(data=self.state.name))

    def publish_markers(self):
        """Publish visualization markers for RViz."""
        markers = MarkerArray()
        now = rospy.Time.now()

        # 1. Human cylinder marker (blue) - at human model position
        if self.human_position_msg is not None:
            human_marker = Marker()
            human_marker.header.frame_id = "odom"  # Use odom frame for RViz
            human_marker.header.stamp = now
            human_marker.ns = "human_point_follower"
            human_marker.id = 0
            human_marker.type = Marker.CYLINDER
            human_marker.action = Marker.ADD
            human_marker.pose.position.x = self.human_position_msg.point.x
            human_marker.pose.position.y = self.human_position_msg.point.y
            human_marker.pose.position.z = 0.9  # Half height (1.8m human)
            human_marker.pose.orientation.w = 1.0
            human_marker.scale.x = 0.5  # Diameter
            human_marker.scale.y = 0.5
            human_marker.scale.z = 1.8  # Height
            human_marker.color.r = 0.0
            human_marker.color.g = 0.0
            human_marker.color.b = 1.0
            human_marker.color.a = 0.7
            human_marker.lifetime = rospy.Duration(0)  # Persistent
            markers.markers.append(human_marker)

        # 2. Ground truth marker (green) - where arm vector intersects ground
        if self.ground_truth_point_msg is not None:
            gt_marker = Marker()
            gt_marker.header.frame_id = "odom"  # Use odom frame for RViz
            gt_marker.header.stamp = now
            gt_marker.ns = "human_point_follower"
            gt_marker.id = 1
            gt_marker.type = Marker.SPHERE
            gt_marker.action = Marker.ADD
            gt_marker.pose.position.x = self.ground_truth_point_msg.point.x
            gt_marker.pose.position.y = self.ground_truth_point_msg.point.y
            gt_marker.pose.position.z = 0.05
            gt_marker.pose.orientation.w = 1.0
            gt_marker.scale.x = 0.2
            gt_marker.scale.y = 0.2
            gt_marker.scale.z = 0.2
            gt_marker.color.r = 0.0
            gt_marker.color.g = 1.0
            gt_marker.color.b = 0.0
            gt_marker.color.a = 1.0
            gt_marker.lifetime = rospy.Duration(0)
            markers.markers.append(gt_marker)

        # 3. Robot computed marker (red) - where robot thinks the point is
        # Use odom-frame coordinates so marker stays fixed as robot moves
        if self.target_ground_point_odom is not None:
            robot_marker = Marker()
            robot_marker.header.frame_id = "odom"  # Use odom frame so it stays fixed
            robot_marker.header.stamp = now
            robot_marker.ns = "human_point_follower"
            robot_marker.id = 2
            robot_marker.type = Marker.SPHERE
            robot_marker.action = Marker.ADD
            robot_marker.pose.position.x = self.target_ground_point_odom[0]
            robot_marker.pose.position.y = self.target_ground_point_odom[1]
            robot_marker.pose.position.z = 0.05
            robot_marker.pose.orientation.w = 1.0
            robot_marker.scale.x = 0.2
            robot_marker.scale.y = 0.2
            robot_marker.scale.z = 0.2
            robot_marker.color.r = 1.0
            robot_marker.color.g = 0.0
            robot_marker.color.b = 0.0
            robot_marker.color.a = 1.0
            robot_marker.lifetime = rospy.Duration(0)
            markers.markers.append(robot_marker)

        if markers.markers:
            self.marker_pub.publish(markers)

    def clear_markers(self):
        """Clear all visualization markers for demo reset."""
        markers = MarkerArray()
        for i in range(3):
            marker = Marker()
            marker.header.frame_id = "odom"  # Use odom frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "human_point_follower"
            marker.id = i
            marker.action = Marker.DELETE
            markers.markers.append(marker)
        self.marker_pub.publish(markers)
        # Reset cached visualization points
        self.ground_truth_point_msg = None
        self.target_ground_point_odom = None

    def run(self):
        """Main control loop."""
        rate = rospy.Rate(self.control_rate)

        rospy.sleep(2.0)  # Wait for connections

        # Start in IDLE state - waiting for start signal
        self.transition_to_idle()

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()

            # Publish current state
            self.publish_state()

            if self.state == State.IDLE:
                # Just wait for start signal (handled by callback)
                rospy.loginfo_throttle(5.0, "IDLE - Waiting for start signal on /human_point_follower/start")

            elif self.state == State.SEARCHING:
                self.run_searching(current_time)

            elif self.state == State.TRACKING:
                self.run_tracking(current_time)

            elif self.state == State.DETECTED:
                self.run_detected(current_time)

            elif self.state == State.NAVIGATING:
                self.run_navigating(current_time)

            elif self.state == State.COMPLETED:
                # Just wait for reset signal (handled by callback)
                rospy.loginfo_throttle(5.0, "COMPLETED - Demo finished. Publish to /human_point_follower/start to reset")

            # Publish visualization markers
            self.publish_markers()

            rate.sleep()

        # Cleanup
        self.stop_base()
        rospy.loginfo("Human Point Follower shutting down")


if __name__ == '__main__':
    try:
        follower = HumanPointFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
