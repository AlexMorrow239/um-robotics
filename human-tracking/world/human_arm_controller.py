#!/usr/bin/env python3
"""
Human Arm Controller

Controls the arm poses of the human character in Isaac Sim for pointing gestures.
Subscribes to ROS topics to receive pose commands.

Workaround approach: Hides character's arms/skin and spawns capsule geometry
for pointing poses, since skeletal animation doesn't work in standalone apps.
"""

import numpy as np
import math
import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped, PointStamped
from queue import Queue, Empty

# Isaac Sim imports (must be called after SimulationApp is started)
from pxr import Usd, UsdSkel, UsdGeom, Gf, Vt, Sdf
import omni.usd


class HumanArmController:
    """
    Controller for human character arm poses in Isaac Sim.

    Uses spawned capsule geometry for arms since skeletal animation
    doesn't work in standalone Python apps.
    """

    # Arm geometry specs
    ARM_LENGTH = 0.50  # Combined upper arm + forearm (cylindrical part)
    ARM_RADIUS = 0.04  # Thickness
    SHOULDER_HEIGHT_OFFSET = 1.40  # Height above character origin
    SHOULDER_SIDE_OFFSET = 0.20  # Left/right offset from center

    # Skin-like color (RGB)
    SKIN_COLOR = (0.87, 0.72, 0.53)

    # Highly pronounced pointing directions for testing
    # Character faces -Y in world frame. Left arm natural side is +X, right arm is -X.

    # Right arm pointing strongly lateral-right with moderate downward angle
    # Highly visible from robot's perspective (appears on robot's left)
    RIGHT_LATERAL_DIRECTION = (-0.85, -0.35, -0.40)  # ~70° right, ~35° down

    # Left arm pointing forward at distant ground (shallow angle)
    # Clearly forward-facing pose
    LEFT_FORWARD_DIRECTION = (0.25, -0.90, -0.35)  # ~15° left, ~20° down

    # Left arm pointing strongly lateral-left with moderate downward angle
    # Highly visible from robot's perspective (appears on robot's right)
    LEFT_LATERAL_DIRECTION = (0.85, -0.35, -0.40)  # ~70° left, ~35° down

    # Resting arm directions - arms hang naturally at sides, slightly forward
    # Used for the non-pointing arm to help robot track the human model
    LEFT_RESTING_DIRECTION = (0.15, -0.15, -0.97)   # Slight outward/forward, mostly down
    RIGHT_RESTING_DIRECTION = (-0.15, -0.15, -0.97)  # Slight outward/forward, mostly down

    # Arm segment lengths (realistic human proportions)
    UPPER_ARM_LENGTH = 0.28   # Shoulder to elbow
    FOREARM_LENGTH = 0.25     # Elbow to wrist

    # Segment radii (tapering for natural appearance)
    UPPER_ARM_RADIUS = 0.045
    FOREARM_RADIUS = 0.035
    WRIST_RADIUS = 0.045      # Sphere radius for wrist joint (larger for better detection)
    HAND_RADIUS = 0.04        # Sphere radius for hand

    # Elbow bend angle (degrees) for natural appearance
    ELBOW_BEND_ANGLE = 12.0

    # Head geometry specs
    HEAD_RADIUS = 0.11  # Approximate head radius (sphere)
    HEAD_HEIGHT_OFFSET = 1.45  # Height of head center above character origin

    # Skeleton joint paths (relative to skeleton prim)
    LEFT_SHOULDER_JOINT = "RL_BoneRoot/Hip/Waist/Spine01/Spine02/L_Clavicle/L_Upperarm"
    RIGHT_SHOULDER_JOINT = "RL_BoneRoot/Hip/Waist/Spine01/Spine02/R_Clavicle/R_Upperarm"

    POSES = {
        "neutral": {
            "description": "Both arms hidden (no pointing)",
        },
        "right_point": {
            "description": "Right arm pointing strongly to human's right side (lateral)",
        },
        "left_point_far": {
            "description": "Left arm pointing forward at distant ground",
        },
        "left_point_lateral": {
            "description": "Left arm pointing strongly to human's left side (lateral)",
        },
    }

    def __init__(self, character_prim_path="/World/Characters/F_Business_02"):
        self.character_prim_path = character_prim_path
        self.stage = None
        self.current_pose = "neutral"
        self.initialized = False
        self.arms_hidden = False

        # Paths for spawned arm geometry - as children of character so they move with it
        # Each arm has four segments: upper arm, forearm, wrist, and hand
        self.left_arm_paths = {
            'upper': f"{character_prim_path}/SpawnedArms/Left/UpperArm",
            'forearm': f"{character_prim_path}/SpawnedArms/Left/Forearm",
            'wrist': f"{character_prim_path}/SpawnedArms/Left/Wrist",
            'hand': f"{character_prim_path}/SpawnedArms/Left/Hand"
        }
        self.right_arm_paths = {
            'upper': f"{character_prim_path}/SpawnedArms/Right/UpperArm",
            'forearm': f"{character_prim_path}/SpawnedArms/Right/Forearm",
            'wrist': f"{character_prim_path}/SpawnedArms/Right/Wrist",
            'hand': f"{character_prim_path}/SpawnedArms/Right/Hand"
        }

        # Path for spawned head geometry
        self.head_path = f"{character_prim_path}/SpawnedArms/Head"
        self.head_spawned = False

        # ROS subscribers
        self.pose_sub = None
        self.pose_index_sub = None

        # Queue for thread-safe pose changes (ROS callbacks run on different thread)
        self.pose_queue = Queue()

        # ROS publisher for left arm pose
        self.left_arm_pose_pub = None

        # ROS publisher for ground truth intersection point
        self.ground_truth_pub = None

        # Store current arm directions for publishing
        self.current_left_arm_direction = None
        self.current_right_arm_direction = None

        # Track which arm is actively pointing (set by human_point_follower)
        self.active_pointing_arm = None  # 'left', 'right', or None

        # World-to-odom transform parameters
        # Robot spawns at world [0, 1] with 90° yaw (facing +Y in world)
        # In odom frame, robot starts at [0, 0] facing +X (ROS convention)
        self.robot_initial_world = np.array([0.0, 1.0, 0.0])
        self.robot_initial_yaw = np.radians(90.0)  # 90 degrees

        # Cached skeleton prim path (found during initialization)
        self.skeleton_prim_path = None

    def initialize(self):
        """Initialize the controller after simulation has started."""
        try:
            self.stage = omni.usd.get_context().get_stage()
            if self.stage is None:
                print("[HumanArmController] ERROR: Stage not available")
                return False

            character_prim = self.stage.GetPrimAtPath(self.character_prim_path)
            if not character_prim or not character_prim.IsValid():
                print(f"[HumanArmController] ERROR: Character not found at {self.character_prim_path}")
                return False

            # Find the skeleton prim within the character
            self.skeleton_prim_path = self._find_skeleton_prim(character_prim)
            if self.skeleton_prim_path:
                print(f"[HumanArmController] Found skeleton: {self.skeleton_prim_path}")
            else:
                print("[HumanArmController] WARNING: Skeleton not found, using fallback offsets")

            self.initialized = True
            print(f"[HumanArmController] Initialized successfully (spawn-arms mode)")
            print(f"[HumanArmController] Character: {self.character_prim_path}")

            return True

        except Exception as e:
            print(f"[HumanArmController] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_skeleton_prim(self, character_prim):
        """Find the UsdSkel.Skeleton prim within the character hierarchy."""
        for prim in Usd.PrimRange(character_prim):
            if prim.IsA(UsdSkel.Skeleton):
                return str(prim.GetPath())
        return None

    def hide_arms(self):
        """Hide the character's original skin/arms and spawn a replacement head.

        Since the skin mesh combines head and arms in one layer, we hide the entire
        skin mesh and spawn a sphere for the head. Arms are managed separately.
        Keeps visible: vest, pants, shoes.
        """
        if not self.stage:
            return False

        character_prim = self.stage.GetPrimAtPath(self.character_prim_path)
        if not character_prim:
            return False

        # Hide skin (contains both head and arms) and shirt (covers arms)
        hide_keywords = ['skin', 'shirt']

        hidden_count = 0
        for prim in Usd.PrimRange(character_prim):
            if prim.IsA(UsdGeom.Mesh):
                prim_name = prim.GetName().lower()
                if any(kw in prim_name for kw in hide_keywords):
                    imageable = UsdGeom.Imageable(prim)
                    imageable.MakeInvisible()
                    print(f"[HumanArmController] Hidden mesh: {prim.GetPath()}")
                    hidden_count += 1

        self.arms_hidden = True
        print(f"[HumanArmController] Hidden {hidden_count} meshes (skin hidden)")

        # Spawn replacement head geometry
        self.spawn_head()

        return True

    def _get_character_world_position(self):
        """Get the character's world position from its prim transform."""
        character_prim = self.stage.GetPrimAtPath(self.character_prim_path)
        if not character_prim or not character_prim.IsValid():
            return Gf.Vec3d(0, 0, 0)

        xformable = UsdGeom.Xformable(character_prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        return Gf.Vec3d(translation[0], translation[1], translation[2])

    def _get_joint_local_position(self, joint_path):
        """Get position of a skeleton joint relative to the character prim.

        Args:
            joint_path: Joint path relative to skeleton (e.g., "RL_BoneRoot/Hip/...")

        Returns:
            Gf.Vec3d with position relative to character, or None if joint not found
        """
        if not self.skeleton_prim_path or not self.stage:
            return None

        skeleton_prim = self.stage.GetPrimAtPath(self.skeleton_prim_path)
        if not skeleton_prim or not skeleton_prim.IsValid():
            return None

        skeleton = UsdSkel.Skeleton(skeleton_prim)

        # Get the joints attribute to find the index of our target joint
        joints_attr = skeleton.GetJointsAttr()
        if not joints_attr or not joints_attr.HasValue():
            return None

        joints = joints_attr.Get()
        try:
            joint_index = list(joints).index(joint_path)
        except ValueError:
            print(f"[HumanArmController] Joint not found: {joint_path}")
            return None

        # Create a skeleton query to compute joint transforms
        skel_cache = UsdSkel.Cache()
        skel_query = skel_cache.GetSkelQuery(skeleton)

        # ComputeJointWorldTransforms requires an XformCache
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        # Compute joint world transforms
        joint_world_transforms = skel_query.ComputeJointWorldTransforms(xform_cache)
        if not joint_world_transforms or joint_index >= len(joint_world_transforms):
            print(f"[HumanArmController] DEBUG: ComputeJointWorldTransforms returned empty or invalid")
            return None

        # Extract translation from the joint's world transform
        joint_transform = joint_world_transforms[joint_index]
        joint_world_pos = joint_transform.ExtractTranslation()

        # Convert to local coordinates (relative to character prim)
        char_pos = self._get_character_world_position()
        local_pos = Gf.Vec3d(
            joint_world_pos[0] - char_pos[0],
            joint_world_pos[1] - char_pos[1],
            joint_world_pos[2] - char_pos[2]
        )

        return local_pos

    def _get_shoulder_position(self, side="left"):
        """Get shoulder position in local coordinates (relative to character prim)."""
        # Try to get actual joint position from skeleton
        joint_path = self.LEFT_SHOULDER_JOINT if side == "left" else self.RIGHT_SHOULDER_JOINT
        joint_pos = self._get_joint_local_position(joint_path)

        if joint_pos is not None:
            return joint_pos

        # Fallback to hardcoded offsets if skeleton lookup fails
        print(f"[HumanArmController] WARNING: Using fallback offset for {side} shoulder")
        x_offset = -self.SHOULDER_SIDE_OFFSET if side == "left" else self.SHOULDER_SIDE_OFFSET
        return Gf.Vec3d(
            x_offset,
            0,
            self.SHOULDER_HEIGHT_OFFSET
        )

    def _direction_to_quaternion(self, direction, side="left"):
        """Convert a direction vector to a quaternion for capsule orientation.

        Capsule default axis is Y, so we need to rotate to align with direction.
        """
        # Normalize direction
        dir_np = np.array([direction[0], direction[1], direction[2]])
        dir_np = dir_np / np.linalg.norm(dir_np)

        # Default capsule axis is Y (0, 1, 0)
        default_axis = np.array([0, 1, 0])

        # Calculate rotation axis and angle
        cross = np.cross(default_axis, dir_np)
        dot = np.dot(default_axis, dir_np)

        if np.linalg.norm(cross) < 1e-6:
            # Vectors are parallel
            if dot > 0:
                return Gf.Quatf(1, 0, 0, 0)  # Identity
            else:
                return Gf.Quatf(0, 1, 0, 0)  # 180 degree rotation around X

        cross = cross / np.linalg.norm(cross)
        angle = math.acos(np.clip(dot, -1, 1))

        # Quaternion from axis-angle
        half_angle = angle / 2
        w = math.cos(half_angle)
        x = cross[0] * math.sin(half_angle)
        y = cross[1] * math.sin(half_angle)
        z = cross[2] * math.sin(half_angle)

        return Gf.Quatf(w, x, y, z)

    def _spawn_capsule_segment(self, path, start_pos, end_pos, radius):
        """Spawn a capsule segment between two positions.

        Args:
            path: USD prim path for the capsule
            start_pos: Starting position (Gf.Vec3d or tuple)
            end_pos: Ending position (Gf.Vec3d or tuple)
            radius: Capsule radius

        Returns:
            The created capsule prim, or None on failure
        """
        if not self.stage:
            return None

        # Convert to numpy for calculations
        start = np.array([start_pos[0], start_pos[1], start_pos[2]])
        end = np.array([end_pos[0], end_pos[1], end_pos[2]])

        # Calculate direction and length
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None

        direction = direction / length  # Normalize

        # Capsule height is the cylindrical part (excluding hemispherical caps)
        # Total length = height + 2*radius, so height = length - 2*radius
        capsule_height = max(0.01, length - 2 * radius)

        # Center position is midpoint between start and end
        center = (start + end) / 2
        center_gf = Gf.Vec3d(center[0], center[1], center[2])

        # Remove existing prim if present
        existing = self.stage.GetPrimAtPath(path)
        if existing and existing.IsValid():
            self.stage.RemovePrim(path)

        # Create capsule
        capsule = UsdGeom.Capsule.Define(self.stage, path)
        capsule.CreateHeightAttr(capsule_height)
        capsule.CreateRadiusAttr(radius)
        capsule.CreateAxisAttr(UsdGeom.Tokens.y)
        capsule.CreateDisplayColorAttr([self.SKIN_COLOR])

        # Get orientation quaternion
        quat = self._direction_to_quaternion(tuple(direction))

        # Apply transforms
        xformable = UsdGeom.Xformable(capsule.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(center_gf)
        xformable.AddOrientOp().Set(Gf.Quatf(quat.GetReal(), quat.GetImaginary()[0],
                                              quat.GetImaginary()[1], quat.GetImaginary()[2]))

        return capsule.GetPrim()

    def _spawn_hand_sphere(self, path, position, radius):
        """Spawn a sphere for the hand at the given position.

        Args:
            path: USD prim path for the sphere
            position: Center position (Gf.Vec3d or tuple)
            radius: Sphere radius

        Returns:
            The created sphere prim, or None on failure
        """
        if not self.stage:
            return None

        # Remove existing prim if present
        existing = self.stage.GetPrimAtPath(path)
        if existing and existing.IsValid():
            self.stage.RemovePrim(path)

        # Create sphere
        sphere = UsdGeom.Sphere.Define(self.stage, path)
        sphere.CreateRadiusAttr(radius)
        sphere.CreateDisplayColorAttr([self.SKIN_COLOR])

        # Apply position
        pos_gf = Gf.Vec3d(position[0], position[1], position[2])
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(pos_gf)

        return sphere.GetPrim()

    def spawn_head(self):
        """Spawn a sphere for the head at the correct position.

        Called after hiding the skin mesh to provide a visible head.
        """
        if not self.stage:
            return False

        # Remove existing head if present
        existing = self.stage.GetPrimAtPath(self.head_path)
        if existing and existing.IsValid():
            self.stage.RemovePrim(self.head_path)

        # Create parent container if needed
        arms_container = f"{self.character_prim_path}/SpawnedArms"
        if not self.stage.GetPrimAtPath(arms_container):
            UsdGeom.Xform.Define(self.stage, arms_container)

        # Create head sphere at local position (relative to character)
        sphere = UsdGeom.Sphere.Define(self.stage, self.head_path)
        sphere.CreateRadiusAttr(self.HEAD_RADIUS)
        sphere.CreateDisplayColorAttr([self.SKIN_COLOR])

        # Position head at correct height (local coordinates)
        head_pos = Gf.Vec3d(0, 0, self.HEAD_HEIGHT_OFFSET)
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(head_pos)

        self.head_spawned = True
        print(f"[HumanArmController] Spawned head at height {self.HEAD_HEIGHT_OFFSET}m")
        return True

    def remove_head(self):
        """Remove the spawned head geometry."""
        if not self.stage:
            return

        prim = self.stage.GetPrimAtPath(self.head_path)
        if prim and prim.IsValid():
            self.stage.RemovePrim(self.head_path)
            print(f"[HumanArmController] Removed head")

        self.head_spawned = False

    def _rotate_vector_around_axis(self, vector, axis, angle_degrees):
        """Rotate a vector around an axis by the given angle.

        Args:
            vector: The vector to rotate (numpy array)
            axis: The axis to rotate around (numpy array, will be normalized)
            angle_degrees: Rotation angle in degrees

        Returns:
            Rotated vector as numpy array
        """
        angle_rad = math.radians(angle_degrees)
        axis = axis / np.linalg.norm(axis)

        # Rodrigues' rotation formula
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotated = (vector * cos_a +
                   np.cross(axis, vector) * sin_a +
                   axis * np.dot(axis, vector) * (1 - cos_a))

        return rotated

    def _get_shoulder_world_position(self, side="left"):
        """Get shoulder position in world coordinates.

        Args:
            side: 'left' or 'right'

        Returns:
            Gf.Vec3d with shoulder position in world frame
        """
        shoulder_local = self._get_shoulder_position(side)
        char_pos = self._get_character_world_position()
        return Gf.Vec3d(
            char_pos[0] + shoulder_local[0],
            char_pos[1] + shoulder_local[1],
            char_pos[2] + shoulder_local[2]
        )

    def _publish_left_arm_pose(self):
        """Publish the current left arm pose as a PoseStamped message.

        The pose includes:
        - Position: Shoulder position in world coordinates (dynamically tracks character prim)
        - Orientation: Arm direction as quaternion

        Frame ID: 'world' since Isaac Sim coordinates are in world frame.
        """
        if not self.initialized or self.left_arm_pose_pub is None:
            return

        # Only publish if we have a valid arm direction
        if self.current_left_arm_direction is None:
            return

        # Get current shoulder position in world coords
        shoulder_world = self._get_shoulder_world_position("left")

        # Get quaternion for current arm direction
        quat = self._direction_to_quaternion(self.current_left_arm_direction, "left")

        # Create PoseStamped message
        pose_msg = PoseStamped()

        # Header
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"  # Isaac Sim world frame

        # Position (shoulder position in world coordinates)
        pose_msg.pose.position.x = shoulder_world[0]
        pose_msg.pose.position.y = shoulder_world[1]
        pose_msg.pose.position.z = shoulder_world[2]

        # Orientation (arm direction as quaternion)
        pose_msg.pose.orientation.w = quat.GetReal()
        pose_msg.pose.orientation.x = quat.GetImaginary()[0]
        pose_msg.pose.orientation.y = quat.GetImaginary()[1]
        pose_msg.pose.orientation.z = quat.GetImaginary()[2]

        # Publish
        self.left_arm_pose_pub.publish(pose_msg)
        # Note: Ground truth is now published separately in update() based on active_pointing_arm

    def _publish_ground_truth_intersection(self, shoulder_world, side="left"):
        """Publish where the arm vector intersects the ground (z=0) in odom frame.

        This is the 'ground truth' point - what the human is actually pointing at,
        computed directly from the arm geometry in Isaac Sim, then transformed
        from world frame to odom frame for RViz visualization.

        Args:
            shoulder_world: Shoulder position in world coordinates
            side: Which arm to use ('left' or 'right')
        """
        # Get arm direction based on side
        arm_direction = self.current_left_arm_direction if side == "left" else self.current_right_arm_direction

        if self.ground_truth_pub is None or arm_direction is None:
            return

        # Get arm direction as numpy array
        dir_np = np.array(arm_direction)
        shoulder_np = np.array([shoulder_world[0], shoulder_world[1], shoulder_world[2]])

        # Ray-plane intersection at z=0 (ground plane)
        if abs(dir_np[2]) < 0.01:
            # Ray is parallel to ground, no intersection
            return

        t = -shoulder_np[2] / dir_np[2]
        if t < 0:
            # Intersection is behind shoulder (pointing up)
            return

        # Calculate ground intersection point in world frame
        ground_pt_world = shoulder_np + t * dir_np

        # Transform from world to odom frame
        # Step 1: Translate (shift origin to robot's initial position)
        translated = ground_pt_world - self.robot_initial_world

        # Step 2: Rotate by -90° around Z axis (from world +Y facing to odom +X facing)
        cos_theta = np.cos(-self.robot_initial_yaw)  # cos(-90°) = 0
        sin_theta = np.sin(-self.robot_initial_yaw)  # sin(-90°) = -1

        odom_x = cos_theta * translated[0] - sin_theta * translated[1]
        odom_y = sin_theta * translated[0] + cos_theta * translated[1]

        # Create and publish message in odom frame
        gt_msg = PointStamped()
        gt_msg.header.stamp = rospy.Time.now()
        gt_msg.header.frame_id = "odom"  # Use odom frame for RViz
        gt_msg.point.x = odom_x
        gt_msg.point.y = odom_y
        gt_msg.point.z = 0.0  # On ground plane
        self.ground_truth_pub.publish(gt_msg)

    def spawn_pointing_arm(self, side="left", direction=None):
        """Spawn arm geometry (upper arm, forearm, hand) pointing in the given direction.

        Creates three segments with a slight elbow bend for natural appearance:
        - Upper arm: capsule from shoulder to elbow
        - Forearm: capsule from elbow to wrist
        - Hand: sphere at the end of the wrist

        Args:
            side: "left" or "right"
            direction: (x, y, z) direction vector. If None, uses side-appropriate default
        """
        if not self.stage:
            return False

        # Default direction: use side-appropriate direction with outward angle
        if direction is None:
            direction = self.LEFT_DEFAULT_DIRECTION if side == "left" else self.RIGHT_DEFAULT_DIRECTION

        arm_paths = self.left_arm_paths if side == "left" else self.right_arm_paths

        # Remove existing arm segments if present
        for segment_path in arm_paths.values():
            existing = self.stage.GetPrimAtPath(segment_path)
            if existing and existing.IsValid():
                self.stage.RemovePrim(segment_path)

        # Create parent Xform containers
        arms_container = f"{self.character_prim_path}/SpawnedArms"
        if not self.stage.GetPrimAtPath(arms_container):
            UsdGeom.Xform.Define(self.stage, arms_container)

        side_container = f"{arms_container}/{side.capitalize()}"
        if not self.stage.GetPrimAtPath(side_container):
            UsdGeom.Xform.Define(self.stage, side_container)

        # Get shoulder position
        shoulder_pos = self._get_shoulder_position(side)
        shoulder_np = np.array([shoulder_pos[0], shoulder_pos[1], shoulder_pos[2]])

        # Normalize pointing direction
        dir_np = np.array([direction[0], direction[1], direction[2]])
        dir_np = dir_np / np.linalg.norm(dir_np)

        # Store direction for pose publishing (track both arms)
        if side == "left":
            self.current_left_arm_direction = tuple(dir_np)
        else:
            self.current_right_arm_direction = tuple(dir_np)

        # Calculate elbow bend direction
        # The upper arm bends slightly inward/upward from the pointing direction
        # Use a perpendicular axis for rotation (cross with up vector, or fallback)
        up = np.array([0, 0, 1])
        if abs(np.dot(dir_np, up)) > 0.99:
            # Direction is nearly vertical, use forward as reference
            up = np.array([0, 1, 0])

        # Rotation axis is perpendicular to both direction and up
        bend_axis = np.cross(dir_np, up)
        bend_axis = bend_axis / np.linalg.norm(bend_axis)

        # Apply elbow bend to upper arm direction (bend toward body)
        # For left arm, bend right; for right arm, bend left
        bend_sign = 1 if side == "left" else -1
        upper_arm_dir = self._rotate_vector_around_axis(
            dir_np, bend_axis, bend_sign * self.ELBOW_BEND_ANGLE
        )
        upper_arm_dir = upper_arm_dir / np.linalg.norm(upper_arm_dir)

        # Calculate joint positions
        # Elbow = shoulder + upper_arm_direction * UPPER_ARM_LENGTH
        elbow_pos = shoulder_np + upper_arm_dir * self.UPPER_ARM_LENGTH

        # Wrist = elbow + pointing_direction * FOREARM_LENGTH
        wrist_pos = elbow_pos + dir_np * self.FOREARM_LENGTH

        # Hand center = wrist + pointing_direction * HAND_RADIUS (offset by radius)
        hand_pos = wrist_pos + dir_np * self.HAND_RADIUS

        # Spawn upper arm (shoulder to elbow)
        self._spawn_capsule_segment(
            arm_paths['upper'],
            shoulder_np,
            elbow_pos,
            self.UPPER_ARM_RADIUS
        )

        # Spawn forearm (elbow to wrist)
        self._spawn_capsule_segment(
            arm_paths['forearm'],
            elbow_pos,
            wrist_pos,
            self.FOREARM_RADIUS
        )

        # Spawn wrist joint (sphere at wrist position for better OpenPose detection)
        # This gives the depth sensor a solid target at the wrist location
        self._spawn_hand_sphere(
            arm_paths['wrist'],
            wrist_pos,
            self.WRIST_RADIUS
        )

        # Spawn hand (sphere at hand end)
        self._spawn_hand_sphere(
            arm_paths['hand'],
            hand_pos,
            self.HAND_RADIUS
        )

        print(f"[HumanArmController] Spawned {side} arm segments:")
        print(f"  Shoulder: ({shoulder_np[0]:.3f}, {shoulder_np[1]:.3f}, {shoulder_np[2]:.3f})")
        print(f"  Elbow:    ({elbow_pos[0]:.3f}, {elbow_pos[1]:.3f}, {elbow_pos[2]:.3f})")
        print(f"  Wrist:    ({wrist_pos[0]:.3f}, {wrist_pos[1]:.3f}, {wrist_pos[2]:.3f}) [sphere r={self.WRIST_RADIUS}]")
        print(f"  Hand:     ({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f})")
        print(f"  Direction: {direction}")

        return True

    def remove_pointing_arms(self):
        """Remove all spawned arm geometry (upper arm, forearm, hand for both sides)."""
        if not self.stage:
            return

        # Remove all segments for both arms
        for arm_paths in [self.left_arm_paths, self.right_arm_paths]:
            for segment_name, segment_path in arm_paths.items():
                prim = self.stage.GetPrimAtPath(segment_path)
                if prim and prim.IsValid():
                    self.stage.RemovePrim(segment_path)
                    print(f"[HumanArmController] Removed {segment_name}: {segment_path}")

        # Clear stored arm directions for both arms
        self.current_left_arm_direction = None
        self.current_right_arm_direction = None

    def set_pose(self, pose_name):
        """Set the character's arm pose by spawning/removing arm geometry."""
        if not self.initialized:
            print("[HumanArmController] Not initialized")
            return False

        if pose_name not in self.POSES:
            print(f"[HumanArmController] Unknown pose: {pose_name}")
            return False

        print(f"[HumanArmController] Setting pose: {pose_name}")

        try:
            if pose_name == "neutral":
                # No arms visible
                self.remove_pointing_arms()
                self.active_pointing_arm = None  # No arm pointing

            elif pose_name == "right_point":
                # Right arm pointing strongly lateral-right (~70° outward, ~35° down)
                # Left arm in resting position
                self.remove_pointing_arms()
                self.spawn_pointing_arm("right", direction=self.RIGHT_LATERAL_DIRECTION)
                self.spawn_pointing_arm("left", direction=self.LEFT_RESTING_DIRECTION)
                self.active_pointing_arm = "right"  # Right arm is the pointing arm

            elif pose_name == "left_point_far":
                # Left arm pointing forward at distant ground (~15° left, ~20° down)
                # Right arm in resting position
                self.remove_pointing_arms()
                self.spawn_pointing_arm("left", direction=self.LEFT_FORWARD_DIRECTION)
                self.spawn_pointing_arm("right", direction=self.RIGHT_RESTING_DIRECTION)
                self.active_pointing_arm = "left"  # Left arm is the pointing arm

            elif pose_name == "left_point_lateral":
                # Left arm pointing strongly lateral-left (~70° outward, ~35° down)
                # Right arm in resting position
                self.remove_pointing_arms()
                self.spawn_pointing_arm("left", direction=self.LEFT_LATERAL_DIRECTION)
                self.spawn_pointing_arm("right", direction=self.RIGHT_RESTING_DIRECTION)
                self.active_pointing_arm = "left"  # Left arm is the pointing arm

            self.current_pose = pose_name
            print(f"[HumanArmController] Pose '{pose_name}' applied, active_pointing_arm={self.active_pointing_arm}")
            return True

        except Exception as e:
            print(f"[HumanArmController] Error setting pose: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_pose_by_index(self, index):
        """Set pose by index number."""
        pose_names = list(self.POSES.keys())
        if 0 <= index < len(pose_names):
            return self.set_pose(pose_names[index])
        print(f"[HumanArmController] Invalid pose index: {index}")
        return False

    def get_current_pose(self):
        return self.current_pose

    def update(self):
        """Called per-frame to process queued pose changes.

        ROS callbacks queue pose changes since they run on a different thread.
        This method processes them on the main thread where USD stage
        modifications are safe.
        """
        # Process any queued pose changes (from ROS callbacks)
        try:
            while True:
                pose_name = self.pose_queue.get_nowait()
                self.set_pose(pose_name)
        except Empty:
            pass

        # Publish left arm pose every frame (dynamically tracks character prim position)
        if self.current_left_arm_direction is not None:
            self._publish_left_arm_pose()

        # Publish ground truth for the active pointing arm (set by human_point_follower)
        if self.active_pointing_arm == 'left' and self.current_left_arm_direction is not None:
            shoulder_world = self._get_shoulder_world_position("left")
            self._publish_ground_truth_intersection(shoulder_world, "left")
        elif self.active_pointing_arm == 'right' and self.current_right_arm_direction is not None:
            shoulder_world = self._get_shoulder_world_position("right")
            self._publish_ground_truth_intersection(shoulder_world, "right")

    def list_poses(self):
        print("\n[HumanArmController] Available Poses:")
        for idx, (name, data) in enumerate(self.POSES.items()):
            print(f"  {idx}: {name} - {data['description']}")
        return list(self.POSES.keys())

    def init_ros_subscribers(self):
        """Initialize ROS subscribers for pose control."""
        try:
            self.pose_sub = rospy.Subscriber(
                '/human/arm_pose',
                String,
                self._ros_pose_callback,
                queue_size=1
            )
            self.pose_index_sub = rospy.Subscriber(
                '/human/arm_pose_index',
                Int32,
                self._ros_pose_index_callback,
                queue_size=1
            )
            # Publisher for left arm pose
            self.left_arm_pose_pub = rospy.Publisher(
                '/human/left_arm_pose',
                PoseStamped,
                queue_size=1
            )
            # Publisher for ground truth intersection point (for RViz visualization)
            self.ground_truth_pub = rospy.Publisher(
                '/pointing/ground_truth',
                PointStamped,
                queue_size=1
            )
            # Subscriber for active pointing arm (from human_point_follower)
            self.active_arm_sub = rospy.Subscriber(
                '/pointing/active_arm',
                String,
                self._active_arm_callback,
                queue_size=1
            )
            print("[HumanArmController] ROS subscribers initialized")
            print("  - /human/arm_pose (String)")
            print("  - /human/arm_pose_index (Int32)")
            print("  - /human/left_arm_pose (PoseStamped) [publisher]")
            print("  - /pointing/ground_truth (PointStamped) [publisher]")
            print("  - /pointing/active_arm (String) [subscriber]")
        except Exception as e:
            print(f"[HumanArmController] ROS init error: {e}")

    def _ros_pose_callback(self, msg):
        """Queue pose change for main thread processing."""
        self.pose_queue.put(msg.data)

    def _ros_pose_index_callback(self, msg):
        """Queue pose change by index for main thread processing."""
        pose_names = list(self.POSES.keys())
        if 0 <= msg.data < len(pose_names):
            self.pose_queue.put(pose_names[msg.data])

    def _active_arm_callback(self, msg):
        """Callback for active arm topic from human_point_follower.

        This callback is intentionally a no-op. Ground truth is determined
        solely by set_pose() based on the actual Isaac Sim arm geometry,
        not by the robot's detection (which may be inaccurate).
        """
        # Ground truth comes from Isaac Sim pose, not robot detection
        pass


# Singleton instance
_controller_instance = None


def get_arm_controller(character_path="/World/Characters/F_Business_02"):
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = HumanArmController(character_path)
    return _controller_instance


def initialize_arm_controller(character_path="/World/Characters/F_Business_02"):
    controller = get_arm_controller(character_path)
    if controller.initialize():
        try:
            if rospy.core.is_initialized():
                controller.init_ros_subscribers()
        except:
            pass
        return controller
    return None
