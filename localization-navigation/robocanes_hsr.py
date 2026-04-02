# Copyright (c) 2023, Toyota Motor Corporation
# Copyright (c) 2023, MID Academic Promotions, Inc.
# All rights reserved.

from robocanes_hsr_isaac_world import sim_app

import os
import math
import omni.ui
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils import extensions, viewports, stage, nucleus, prims
from omni.isaac.core.utils.render_product import create_hydra_texture
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni.kit.commands
from omni.isaac.dynamic_control import _dynamic_control
from pxr import Usd, Sdf, Gf, UsdPhysics, UsdLux, PhysxSchema, UsdGeom
from omni.isaac.core.utils.prims import set_targets
# from omni.isaac.core.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
import omni.graph.core as og
import omni.replicator.core as rep
# from omni.isaac.sensor import _sensor
from isaacsim.sensors.physics import _sensor
import rospy
import tf.transformations
import actionlib
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PoseWithCovarianceStamped
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal, GripperCommandAction, GripperCommandActionGoal
from actionlib_msgs.msg import GoalStatusArray, GoalStatus, GoalID
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from tmc_control_msgs.msg import GripperApplyEffortAction, GripperApplyEffortResult, GripperApplyEffortFeedback

from rospkg import RosPack; rp = RosPack()
usd_repo_path = rp.get_path('usd')

# enable ROS bridge extension
extensions.enable_extension("isaacsim.ros1.bridge")
extensions.enable_extension("isaacsim.sensors.physx")
extensions.enable_extension("isaacsim.util.debug_draw")
extensions.enable_extension("semu.robotics.ros_bridge")

# extensions.enable_extension("omni.isaac.ros_bridge")
# extensions.enable_extension("omni.isaac.range_sensor")
# extensions.enable_extension("omni.isaac.debug_draw")
# extensions.enable_extension("semu.robotics.ros_bridge")

from semu.robotics.ros_bridge.ogn.nodes.OgnROS1ActionFollowJointTrajectory import InternalState as semuInternalState
from semu.robotics.ros_bridge.ogn.nodes.OgnROS1ActionGripperCommand import InternalState as semuGripperInternalState

robot_prefix = 'hsrb'

class odom_trajectory_action_server(semuInternalState):
    def _init_articulation(self) -> None:
        # get articulation
        path = self.articulation_path
        self._articulation = self.dci.get_articulation(path)
        if self._articulation == _dynamic_control.INVALID_HANDLE:
            print("[Warning][semu.robotics.ros_bridge] ROS1 FollowJointTrajectory: {} is not an articulation".format(path))
            return
        for dof_name in ['odom_x', 'odom_y', 'odom_t']:
            self._joints[dof_name] = 0.0
        self._odometry = None
        self._remaining_start_time = None

    def _set_joint_position(self, name: str, target_position: float) -> None:
        self._joints[name] = target_position

    def _get_joint_position(self, name: str) -> float:
        return self._joints[name]

    def step(self, dt: float) -> None:
        if self._action_goal is not None and self._action_goal_handle is not None:
            # end of trajectory
            if self._odometry is not None and self._action_point_index >= len(self._action_goal.trajectory.points):
                diff = 0.0
                diff += abs(self._get_joint_position('odom_x') - self._odometry.x)
                diff += abs(self._get_joint_position('odom_y') - self._odometry.y)
                diff += abs(self._get_joint_position('odom_t') - self._odometry.ang)
                # rospy.loginfo('omni trajectory remaining: %f', diff)
                if self._remaining_start_time is None:
                    self._remaining_start_time = rospy.get_time()
                time_passed = rospy.get_time() - self._remaining_start_time
                if diff > 0.001 and time_passed < 5.0:
                    return
            else:
                self._remaining_start_time = None
        super().step(dt)


class arm_trajectory_action_server(semuInternalState):
    def _set_joint_position(self, name: str, target_position: float) -> None:
        if name == 'arm_lift_joint':
            super()._set_joint_position('torso_lift_joint', target_position / 2.0)
        if name in ['arm_flex_joint', 'arm_lift_joint', 'wrist_flex_joint', 'arm_roll_joint']:
            target_position = -target_position
        super()._set_joint_position(name, target_position)

    def _get_joint_position(self, name: str) -> float:
        v = super()._get_joint_position(name)
        if name in ['arm_flex_joint', 'arm_lift_joint', 'wrist_flex_joint', 'arm_roll_joint']:
            return -v
        return v


class head_trajectory_action_server(semuInternalState):
    pass


class gripper_trajectory_action_server(semuInternalState):
    def _set_joint_position(self, name: str, target_position: float) -> None:
        if name == 'hand_motor_joint':
            super()._set_joint_position('hand_l_proximal_joint', target_position)
            super()._set_joint_position('hand_l_distal_joint', -target_position)
            super()._set_joint_position('hand_r_proximal_joint', target_position)
            super()._set_joint_position('hand_r_distal_joint', -target_position)


class gripper_command_action_server(semuGripperInternalState):
    def __init__(self):
        super().__init__()

        global robot_prefix

        self.gripper_joints_paths = [
            f'/World{robot_prefix}/hand_palm_link/hand_l_proximal_joint',
            f'/World{robot_prefix}/hand_l_mimic_distal_link/hand_l_distal_joint',
            f'/World{robot_prefix}/hand_palm_link/hand_r_proximal_joint',
            f'/World{robot_prefix}/hand_r_mimic_distal_link/hand_r_distal_joint'
        ]
        self.gripper_joints_paths = [
            f'/World{robot_prefix}/hand_palm_link/hand_l_proximal_joint',
            f'/World{robot_prefix}/hand_l_mimic_distal_link/hand_l_distal_joint',
            f'/World{robot_prefix}/hand_palm_link/hand_r_proximal_joint',
            f'/World{robot_prefix}/hand_r_mimic_distal_link/hand_r_distal_joint'
        ]

    def _set_joint_position(self, name: str, target_position: float) -> None:
        if name == 'hand_l_distal_joint':
            target_position = -target_position
        if name == 'hand_r_distal_joint':
            target_position = -target_position
        super()._set_joint_position(name, target_position)


class gripper_apply_force_action_server(gripper_command_action_server):
    def __init__(self):
        super().__init__()
        self._action_result_message = GripperApplyEffortResult()
        self._action_feedback_message = GripperApplyEffortFeedback()
        self._inverse_direction = False

    def _get_joint_effort(self, name: str) -> float:
        effort = self.dci.get_dof_state(self._joints[name]["dof"], _dynamic_control.STATE_EFFORT).effort
        return effort

    # most of this part is copied from:
    #  https://github.com/Toni-SM/semu.robotics.ros_bridge/blob/main/exts/semu.robotics.ros_bridge/semu/robotics/ros_bridge/ogn/nodes/OgnROS1ActionGripperCommand.py
    def step(self, dt: float) -> None:
        if not self.initialized:
            return
        if not self._joints:
            self._init_articulation()
            return
        if self._action_goal is not None and self._action_goal_handle is not None:
            target_effort = self._action_goal.effort
            if self._inverse_direction:
                target_effort = -target_effort

            self.dci.wake_up_articulation(self._articulation)
            for name in self._joints:
                if target_effort >= 0.0:
                    self._set_joint_position(name, 0.0)
                else:
                    self._set_joint_position(name, math.pi)

            # compare target and current effort
            effort = 0
            effort_reached = True
            for name in self._joints:
                effort = self._get_joint_effort(name)
                if abs(effort) - abs(target_effort) < 0.0:
                    effort_reached = False
                    break
            if effort_reached:
                self._action_goal = None
                self._action_result_message.effort = effort
                self._action_result_message.stalled = False
                if self._action_goal_handle is not None:
                    self._action_goal_handle.set_succeeded(self._action_result_message)
                    self._action_goal_handle = None
                return

            # check if joints are moving (if not, results "stalled")
            current_position_sum = 0
            for name in self._joints:
                position = self._get_joint_position(name)
                current_position_sum += position
            if abs(current_position_sum - self._action_previous_position_sum) < 1e-6:
                self._action_goal = None
                self._action_result_message.effort = effort
                self._action_result_message.stalled = True
                if self._action_goal_handle is not None:
                    self._action_goal_handle.set_succeeded(self._action_result_message)
                    self._action_goal_handle = None
                return
            self._action_previous_position_sum = current_position_sum

            # check timeout
            time_passed = rospy.get_time() - self._action_start_time
            if time_passed >= self._action_timeout:
                self._action_goal = None
                if self._action_goal_handle is not None:
                    self._action_goal_handle.set_aborted()
                    self._action_goal_handle = None

class hsr_config:
    def __init__(self) -> None:
        self.use_ros = True


wheel_separation = 0.266
wheel_radius = 0.04
wheel_offset = 0.11


class BaseOdometry:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.ang = 0.0


class JointSpace:
    def __init__(self) -> None:
        self.vel_wheel_l = 0.0
        self.vel_wheel_r = 0.0
        self.vel_steer = 0.0


class CartSpace:
    def __init__(self) -> None:
        self.dot_x = 0.0
        self.dot_y = 0.0
        self.dot_r = 0.0


class VehicleState:
    def __init__(self) -> None:
        self.steer_angle = 0.0


# Dynamics of offset diff drive vehicle
#  Equations are from the paper written by Masayoshi Wada etal.
#  https://www.jstage.jst.go.jp/article/jrsj1983/18/8/18_8_1166/_pdf
def forward_dynamics(input: JointSpace, state: VehicleState) -> CartSpace:
    global wheel_separation, wheel_offset, wheel_radius
    cos_s = math.cos(state.steer_angle)
    sin_s = math.sin(state.steer_angle)
    output = CartSpace()

    # Debugging
    # print(f'input.vel_wheel_r:', input.vel_wheel_r)
    # print(f'input.vel_wheel_l:', input.vel_wheel_l)

    output.dot_x = (wheel_radius / 2.0 * cos_s - wheel_radius * wheel_offset / wheel_separation * sin_s) * input.vel_wheel_r + (wheel_radius / 2.0 * cos_s + wheel_radius * wheel_offset / wheel_separation * sin_s) * input.vel_wheel_l
    output.dot_y = (wheel_radius / 2.0 * sin_s + wheel_radius * wheel_offset / wheel_separation * cos_s) * input.vel_wheel_r + (wheel_radius / 2.0 * sin_s - wheel_radius * wheel_offset / wheel_separation * cos_s) * input.vel_wheel_l


    # Debugging
    # first = wheel_radius / wheel_separation * input.vel_wheel_r
    # second = wheel_radius / wheel_separation * input.vel_wheel_l
    # third = input.vel_steer

    # print(f'first:', first)
    # print(f'second:', second)
    # print(f'third:', third)

    output.dot_r = wheel_radius / wheel_separation * input.vel_wheel_r - wheel_radius / wheel_separation * input.vel_wheel_l - input.vel_steer

    if (abs(output.dot_r) < 0.01):
        output.dot_r = 0.0

    return output


def inverse_dynamics(input: CartSpace, state: VehicleState) -> JointSpace:
    global wheel_separation, wheel_offset, wheel_radius
    cos_s = math.cos(state.steer_angle)
    sin_s = math.sin(state.steer_angle)
    output = JointSpace()
    output.vel_wheel_r = (cos_s / wheel_radius - wheel_separation * sin_s / 2.0 / wheel_radius / wheel_offset) * input.dot_x + (sin_s / wheel_radius + wheel_separation * cos_s / 2.0 / wheel_radius / wheel_offset) * input.dot_y
    output.vel_wheel_l = (cos_s / wheel_radius + wheel_separation * sin_s / 2.0 / wheel_radius / wheel_offset) * input.dot_x + (sin_s / wheel_radius - wheel_separation * cos_s / 2.0 / wheel_radius / wheel_offset) * input.dot_y
    output.vel_steer = -sin_s / wheel_offset * input.dot_x + cos_s / wheel_offset * input.dot_y - input.dot_r
    return output


class hsr:

    def __init__(self, prefix='/hsrb', config=None, spawn_config=None) -> None:
        if config is None:
            config = hsr_config()

        # rospy.init_node("isaac_sim_hsr", anonymous=True, disable_signals=True, log_level=rospy.ERROR)

        # self.acc_odom_angular_z = 0

        global robot_prefix
        robot_prefix = prefix

        self.prefix = prefix
        self.simulation_context = None
        self.art = None
        #self.hsr = stage.add_reference_to_stage("https://cdn.statically.io/gh/hsr-project/hsrb_usd/main/hsrb4s.usd", "/World" + self.prefix)
        # self.hsr = stage.add_reference_to_stage(os.path.dirname(os.path.abspath(__file__)) + "/usd/hsrb/hsrb4s.usd", "/World" + self.prefix)

        if spawn_config:
            self.hsr = prims.create_prim(
                prim_path=f'/World' + self.prefix,
                # usd_path=os.path.dirname(os.path.abspath(__file__)) + "/usd/hsrb/hsrb4s.usd",
                usd_path=os.path.join(usd_repo_path, 'hsrb', 'hsrb4s.usd'),
                translation=spawn_config['translation'],
                orientation=euler_angles_to_quat(spawn_config['orientation']),
                scale=spawn_config['scale'],
                semantic_label='hsr'
            )
           
        else:
             self.hsr = prims.create_prim(
                prim_path=f'/World' + self.prefix,
                # usd_path=os.path.dirname(os.path.abspath(__file__)) + "/usd/hsrb/hsrb4s.usd",
                usd_path=os.path.join(usd_repo_path, 'hsrb', 'hsrb4s.usd'),
                translation=[0, 0, 0],
                orientation=euler_angles_to_quat([0, 0, 0]),
                scale=[1, 1, 1],
                semantic_label='hsr'
            )

        self.set_base_joint_and_material()

        self.create_cameras()
        self.create_lidar()

        self.create_imu()
        self._if_imu = _sensor.acquire_imu_sensor_interface()
        self.imu_pub = rospy.Publisher(self.prefix + '/base_imu/data', Imu, queue_size=5)

        self.on_posestimate_pose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.on_pose_estimate_pose)

        # self.laserscan_pose_sub = rospy.Subscriber(self.prefix + '/laser_scan_matcher/pose', PoseStamped, self.on_laserscan_pose)
        self.laser_odom_pub = rospy.Publisher(self.prefix + '/laser_odom', Odometry, queue_size=5)
        self.base_odom_pub = rospy.Publisher(self.prefix + '/base_controller/odom', Odometry, queue_size=5)

        self.create_force_sensor()
        #self.robots = ArticulationView(prim_paths_expr=self.prefix, name="hsr_view")

        # dynamic control can also be used to interact with the imported urdf.
        self.dc = _dynamic_control.acquire_dynamic_control_interface()

        self.cmd_vel_msg = None
        self.last_cmd_vel_time = 0.0
        rospy.Subscriber(self.prefix + "/command_velocity", Twist, self.on_cmd_vel)
        self.odometry_ = BaseOdometry()
        self.vel_limit_steer_ = 8.0
        self.vel_limit_wheel_ = 8.0

        self.joint_state_pub = rospy.Publisher(self.prefix + '/joint_states', JointState, queue_size=5)

        def init_action_server(srv, name, msg):
            srv.articulation_path = '/World' + self.prefix
            srv.usd_context = omni.usd.get_context()
            srv.dci = self.dc
            action_topic_name = self.prefix + "/" + name
            srv.action_server = actionlib.ActionServer(
                action_topic_name,
                msg,
                goal_cb=srv.on_goal,
                cancel_cb=srv.on_cancel,
                auto_start=False)
            srv.action_server.start()
            if msg == FollowJointTrajectoryAction:
                srv.action_client = actionlib.SimpleActionClient(action_topic_name, msg)
                def command_topic_callback(msg, args):
                    srv = args[0]
                    goal = FollowJointTrajectoryGoal(trajectory=msg)
                    if srv._action_goal is not None:
                        # reject if joints don't match
                        for name in goal.trajectory.joint_names:
                            if name not in self._joints:
                                print("[Warning][semu.robotics.ros_bridge] ROS1 FollowJointTrajectory: joints don't match ({} not in {})" \
                                    .format(name, list(self._joints.keys())))
                                return
                        # check initial position
                        if goal.trajectory.points[0].time_from_start.to_sec():
                            initial_point = JointTrajectoryPoint(positions=[srv._get_joint_position(name) for name in goal.trajectory.joint_names],
                                                                time_from_start=rospy.Duration())
                            goal.trajectory.points.insert(0, initial_point)
                        # store goal data
                        srv._action_goal = goal
                        srv._action_point_index = 1
                        srv._action_start_time = rospy.get_time()
                        srv._action_feedback_message.joint_names = list(goal.trajectory.joint_names)
                    else:
                        srv.action_client.send_goal(goal)
                srv.topic_interface = rospy.Subscriber(action_topic_name.replace('/follow_joint_trajectory', '/command'), JointTrajectory, command_topic_callback, (srv,) )
            srv.initialized = True

        self.arm_trajectory_action_server = arm_trajectory_action_server()
        init_action_server(self.arm_trajectory_action_server, 'arm_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.head_trajectory_action_server = head_trajectory_action_server()
        init_action_server(self.arm_trajectory_action_server, 'head_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.odom_trajectory_action_server = odom_trajectory_action_server()
        init_action_server(self.odom_trajectory_action_server, 'omni_base_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.gripper_trajectory_action_server = gripper_trajectory_action_server()
        init_action_server(self.gripper_trajectory_action_server, 'gripper_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        self.gripper_apply_force_action_server = gripper_apply_force_action_server()
        init_action_server(self.gripper_apply_force_action_server, 'gripper_controller/apply_force', GripperApplyEffortAction)

        #self.gripper_command_action_server = gripper_command_action_server()
        #init_action_server(self.gripper_command_action_server, 'gripper_controller/grasp', GripperCommandAction)
        self.gripper_command_action_server = gripper_apply_force_action_server()
        self.gripper_command_action_server._inverse_direction = True
        init_action_server(self.gripper_command_action_server, 'gripper_controller/grasp', GripperApplyEffortAction)

    def create_cameras(self) -> None:
        # Creating a Camera prim
        # l_camera_prim = UsdGeom.Camera(omni.usd.get_context().get_stage().DefinePrim('/World' + self.prefix + "/head_l_stereo_camera_link/Camera", "Camera"))
        # xform_api = UsdGeom.XformCommonAPI(l_camera_prim)
        # xform_api.SetRotate((180, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        # l_camera_prim.GetHorizontalApertureAttr().Set(1280 * 0.003)
        # l_camera_prim.GetVerticalApertureAttr().Set(960 * 0.003)
        # l_camera_prim.GetProjectionAttr().Set("perspective")
        # l_camera_prim.GetFocalLengthAttr().Set(968.770306867 * 0.003)  #  (1280/2) / tan(1.16762527/2)
        # l_camera_prim.GetFocusDistanceAttr().Set(400)

        # Creating a Camera prim
        # r_camera_prim = UsdGeom.Camera(omni.usd.get_context().get_stage().DefinePrim('/World' + self.prefix + "/head_r_stereo_camera_link/Camera", "Camera"))
        # xform_api = UsdGeom.XformCommonAPI(r_camera_prim)
        # xform_api.SetRotate((180, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        # r_camera_prim.GetHorizontalApertureAttr().Set(1280 * 0.003)
        # r_camera_prim.GetVerticalApertureAttr().Set(960 * 0.003)
        # r_camera_prim.GetProjectionAttr().Set("perspective")
        # r_camera_prim.GetFocalLengthAttr().Set(968.770306867 * 0.003)  #  (1280/2) / tan(1.16762527/2)
        # r_camera_prim.GetFocusDistanceAttr().Set(400)

        # Creating a Camera prim
        rgbd_camera_prim = UsdGeom.Camera(omni.usd.get_context().get_stage().DefinePrim('/World' + self.prefix + "/head_rgbd_sensor_link/Camera", "Camera"))
        xform_api = UsdGeom.XformCommonAPI(rgbd_camera_prim)
        xform_api.SetRotate((180, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        rgbd_camera_prim.GetHorizontalApertureAttr().Set(640 * 0.003)
        rgbd_camera_prim.GetVerticalApertureAttr().Set(480 * 0.003)
        rgbd_camera_prim.GetProjectionAttr().Set("perspective")
        rgbd_camera_prim.GetFocalLengthAttr().Set(554.382712823 * 0.003)  #  (640/2) / tan(1.047/2)
        rgbd_camera_prim.GetFocusDistanceAttr().Set(400)

        # Slight modifications by chris, will only affect render.
        rgbd_camera_prim.GetClippingRangeAttr().Set((0.1, 1000000.0))

        # Creating a Camera prim
        # hand_camera_prim = UsdGeom.Camera(omni.usd.get_context().get_stage().DefinePrim('/World' + self.prefix + "/hand_camera_frame/Camera", "Camera"))
        # xform_api = UsdGeom.XformCommonAPI(hand_camera_prim)
        # xform_api.SetRotate((180, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        # hand_camera_prim.GetHorizontalApertureAttr().Set(640 * 0.003)
        # hand_camera_prim.GetVerticalApertureAttr().Set(480 * 0.003)
        # hand_camera_prim.GetProjectionAttr().Set("perspective")
        # hand_camera_prim.GetFocalLengthAttr().Set(205.469637099 * 0.003)  #  (640/2) / tan(2.0/2)
        # hand_camera_prim.GetFocusDistanceAttr().Set(400)

        # Creating a action graph with ROS component nodes
        try:
            og.Controller.edit(
                {"graph_path": "/ros_controllers", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PublishClock", "isaacsim.ros1.bridge.ROS1PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                    ],
                },
            )

            # og.Controller.edit(
            #     {"graph_path": "/ros_controllers", "evaluator_name": "execution"},
            #     {
            #         og.Controller.Keys.CREATE_NODES: [
            #             ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
            #             ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
            #             ("PublishClock", "omni.isaac.ros_bridge.ROS1PublishClock"),
            #         ],
            #         og.Controller.Keys.CONNECT: [
            #             ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
            #             ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            #         ],
            #         og.Controller.Keys.SET_VALUES: [
            #         ],
            #     },
            # )
            # (self.ros_camera_graph_l, _, _, _) = og.Controller.edit(
            #     {
            #         "graph_path": "/head_l_camera",
            #         "evaluator_name": "push",
            #         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            #     },
            #     {
            #         og.Controller.Keys.CREATE_NODES: [
            #             ("OnTick", "omni.graph.action.OnTick"),
            #             ("createRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
            #             ("cameraHelperRgb", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #             ("cameraHelperInfo", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #         ],
            #         og.Controller.Keys.CONNECT: [
            #             ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
            #         ],
            #         og.Controller.Keys.SET_VALUES: [
            #             ("createRenderProduct.inputs:width", 1280),
            #             ("createRenderProduct.inputs:height", 960),
            #             ("cameraHelperRgb.inputs:frameId", "head_l_stereo_camera_frame"),
            #             ("cameraHelperRgb.inputs:topicName", self.prefix + "/head_l_stereo_camera/image_rect_color"),
            #             ("cameraHelperRgb.inputs:type", "rgb"),
            #             ("cameraHelperInfo.inputs:frameId", "head_l_stereo_camera_frame"),
            #             ("cameraHelperInfo.inputs:topicName", self.prefix + "/head_l_stereo_camera/camera_info"),
            #             ("cameraHelperInfo.inputs:type", "camera_info"),
            #         ],
            #     },
            # )
            # (self.ros_camera_graph_r, _, _, _) = og.Controller.edit(
            #     {
            #         "graph_path": "/head_r_camera",
            #         "evaluator_name": "push",
            #         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            #     },
            #     {
            #         og.Controller.Keys.CREATE_NODES: [
            #             ("OnTick", "omni.graph.action.OnTick"),
            #             ("createRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
            #             ("cameraHelperRgb", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #             ("cameraHelperInfo", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #         ],
            #         og.Controller.Keys.CONNECT: [
            #             ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
            #         ],
            #         og.Controller.Keys.SET_VALUES: [
            #             ("createRenderProduct.inputs:width", 1280),
            #             ("createRenderProduct.inputs:height", 960),
            #             ("cameraHelperRgb.inputs:frameId", "head_r_stereo_camera_frame"),
            #             ("cameraHelperRgb.inputs:topicName", self.prefix + "/head_r_stereo_camera/image_rect_color"),
            #             ("cameraHelperRgb.inputs:type", "rgb"),
            #             ("cameraHelperInfo.inputs:frameId", "head_r_stereo_camera_frame"),
            #             ("cameraHelperInfo.inputs:topicName", self.prefix + "/head_r_stereo_camera/camera_info"),
            #             ("cameraHelperInfo.inputs:type", "camera_info"),
            #         ],
            #     },
            # )

            (self.ros_camera_graph_rgbd, _, _, _) = og.Controller.edit(
                {
                    "graph_path": "/head_rgbd_camera",
                    "evaluator_name": "push",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
                },
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("createRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("cameraHelperRgb", "isaacsim.ros1.bridge.ROS1CameraHelper"),
                        ("cameraHelperInfo", "isaacsim.ros1.bridge.ROS1CameraHelper"),
                        ("cameraHelperDepth", "isaacsim.ros1.bridge.ROS1CameraHelper"),
                        ("cameraHelperDepthInfo", "isaacsim.ros1.bridge.ROS1CameraHelper"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
                        ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                        ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                        ("createRenderProduct.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                        ("createRenderProduct.outputs:renderProductPath", "cameraHelperDepthInfo.inputs:renderProductPath"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("createRenderProduct.inputs:width", 640),
                        ("createRenderProduct.inputs:height", 480),
                        ("cameraHelperRgb.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
                        ("cameraHelperRgb.inputs:topicName", self.prefix + "/head_rgbd_sensor/rgb/image_rect_color"),
                        ("cameraHelperRgb.inputs:type", "rgb"),
                        ("cameraHelperInfo.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
                        ("cameraHelperInfo.inputs:topicName", self.prefix + "/head_rgbd_sensor/rgb/camera_info"),
                        ("cameraHelperInfo.inputs:type", "camera_info"),
                        ("cameraHelperDepth.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
                        ("cameraHelperDepth.inputs:topicName", self.prefix + "/head_rgbd_sensor/depth_registered/image_rect_raw"),
                        ("cameraHelperDepth.inputs:type", "depth"),
                        ("cameraHelperDepthInfo.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
                        ("cameraHelperDepthInfo.inputs:topicName", self.prefix + "/head_rgbd_sensor/depth_registered/camera_info"),
                        ("cameraHelperDepthInfo.inputs:type", "camera_info"),
                    ],
                },
            )


            # (self.ros_camera_graph_rgbd, _, _, _) = og.Controller.edit(
            #     {
            #         "graph_path": "/head_rgbd_camera",
            #         "evaluator_name": "push",
            #         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            #     },
            #     {
            #         og.Controller.Keys.CREATE_NODES: [
            #             ("OnTick", "omni.graph.action.OnTick"),
            #             ("createRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
            #             ("cameraHelperRgb", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #             ("cameraHelperInfo", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #             ("cameraHelperDepth", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #             ("cameraHelperDepthInfo", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #         ],
            #         og.Controller.Keys.CONNECT: [
            #             ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperDepthInfo.inputs:renderProductPath"),
            #         ],
            #         og.Controller.Keys.SET_VALUES: [
            #             ("createRenderProduct.inputs:width", 640),
            #             ("createRenderProduct.inputs:height", 480),
            #             ("cameraHelperRgb.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
            #             ("cameraHelperRgb.inputs:topicName", self.prefix + "/head_rgbd_sensor/rgb/image_rect_color"),
            #             ("cameraHelperRgb.inputs:type", "rgb"),
            #             ("cameraHelperInfo.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
            #             ("cameraHelperInfo.inputs:topicName", self.prefix + "/head_rgbd_sensor/rgb/camera_info"),
            #             ("cameraHelperInfo.inputs:type", "camera_info"),
            #             ("cameraHelperDepth.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
            #             ("cameraHelperDepth.inputs:topicName", self.prefix + "/head_rgbd_sensor/depth_registered/image_rect_raw"),
            #             ("cameraHelperDepth.inputs:type", "depth"),
            #             ("cameraHelperDepthInfo.inputs:frameId", "head_rgbd_sensor_rgb_frame"),
            #             ("cameraHelperDepthInfo.inputs:topicName", self.prefix + "/head_rgbd_sensor/depth_registered/camera_info"),
            #             ("cameraHelperDepthInfo.inputs:type", "camera_info"),
            #         ],
            #     },
            # )


            # (self.ros_camera_graph_hand, _, _, _) = og.Controller.edit(
            #     {
            #         "graph_path": "/hand_camera",
            #         "evaluator_name": "push",
            #         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            #     },
            #     {
            #         og.Controller.Keys.CREATE_NODES: [
            #             ("OnTick", "omni.graph.action.OnTick"),
            #             ("createRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
            #             ("cameraHelperRgb", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #             ("cameraHelperInfo", "omni.isaac.ros_bridge.ROS1CameraHelper"),
            #         ],
            #         og.Controller.Keys.CONNECT: [
            #             ("OnTick.outputs:tick", "createRenderProduct.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            #             ("createRenderProduct.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            #             ("createRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
            #         ],
            #         og.Controller.Keys.SET_VALUES: [
            #             ("createRenderProduct.inputs:width", 640),
            #             ("createRenderProduct.inputs:height", 480),
            #             ("cameraHelperRgb.inputs:frameId", "hand_camera_frame"),
            #             ("cameraHelperRgb.inputs:topicName", self.prefix + "/hand_camera/image_raw"),
            #             ("cameraHelperRgb.inputs:type", "rgb"),
            #             ("cameraHelperInfo.inputs:frameId", "hand_camera_frame"),
            #             ("cameraHelperInfo.inputs:topicName", self.prefix + "/hand_camera/camera_info"),
            #             ("cameraHelperInfo.inputs:type", "camera_info"),
            #         ],
            #     },
            # )
        except Exception as e:
            raise(e)
            #print(e)

        # set_targets(
        #     prim=stage.get_current_stage().GetPrimAtPath("/head_l_camera/createRenderProduct"),
        #     attribute="inputs:cameraPrim",
        #     target_prim_paths=['/World' + self.prefix + "/head_l_stereo_camera_link/Camera"],
        # )

        # set_targets(
        #     prim=stage.get_current_stage().GetPrimAtPath("/head_r_camera/createRenderProduct"),
        #     attribute="inputs:cameraPrim",
        #     target_prim_paths=['/World' + self.prefix + "/head_r_stereo_camera_link/Camera"],
        # )

        set_targets(
            prim=stage.get_current_stage().GetPrimAtPath("/head_rgbd_camera/createRenderProduct"),
            attribute="inputs:cameraPrim",
            target_prim_paths=['/World' + self.prefix + "/head_rgbd_sensor_link/Camera"],
        )

        # set_targets(
        #     prim=stage.get_current_stage().GetPrimAtPath("/hand_camera/createRenderProduct"),
        #     attribute="inputs:cameraPrim",
        #     target_prim_paths=['/World' + self.prefix + "/hand_camera_frame/Camera"],
        # )

        # Run the ROS Camera graph once to generate ROS image publishers in SDGPipeline
        # og.Controller.evaluate_sync(self.ros_camera_graph_l)
        # og.Controller.evaluate_sync(self.ros_camera_graph_r)
        og.Controller.evaluate_sync(self.ros_camera_graph_rgbd)
        # og.Controller.evaluate_sync(self.ros_camera_graph_hand)

    def create_lidar_rtx(self) -> None:
        # rtx based lidar (not working yet)
        lidar_config = "Example_Rotary"
        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path='/World' + self.prefix + '/base_range_sensor_link/Lidar',
            parent=None,
            config=lidar_config,
        )
        _, render_product_path = create_hydra_texture([1, 1], sensor.GetPath().pathString)
        writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
        writer.attach([render_product_path])
        writer = rep.writers.get("RtxLidar" + "ROS1PublishPointCloud")
        writer.attach([render_product_path])

    def create_imu(self) -> None:
        _, sensor = omni.kit.commands.execute(
            'IsaacSensorCreateImuSensor',
            path='/World' + self.prefix + '/base_imu_frame/Imu_Sensor',
            parent=None,
            sensor_period=-1
        )

    def create_force_sensor(self) -> None:
        prim = stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + '/wrist_ft_sensor_frame')
        omni.kit.commands.execute(
            'AddPhysicsComponent',
            usd_prim=prim,
            component='ForceAPI')
        omni.kit.commands.execute(
            'ApplyAPISchema',
            api=PhysxSchema.PhysxForceAPI,
            prim=prim)

    def create_lidar(self) -> None:
        _, sensor = omni.kit.commands.execute(
            'RangeSensorCreateLidar',
            path='/World' + self.prefix + '/base_range_sensor_link/Lidar',
            parent=None,
            min_range=0.3,  # 0.05
            max_range=60.0,
            draw_points=False,
            # draw_lines=True,
            horizontal_fov=240.0,
            horizontal_resolution=1.0,
            rotation_rate=30,
            high_lod=False,
            yaw_offset=0.0,
            enable_semantics=False
        )

        (self.ros_lidar, _, _, _) = og.Controller.edit(
            {
                "graph_path": "/lidar_sensor",
                "evaluator_name": "execution",
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnPlaybackTick"),
                    ("readSimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("readLidarBeams", "isaacsim.sensors.physx.IsaacReadLidarBeams"),
                    ("publishLaserScan", "isaacsim.ros1.bridge.ROS1PublishLaserScan"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "readLidarBeams.inputs:execIn"),
                    ("readLidarBeams.outputs:execOut", "publishLaserScan.inputs:execIn"),
                    ("readSimulationTime.outputs:simulationTime", "publishLaserScan.inputs:timeStamp"),
                    ("readLidarBeams.outputs:horizontalFov", "publishLaserScan.inputs:horizontalFov"),
                    ("readLidarBeams.outputs:horizontalResolution", "publishLaserScan.inputs:horizontalResolution"),
                    ("readLidarBeams.outputs:depthRange", "publishLaserScan.inputs:depthRange"),
                    ("readLidarBeams.outputs:rotationRate", "publishLaserScan.inputs:rotationRate"),
                    ("readLidarBeams.outputs:linearDepthData", "publishLaserScan.inputs:linearDepthData"),
                    ("readLidarBeams.outputs:intensitiesData", "publishLaserScan.inputs:intensitiesData"),
                    ("readLidarBeams.outputs:numRows", "publishLaserScan.inputs:numRows"),
                    ("readLidarBeams.outputs:numCols", "publishLaserScan.inputs:numCols"),
                    ("readLidarBeams.outputs:azimuthRange", "publishLaserScan.inputs:azimuthRange"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("publishLaserScan.inputs:frameId", "base_range_sensor_link"),
                    ("publishLaserScan.inputs:topicName", self.prefix + "/base_scan"),
                ],
            },
        )


        # (self.ros_lidar, _, _, _) = og.Controller.edit(
        #     {
        #         "graph_path": "/lidar_sensor",
        #         "evaluator_name": "execution",
        #     },
        #     {
        #         og.Controller.Keys.CREATE_NODES: [
        #             ("OnTick", "omni.graph.action.OnPlaybackTick"),
        #             ("readSimulationTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
        #             ("readLidarBeams", "omni.isaac.range_sensor.IsaacReadLidarBeams"),
        #             ("publishLaserScan", "omni.isaac.ros_bridge.ROS1PublishLaserScan"),
        #         ],
        #         og.Controller.Keys.CONNECT: [
        #             ("OnTick.outputs:tick", "readLidarBeams.inputs:execIn"),
        #             ("readLidarBeams.outputs:execOut", "publishLaserScan.inputs:execIn"),
        #             ("readSimulationTime.outputs:simulationTime", "publishLaserScan.inputs:timeStamp"),
        #             ("readLidarBeams.outputs:horizontalFov", "publishLaserScan.inputs:horizontalFov"),
        #             ("readLidarBeams.outputs:horizontalResolution", "publishLaserScan.inputs:horizontalResolution"),
        #             ("readLidarBeams.outputs:depthRange", "publishLaserScan.inputs:depthRange"),
        #             ("readLidarBeams.outputs:rotationRate", "publishLaserScan.inputs:rotationRate"),
        #             ("readLidarBeams.outputs:linearDepthData", "publishLaserScan.inputs:linearDepthData"),
        #             ("readLidarBeams.outputs:intensitiesData", "publishLaserScan.inputs:intensitiesData"),
        #             ("readLidarBeams.outputs:numRows", "publishLaserScan.inputs:numRows"),
        #             ("readLidarBeams.outputs:numCols", "publishLaserScan.inputs:numCols"),
        #             ("readLidarBeams.outputs:azimuthRange", "publishLaserScan.inputs:azimuthRange"),
        #         ],
        #         og.Controller.Keys.SET_VALUES: [
        #             ("publishLaserScan.inputs:frameId", "base_range_sensor_link"),
        #             ("publishLaserScan.inputs:topicName", self.prefix + "/base_scan"),
        #         ],
        #     },
        # )

        set_targets(
            prim=stage.get_current_stage().GetPrimAtPath("/lidar_sensor/readLidarBeams"),
            attribute="inputs:lidarPrim",
            target_prim_paths=['/World' + self.prefix + "/base_range_sensor_link/Lidar"],
        )

    def set_base_joint_and_material(self) -> None:
        caster_material = PhysicsMaterial(
            prim_path='/Caster',
            static_friction=0.0,
            dynamic_friction=0.0)
        

        for l in ['/base_l_passive_wheel_z_link/collisions', '/base_r_passive_wheel_z_link/collisions']:
            omni.kit.commands.execute('BindMaterialExt',
                                      material_path='/Caster',
                                      prim_path=['/World' + self.prefix + l],
                                      strength=['weakerThanDescendants'],
                                      material_purpose='physics')

        tire_material = PhysicsMaterial(
            prim_path='/Tire',
            static_friction=100.0,
            dynamic_friction=100.0)
        

        for l in ['/base_l_drive_wheel_link/collisions', '/base_r_drive_wheel_link/collisions']:
            omni.kit.commands.execute('BindMaterialExt',
                                      material_path='/Tire',
                                      prim_path=['/World' + self.prefix + l],
                                      strength=['weakerThanDescendants'],
                                      material_purpose='physics')

        left_passive1_drive = UsdPhysics.DriveAPI.Get(stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + "/base_l_passive_wheel_x_frame/base_l_passive_wheel_y_frame_joint"), "angular")
        left_passive1_drive.GetDampingAttr().Set(0)
        left_passive1_drive.GetStiffnessAttr().Set(0)
        left_passive2_drive = UsdPhysics.DriveAPI.Get(stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + "/base_l_passive_wheel_y_frame/base_l_passive_wheel_z_joint"), "angular")
        left_passive2_drive.GetDampingAttr().Set(0)
        left_passive2_drive.GetStiffnessAttr().Set(0)
        right_passive1_drive = UsdPhysics.DriveAPI.Get(stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + "/base_r_passive_wheel_x_frame/base_r_passive_wheel_y_frame_joint"), "angular")
        right_passive1_drive.GetDampingAttr().Set(0)
        right_passive1_drive.GetStiffnessAttr().Set(0)
        right_passive2_drive = UsdPhysics.DriveAPI.Get(stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + "/base_r_passive_wheel_y_frame/base_r_passive_wheel_z_joint"), "angular")
        right_passive2_drive.GetDampingAttr().Set(0)
        right_passive2_drive.GetStiffnessAttr().Set(0)

        # Get handle to the Drive API for both wheels
        left_wheel_drive = UsdPhysics.DriveAPI.Get(stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + "/base_roll_link/base_l_drive_wheel_joint"), "angular")
        right_wheel_drive = UsdPhysics.DriveAPI.Get(stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + "/base_roll_link/base_r_drive_wheel_joint"), "angular")
        roll_drive = UsdPhysics.DriveAPI.Get(stage.get_current_stage().GetPrimAtPath('/World' + self.prefix + "/base_link/base_roll_joint"), "angular")

        # Set the drive damping, which controls the strength of the velocity drive
        left_wheel_drive.GetDampingAttr().Set(15000)
        right_wheel_drive.GetDampingAttr().Set(15000)
        roll_drive.GetDampingAttr().Set(15000)

        # Set the drive stiffness, which controls the strength of the position drive
        # In this case because we want to do velocity control this should be set to zero
        left_wheel_drive.GetStiffnessAttr().Set(0)
        right_wheel_drive.GetStiffnessAttr().Set(0)
        roll_drive.GetStiffnessAttr().Set(0)

    def on_cmd_vel(self, msg):
        if self.simulation_context is not None:
            self.cmd_vel_msg = msg
            self.last_cmd_vel_time = self.simulation_context.current_time

    def on_pose_estimate_pose(self, msg):
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose = msg.pose
        odom.pose.covariance = [
            0.001, 0, 0, 0, 0, 0,
            0, 0.001, 0, 0, 0, 0,
            0, 0, 100000.0, 0, 0, 0,
            0, 0, 0, 100000.0, 0, 0,
            0, 0, 0, 0, 100000.0, 0,
            0, 0, 0, 0, 0, 1000.0,
        ]
        if self.imu_reading:
            odom.twist.twist.linear.x = self.imu_reading.lin_acc_x  # TODO: convert to velocity
            odom.twist.twist.linear.y = self.imu_reading.lin_acc_y
            odom.twist.twist.linear.z = self.imu_reading.lin_acc_z
            odom.twist.twist.angular.x = self.imu_reading.ang_vel_x
            odom.twist.twist.angular.y = self.imu_reading.ang_vel_y
            odom.twist.twist.angular.z = self.imu_reading.ang_vel_z
        odom.twist.covariance = [
            0.001, 0, 0, 0, 0, 0,
            0, 0.001, 0, 0, 0, 0,
            0, 0, 100000.0, 0, 0, 0,
            0, 0, 0, 100000.0, 0, 0,
            0, 0, 0, 0, 100000.0, 0,
            0, 0, 0, 0, 0, 1000.0,
        ]
        self.laser_odom_pub.publish(odom)

        self.odometry_.x = msg.pose.pose.position.x
        self.odometry_.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.odometry_.ang = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]

    def on_laserscan_pose(self, msg):
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose = msg.pose
        odom.pose.covariance = [
            0.001, 0, 0, 0, 0, 0,
            0, 0.001, 0, 0, 0, 0,
            0, 0, 100000.0, 0, 0, 0,
            0, 0, 0, 100000.0, 0, 0,
            0, 0, 0, 0, 100000.0, 0,
            0, 0, 0, 0, 0, 1000.0,
        ]
        if self.imu_reading:
            odom.twist.twist.linear.x = self.imu_reading.lin_acc_x  # TODO: convert to velocity
            odom.twist.twist.linear.y = self.imu_reading.lin_acc_y
            odom.twist.twist.linear.z = self.imu_reading.lin_acc_z
            odom.twist.twist.angular.x = self.imu_reading.ang_vel_x
            odom.twist.twist.angular.y = self.imu_reading.ang_vel_y
            odom.twist.twist.angular.z = self.imu_reading.ang_vel_z
        odom.twist.covariance = [
            0.001, 0, 0, 0, 0, 0,
            0, 0.001, 0, 0, 0, 0,
            0, 0, 100000.0, 0, 0, 0,
            0, 0, 0, 100000.0, 0, 0,
            0, 0, 0, 0, 100000.0, 0,
            0, 0, 0, 0, 0, 1000.0,
        ]
        self.laser_odom_pub.publish(odom)

        self.odometry_.x = msg.pose.position.x
        self.odometry_.y = msg.pose.position.y
        q = msg.pose.orientation
        self.odometry_.ang = tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]

    def publish_joint_states(self):
        js = JointState()
        js.header.stamp = rospy.Time(self.simulation_context.current_time)
        js.name = list(self._joints.keys())
        js.position = []
        for n in js.name:
            (joint, joint_type, inv) = self._joints[n]
            st = self.dc.get_dof_state(joint, _dynamic_control.STATE_ALL)
            if joint_type == _dynamic_control.JOINT_PRISMATIC:
                st.pos = st.pos * get_stage_units()
            if inv:
                st.pos = -st.pos
                st.vel = -st.vel
                st.effort = -st.effort
            js.position.append(st.pos)
            js.velocity.append(st.vel)
            js.effort.append(st.effort * 1000.0)
        js.name = js.name + ['odom_x', 'odom_y', 'odom_t']
        js.position = js.position + [self.odometry_.x, self.odometry_.y, self.odometry_.ang]
        js.velocity = js.velocity + [0, 0, 0]
        js.effort = js.effort + [0, 0, 0]
        self.joint_state_pub.publish(js)

    def onsimulationstart(self, simulation_context):
        # print('SAYING HELLO WORLD FROM ONSIMULATIONSTART ROBOCANES HSR!!!')

        self.simulation_context = simulation_context
        self.prev_time = self.simulation_context.current_time

    def step(self):
        # Debugging
        # print(f'hello from step!')

        og.Controller.set(og.Controller.attribute("/ros_controllers/OnImpulseEvent.state:enableImpulse"), True)

        dt = self.simulation_context.current_time - self.prev_time

        if dt == 0: return

        if not self.art:
            self.art = self.dc.get_articulation('/World' + self.prefix)
            if self.art == _dynamic_control.INVALID_HANDLE:
                print("{self.prefix} is not an articulation")
            self._joints = {}
            for i in range(self.dc.get_articulation_dof_count(self.art)):
                dof_ptr = self.dc.get_articulation_dof(self.art, i)
                if dof_ptr != _dynamic_control.DofType.DOF_NONE:
                    dof_name = self.dc.get_dof_name(dof_ptr)
                    joint = self.dc.find_articulation_joint(self.art, dof_name)
                    self._joints[dof_name] = (dof_ptr, self.dc.get_joint_type(joint), dof_name in ['arm_flex_joint', 'arm_lift_joint', 'wrist_flex_joint', 'arm_roll_joint'])
            print(self._joints)
            self.left_wheel_ptr = self.dc.find_articulation_dof(self.art, "base_l_drive_wheel_joint")
            self.right_wheel_ptr = self.dc.find_articulation_dof(self.art, "base_r_drive_wheel_joint")
            self.roll_ptr = self.dc.find_articulation_dof(self.art, "base_roll_joint")

        self.dc.wake_up_articulation(self.art)

        self.publish_joint_states()

        left_state = self.dc.get_dof_state(self.left_wheel_ptr, _dynamic_control.STATE_ALL)
        right_state = self.dc.get_dof_state(self.right_wheel_ptr, _dynamic_control.STATE_ALL)
        roll_state = self.dc.get_dof_state(self.roll_ptr, _dynamic_control.STATE_ALL)

        state_ = VehicleState()
        state_.steer_angle = roll_state.pos
        joint_param_ = JointSpace()
        joint_param_.vel_wheel_l = left_state.vel
        joint_param_.vel_wheel_r = right_state.vel
        joint_param_.vel_steer = roll_state.vel

        # Calculate cartesian space velocities by using forward dynamics equations
        cartesian_param_ = forward_dynamics(joint_param_, state_)

        # Integrate velocities to update wheel odometry
        diff_r = cartesian_param_.dot_r * dt
        cosr = math.cos(self.odometry_.ang + 0.5 * diff_r)  # use Runge-Kutta 2nd
        sinr = math.sin(self.odometry_.ang + 0.5 * diff_r)
        abs_dot_x = cartesian_param_.dot_x * cosr - cartesian_param_.dot_y * sinr
        abs_dot_y = cartesian_param_.dot_x * sinr + cartesian_param_.dot_y * cosr
        self.odometry_.x += abs_dot_x * dt
        self.odometry_.y += abs_dot_y * dt
        self.odometry_.ang += diff_r

        odom = Odometry()
        odom.header.stamp = rospy.Time(self.simulation_context.current_time)
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = self.odometry_.x
        odom.pose.pose.position.y = self.odometry_.y
        odom.pose.pose.position.z = 0
        q = tf.transformations.quaternion_from_euler(0, 0, self.odometry_.ang)
        odom.pose.pose.orientation = Quaternion(*q)
        odom.pose.covariance = [
            0.001, 0, 0, 0, 0, 0,
            0, 0.001, 0, 0, 0, 0,
            0, 0, 100000.0, 0, 0, 0,
            0, 0, 0, 100000.0, 0, 0,
            0, 0, 0, 0, 100000.0, 0,
            0, 0, 0, 0, 0, 1000.0,
        ]
        odom.twist.twist.linear.x = abs_dot_x
        odom.twist.twist.linear.y = abs_dot_y
        odom.twist.twist.linear.z = 0
        odom.twist.twist.angular.x = 0
        odom.twist.twist.angular.y = 0
        odom.twist.twist.angular.z = cartesian_param_.dot_r

        # self.acc_odom_angular_z += odom.twist.twist.angular.z

        # print(f'self.acc_odom_angular_z:', self.acc_odom_angular_z)

        odom.twist.covariance = [
            0.001, 0, 0, 0, 0, 0,
            0, 0.001, 0, 0, 0, 0,
            0, 0, 100000.0, 0, 0, 0,
            0, 0, 0, 100000.0, 0, 0,
            0, 0, 0, 0, 100000.0, 0,
            0, 0, 0, 0, 0, 1000.0,
        ]
        self.base_odom_pub.publish(odom)

        cmd = CartSpace()
        if self.odom_trajectory_action_server._action_goal is not None:
            self.odom_trajectory_action_server._odometry = self.odometry_
            cmd.dot_x = self.odom_trajectory_action_server._joints['odom_x'] - self.odometry_.x
            cmd.dot_y = self.odom_trajectory_action_server._joints['odom_y'] - self.odometry_.y
            cmd.dot_r = self.odom_trajectory_action_server._joints['odom_t'] - self.odometry_.ang
        elif self.last_cmd_vel_time + 2.0 > self.simulation_context.current_time and self.cmd_vel_msg is not None:
            ang = self.odometry_.ang + 0.5 * self.cmd_vel_msg.angular.z * dt
            cosr = math.cos(ang)
            sinr = math.sin(ang)
            cmd.dot_x = self.cmd_vel_msg.linear.x * cosr - self.cmd_vel_msg.linear.y * sinr
            cmd.dot_y = self.cmd_vel_msg.linear.x * sinr + self.cmd_vel_msg.linear.y * cosr
            cmd.dot_r = self.cmd_vel_msg.angular.z

        relcmd = CartSpace()
        diff_r = cmd.dot_r * dt
        ang = self.odometry_.ang + 0.5 * diff_r  # use Runge-Kutta 2nd
        cosr = math.cos(-ang)
        sinr = math.sin(-ang)
        relcmd.dot_x = cmd.dot_x * cosr - cmd.dot_y * sinr
        relcmd.dot_y = cmd.dot_x * sinr + cmd.dot_y * cosr
        relcmd.dot_r = diff_r / dt

        jcmd = inverse_dynamics(relcmd, state_)

        # apply velocity limits
        ratio = abs(jcmd.vel_steer) / self.vel_limit_steer_
        ratio = max(ratio, abs(jcmd.vel_wheel_l) / self.vel_limit_wheel_)
        ratio = max(ratio, abs(jcmd.vel_wheel_r) / self.vel_limit_wheel_)
        if ratio > 1.0:
            jcmd.vel_steer /= ratio
            jcmd.vel_wheel_l /= ratio
            jcmd.vel_wheel_r /= ratio

        self.dc.set_dof_velocity_target(self.left_wheel_ptr, jcmd.vel_wheel_l)
        self.dc.set_dof_velocity_target(self.right_wheel_ptr, jcmd.vel_wheel_r)
        self.dc.set_dof_velocity_target(self.roll_ptr, jcmd.vel_steer)

        # self.imu_reading = self._if_imu.get_sensor_sim_reading('/World' + self.prefix + '/base_imu_frame/Imu_Sensor')
        self.imu_reading = self._if_imu.get_sensor_reading('/World' + self.prefix + '/base_imu_frame/Imu_Sensor')

        imu = Imu()
        imu.header.stamp = rospy.Time(self.simulation_context.current_time)
        imu.header.frame_id = 'base_imu_frame'
        imu.orientation.x = self.imu_reading.orientation[0]
        imu.orientation.y = self.imu_reading.orientation[1]
        imu.orientation.z = self.imu_reading.orientation[2]
        imu.orientation.w = self.imu_reading.orientation[3]
        imu.angular_velocity.x = self.imu_reading.ang_vel_x
        imu.angular_velocity.y = self.imu_reading.ang_vel_y
        imu.angular_velocity.z = self.imu_reading.ang_vel_z
        imu.linear_acceleration.x = self.imu_reading.lin_acc_x
        imu.linear_acceleration.y = self.imu_reading.lin_acc_y
        imu.linear_acceleration.z = self.imu_reading.lin_acc_z
        self.imu_pub.publish(imu)

        #force_readings = self.robots._physics_view.get_force_sensor_forces()
        #print(force_readings)

        self.arm_trajectory_action_server.step(dt=dt)
        self.head_trajectory_action_server.step(dt=dt)
        self.odom_trajectory_action_server.step(dt=dt)
        self.gripper_trajectory_action_server.step(dt=dt)
        self.gripper_apply_force_action_server.step(dt=dt)
        self.gripper_command_action_server.step(dt=dt)

        self.prev_time = self.simulation_context.current_time
