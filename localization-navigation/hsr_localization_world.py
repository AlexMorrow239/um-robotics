import argparse
import ast
import sys
import select
import time

import os
import numpy as np

from rospkg import RosPack; rp = RosPack()
repo_path = rp.get_path('hsr_isaac_localization')
usd_repo_path = rp.get_path('usd')
template_repo_path = rp.get_path('hsr-omniverse')
print(f'template_repo_path:', template_repo_path)

sys.path.append(template_repo_path)

import robocanes_hsr

from robocanes_hsr_isaac_world import robocanes_hsr_isaac_world, sim_app, Usd, Gf, omni

print(f'sim_app:', sim_app)

class hsr_localization_isaac_world(robocanes_hsr_isaac_world):
    def __init__(self):
        super().__init__()

    # Decomposition and world transform obtained from https://forums.developer.nvidia.com/t/get-euler-angles-rotation-of-a-prim/275600
    def decompose_matrix(self, mat: Gf.Matrix4d):
        reversed_ident_mtx = reversed(Gf.Matrix3d())
        translate: Gf.Vec3d = mat.ExtractTranslation()
        scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in mat.ExtractRotationMatrix()))

        mat.Orthonormalize()
        rotate = Gf.Vec3d(*reversed(mat.ExtractRotation().Decompose(*reversed_ident_mtx)))

        return np.array(translate), np.array(rotate), np.array(scale)

    def get_world_transform_xform(self, prim: Usd.Prim):
        world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
        decompose_matrix = self.decompose_matrix(world_transform)
        return decompose_matrix
   
if __name__ == '__main__':
    world = hsr_localization_isaac_world()

    sim_app_running = sim_app.is_running()
    print(f'sim_app_running:', sim_app_running)

    sim_world_onetime_trigger = True

    while sim_app.is_running():
        world.sim_world.step(render=True)
        world.hsr_instance.step()

        world.sim_world.play()

        if world.sim_world.is_playing():
            if sim_world_onetime_trigger:
                print(f'ISAAC WORLD ONETIME PLAY!!!')
                sim_world_onetime_trigger = False

        world.isaac_robot_behavior_start.start()

        chosen_prim = world.stage.GetPrimAtPath('/World/hsrb/base_footprint')
        world_translate, world_rotate, world_scale = world.get_world_transform_xform(prim=chosen_prim)

        prim_world_pose = np.array([world_translate[0], world_translate[1], np.radians(world_rotate[2])])
        world.isaac_robot_pose_pub.publish(pose=prim_world_pose)


    sim_app.close()

# import ast
# import argparse
# import inspect
# import json
# import os
# import math
# import time
# import numpy as np

# from rospkg import RosPack; rp = RosPack()
# repo_path = rp.get_path('hsr_isaac_localization')
# usd_repo_path = rp.get_path('usd')

# import omni
# from omni.isaac.kit import SimulationApp

# parser = argparse.ArgumentParser(description="HSR Localization World Setup")
# # parser.add_argument("--config", type=str, default=os.path.join(repo_path, 'config', 'isaac_spawn', '0.json'), help="Configuration file name in hsr_isaac_localization/config/isaac_spawn/")
# parser.add_argument("--robot_spawn_pos_xyz", type=str, default='[0, 0, 0.01]', help="Configuration robot spawn position set as xyz list.")
# parser.add_argument("--robot_spawn_orient_xyz", type=str, default='[0, 0, 0]', help="Configuration robot spawn orientation set as xyz list.")
# args = parser.parse_args()

# sim_app_config = {
#     "width": 1280, 
#     "height": 720, 
#     "sync_loads": True, 
#     "headless": False, 
#     "renderer": "RayTracedLighting"
# }

# sim_app = SimulationApp(sim_app_config)
# sim_app.set_setting("/app/extensions/installUntrustedExtensions", True)
# # sim_app.update()

# import rosgraph

# if not rosgraph.is_master_online():
#     print(f'Please run roscore before executing this script!')
#     sim_app.close()
#     exit()

# from omni.isaac.core import World

# sim_world = World(stage_units_in_meters=1.0)

# from omni.isaac.core.utils import prims, viewports
# from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
# from omni.physx.scripts import utils
# from pxr import PhysxSchema, Gf, UsdGeom, Usd

# usd_context = omni.usd.get_context()
# stage = usd_context.get_stage()

# # print(f'usd_context: {usd_context}')
# # print(f'stage: {stage}')

# import robocanes_hsr

# from scripts.isaac_robot_behavior_start import isaac_robot_behavior_start
# from scripts.isaac_robot_pose_pub import isaac_robot_pose_pub

# class hsr_localization_world:

#     def add_sun(self):
#         dome_light_prim_path = f"/World/Sun"
#         dome_light_position = [0, 0, 0]
#         dome_light_euler = [0, 0, 0]

#         dome_light_prim = prims.create_prim(
#             prim_path=dome_light_prim_path,

#             prim_type="DomeLight",
#             position=np.array(dome_light_position),
#             orientation=np.array(euler_angles_to_quat(dome_light_euler)),
#             attributes={
#                 "inputs:enableColorTemperature": True,
#                 "inputs:colorTemperature": 6500,
#                 "inputs:intensity": 6500
#             }
#         )

#         # Use this to view all available attributes for a prim type.
#         # print(dome_light_prim.GetAttributes())

#     def add_ground_plane(self):
#         # Custom ground plane
#         sim_world.scene.add_ground_plane(
#             size=1000,
#             z_position=0.0,
#             color=np.array([1, 1, 1]))
        
#         sim_app.update()

#     def add_robocanes_lab_lights_uyen(self):
#         # robocanes_lights_usd_path = "/home/cduarte/gitwork/hsr_robocanes_omniverse/usd/robocanes_lab/Collected_stage/ceiling_lights.usd"
#         robocanes_lights_usd_path = os.path.join(usd_repo_path, 'robocanes_lab', 'Collected_stage', 'ceiling_lights.usd')

#         robocanes_lights_name = f'{self.robocanes_lab_name}_lights'
#         robocanes_lights_prim_path = f'/World/ceiling_lights'
#         robocanes_lights_pos = [0, 0, 0.0]
#         robocanes_lights_euler = [0, 0, 0]
#         robocanes_lights_scale = [1, 1, 1]

#         robocanes_lights_prim = prims.create_prim(
#             prim_path=robocanes_lights_prim_path,
#             usd_path=robocanes_lights_usd_path,
#             translation=robocanes_lights_pos,
#             orientation=euler_angles_to_quat(robocanes_lights_euler),
#             scale=robocanes_lights_scale,
#             semantic_label='robocanes_lab'
#         )

#         # Fix light placement from default.
#         light_children_prim = robocanes_lights_prim.GetAllChildren()[0].GetAllChildren()

#         for light_child_prim in light_children_prim:
#             # print(f'light_child_prim: {light_child_prim}')

#             # Get current translation of each light prim
#             translation_attribute = light_child_prim.GetAttribute("xformOp:translate")
#             translation = translation_attribute.Get()
#             # print(f'translation: {translation}')

#             # Set each light prim to the appropriate location (90 degree clockwise rotation from default).
#             new_translation = Gf.Vec3d(translation[0], -translation[2], translation[1])
#             success = translation_attribute.Set(new_translation, 0)
#             # print(f'New Translation Set: {success}')

#         # Disable env light (causes strange rendering issues).
#         env_light_prim_path = f'/World/robocanes_lab/env_light'
#         env_light_prim = stage.GetPrimAtPath(env_light_prim_path)
#         env_light_visibility_attribute = env_light_prim.GetAttribute("visibility")
#         env_light_visibility_attribute.Set("invisible")

#         # To set visible.
#         # env_light_visibility_attribute.Set("inherited")

#     def add_robocanes_lab_lights(self):
#         robocanes_lab_lights_pos = [5, 0, 2.7]

#         robocanes_lab_lights_prim = prims.create_prim(
#             prim_path=f'{str(self.robocanes_lab_prim.GetPrimPath())}/{self.robocanes_lab_name}_lights',
#             translation=robocanes_lab_lights_pos,
#         )

#         config_lights = self.world_config_json['lights']

#         enable_rect_lights = config_lights['rect_lights']
#         enable_disk_lights = config_lights['disk_lights']

#         print(f'enable_rect_lights: {enable_rect_lights}')
#         print(f'enable_disk_lights: {enable_disk_lights}')

#         if enable_rect_lights:
#             rect_lights_prim = prims.create_prim(
#                 prim_path=f'{str(robocanes_lab_lights_prim.GetPrimPath())}/rect_lights',
#             )

#             rect_light_positions = [
#                 [0.75, 3.19, robocanes_lab_lights_pos[2]],
#                 [0.75, 4.79, robocanes_lab_lights_pos[2]],
#                 [2.76, 3.19, robocanes_lab_lights_pos[2]],
#                 [2.76, 4.79, robocanes_lab_lights_pos[2]],
#                 [5.28, 3.19, robocanes_lab_lights_pos[2]],
#                 [5.28, 4.79, robocanes_lab_lights_pos[2]],
#                 [7.61, 3.19, robocanes_lab_lights_pos[2]],
#             ]

#             rect_light_euler= [0, 0, math.pi/2]

#             # print(f'light_positions: {light_positions}')

#             for i,pos in enumerate(rect_light_positions):
#                 # print('rect_light_elem:', (i, pos))

#                 rect_light_prim = prims.create_prim(
#                     # Use this prim path if you need to move the lights around with respect to origin.
#                     # prim_path=f"/World/{robocanes_lab_name}/rect_light_{i}",
                    
#                     # Use this only after moving lights around to desired positions.
#                     prim_path=f"{str(rect_lights_prim.GetPrimPath())}/rect_light_{i}",

#                     prim_type="RectLight",
#                     position=np.array(pos),
#                     orientation=np.array(euler_angles_to_quat(rect_light_euler)),
#                     attributes={
#                         "inputs:height": 1.1,
#                         "inputs:width": 0.3,
#                         "inputs:intensity": 5e3,
#                         "inputs:color": (1.0, 1.0, 0.9)
#                     }
#                 )
#         if enable_disk_lights:
#             disk_lights_prim = prims.create_prim(
#                 prim_path=f'{str(robocanes_lab_lights_prim.GetPrimPath())}/disk_lights',
#             )

#             disk_light_positions = [
#                 [1, 0, robocanes_lab_lights_pos[2]],
#                 [5, 0, robocanes_lab_lights_pos[2]],
#                 [7, 0, robocanes_lab_lights_pos[2]],
#                 [1, 2.37, robocanes_lab_lights_pos[2]],
#                 [3, 2.37, robocanes_lab_lights_pos[2]],
#                 [5, 2.37, robocanes_lab_lights_pos[2]],
#                 [7, 2.37, robocanes_lab_lights_pos[2]],
#                 [1, 4.74, robocanes_lab_lights_pos[2]],
#                 [3, 4.74, robocanes_lab_lights_pos[2]],
#                 [5, 4.74, robocanes_lab_lights_pos[2]]
#             ]

#             disk_light_euler = [0, 0, math.pi/2]

#             for i,pos in enumerate(disk_light_positions):
#                 # print('rect_light_elem:', (i, pos))

#                 disk_light_prim = prims.create_prim(
#                     # Use this prim path if you need to move the lights around with respect to origin.
#                     # prim_path=f"/World/{robocanes_lab_name}/disk_light_{i}",
                    
#                     # Use this only after moving lights around to desired positions.
#                     prim_path=f"{str(disk_lights_prim.GetPrimPath())}/disk_light_{i}",

#                     prim_type="DiskLight",
#                     position=np.array(pos),
#                     orientation=np.array(euler_angles_to_quat(disk_light_euler)),
#                     attributes={
#                         "inputs:radius": 0.2,
#                         "inputs:intensity": 13e3,
#                         "inputs:color": (1.0, 0.755, 0.367)
#                     }
#                 )

#     def add_robocanes_lab_uyen(self):
#         # robocanes_lab_usd_path = "/home/cduarte/gitwork/hsr_robocanes_omniverse/usd/robocanes_lab/Collected_stage/robocanes_lab_model.usdc"
#         robocanes_lab_usd_path = os.path.join(usd_repo_path, 'robocanes_lab', 'Collected_stage', 'robocanes_lab_model.usdc')

#         self.robocanes_lab_name = 'robocanes_lab'
#         robocanes_lab_prim_path = f'/World/{self.robocanes_lab_name}'
#         robocanes_lab_pos = [0, 0, 0.0]
#         robocanes_lab_euler = [0, 0, 0]
#         robocanes_lab_scale = [1, 1, 1]

#         self.robocanes_lab_prim = prims.create_prim(
#             prim_path=robocanes_lab_prim_path,
#             usd_path=robocanes_lab_usd_path,
#             translation=robocanes_lab_pos,
#             orientation=euler_angles_to_quat(robocanes_lab_euler),
#             scale=robocanes_lab_scale,
#             semantic_label='robocanes_lab'
#         )

#         # utils.setCollider(prim=self.robocanes_lab_prim, approximationShape='sdfMesh')

#         self.add_robocanes_lab_lights_uyen()

#     def add_robocanes_lab(self):
#         # RoboCanes Lab
#         robocanes_lab_usd_path = os.path.join(usd_repo_path, 'robocanes_lab', 'robocanes_lab.usdc')

#         self.robocanes_lab_name = 'robocanes_lab'
#         robocanes_lab_prim_path = f'/World/{self.robocanes_lab_name}'
#         robocanes_lab_pos = [0, 0, 0.0]
#         robocanes_lab_euler = [0, 0, 0]
#         robocanes_lab_scale = [1, 1, 1]

#         self.robocanes_lab_prim = prims.create_prim(
#             prim_path=robocanes_lab_prim_path,
#             usd_path=robocanes_lab_usd_path,
#             translation=robocanes_lab_pos,
#             orientation=euler_angles_to_quat(robocanes_lab_euler),
#             scale=robocanes_lab_scale,
#             semantic_label='robocanes_lab'
#         )

#         # This approach works for a simple mesh approximation using defaults 

#         # utils.setCollider(prim=self.robocanes_lab_prim, approximationShape='convexHull')
#         utils.setCollider(prim=self.robocanes_lab_prim, approximationShape='sdfMesh')
#         # print(f'inspect output: {inspect.signature(utils.setCollider)}')

#         self.add_robocanes_lab_lights()

#     def delete_hsr(self):
#        print(f'in delete hsr!')
#        prims.delete_prim(prim_path="/World/hsrb")

#        prims.delete_prim(prim_path="/ros_controllers")
#        prims.delete_prim(prim_path="/head_rgbd_camera")
#        prims.delete_prim(prim_path="/lidar_sensor")

#     def add_hsr(self, robot_name, robot_position, robot_orientation):
#         self.hsr_instance = robocanes_hsr.hsr(
#             prefix=f'/{robot_name}',
#             spawn_config={
#                 'translation': robot_position,
#                 'orientation': robot_orientation,
#                 'scale': [1, 1, 1]
#             }
#         )

#         self.hsr_instance.onsimulationstart(sim_world)

#         self.isaac_robot_behavior_start = isaac_robot_behavior_start()
#         self.isaac_robot_pose_pub = isaac_robot_pose_pub()

#     def read_json_file(self, data_file_path):
#         if os.path.isfile(data_file_path):
#             with open(data_file_path, mode='r', encoding='utf-8') as f:
#                 data = json.load(f)
#             return data

#     def add_qr_codes(self):
#         # qr_objects_file_path = os.path.join(repo_path, 'config', 'localization_objects', 'isaac_lab_qr_objects_0.json')
#         qr_objects_file_path = os.path.join(repo_path, 'config', 'localization_objects', 'isaac_lab_qr_objects_1.json')

#         self.qr_objects_dict = self.read_json_file(data_file_path=qr_objects_file_path)['objects']
        
#         qr_usd_dir_path = os.path.join(repo_path, 'vision', 'qr_usd')

#         for k,v in self.qr_objects_dict.items():
#             qr_usd_path = os.path.join(qr_usd_dir_path, f'{k}.usd')

#             qr_name = k
#             qr_prim_path = f'/World/{k}'
#             qr_pos = v['position']
#             qr_euler = v['orientation']
#             qr_scale = [0.01, 0.01, 0.01]

#             qr_prim = prims.create_prim(
#                 prim_path=qr_prim_path,
#                 usd_path=qr_usd_path,
#                 translation=qr_pos,
#                 orientation=euler_angles_to_quat(qr_euler),
#                 scale=qr_scale,
#                 semantic_label=qr_name
#             )

#         # # QR Code 0
#         # qr_usd_path = os.path.join(qr_usd_dir_path, 'qr_0.usd')

#         # qr_name = 'qr_0'
#         # # qr_prim_path = f'{str(self.robocanes_lab_prim.GetPrimPath())}/{qr_name}'
#         # qr_prim_path = f'/World/{qr_name}'
#         # qr_pos = [1, 0, 1]
#         # qr_euler = [0, 0, 1.54]
#         # qr_scale = [0.01, 0.01, 0.01]

#         # qr_prim = prims.create_prim(
#         #     prim_path=qr_prim_path,
#         #     usd_path=qr_usd_path,
#         #     translation=qr_pos,
#         #     orientation=euler_angles_to_quat(qr_euler),
#         #     scale=qr_scale,
#         #     semantic_label=qr_name
#         # )

#         # # QR Code 1
#         # qr_usd_path = os.path.join(qr_usd_dir_path, 'qr_1.usd')

#         # qr_name = 'qr_1'
#         # # qr_prim_path = f'{str(self.robocanes_lab_prim.GetPrimPath())}/{qr_name}'
#         # qr_prim_path = f'/World/{qr_name}'
#         # qr_pos = [1, 0.5, 1]
#         # qr_euler = [0, 0, 1.54]
#         # qr_scale = [0.01, 0.01, 0.01]

#         # qr_prim = prims.create_prim(
#         #     prim_path=qr_prim_path,
#         #     usd_path=qr_usd_path,
#         #     translation=qr_pos,
#         #     orientation=euler_angles_to_quat(qr_euler),
#         #     scale=qr_scale,
#         #     semantic_label=qr_name
#         # )


#     def modify_viewport_camera(self):
#         # Modify eye target of perspective camera
#         viewports.set_camera_view(
#             eye=np.array([4.0, -0.25, 2.65]), 
#             target=np.array([2.0, 2.0, 0.5]))

#         # Modify focal length of perspective camera
#         prim = stage.GetPrimAtPath('/OmniverseKit_Persp')
#         print(f'camera prim: {prim}')

#         prim.GetAttribute('horizontalAperture').Set(43.4)

#     # Decomposition and world transform obtained from https://forums.developer.nvidia.com/t/get-euler-angles-rotation-of-a-prim/275600
#     def decompose_matrix(self, mat: Gf.Matrix4d):
#         reversed_ident_mtx = reversed(Gf.Matrix3d())
#         translate: Gf.Vec3d = mat.ExtractTranslation()
#         scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in mat.ExtractRotationMatrix()))

#         mat.Orthonormalize()
#         rotate = Gf.Vec3d(*reversed(mat.ExtractRotation().Decompose(*reversed_ident_mtx)))

#         return np.array(translate), np.array(rotate), np.array(scale)
    
#     def get_world_transform_xform(self, prim: Usd.Prim):
#         world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
#         decompose_matrix = self.decompose_matrix(world_transform)
#         return decompose_matrix

#     def get_local_transform_xform(self, prim: Usd.Prim):
#         xform = UsdGeom.Xformable(prim)
#         local_transform: Gf.Matrix4d = xform.GetLocalTransformation()

#         decompose_matrix = self.decompose_matrix(local_transform)
#         return decompose_matrix

#     # def get_prim_transform(self, prim_path):
#     #     current_prim = stage.GetPrimAtPath(prim_path)

#     #     print(f'current_prim: {current_prim}')

#     #     # Note: if in docs, you see references to omni.usd.utils it is old. Newer references are simply omni.usd
#     #     prim_pose = omni.usd.get_world_transform_matrix(current_prim)

#     #     print(f'Matrix Form: {prim_pose}')
#     #     print(f'Translation: {prim_pose.ExtractTranslation()}')

#     #     quat = prim_pose.ExtractRotation().GetQuaternion()
#     #     print(f'Rotation: {quat}')

#     #     euler = quat_to_euler_angles(quat)
#     #     print(f'Euler: {euler}')

#     def generate_world(self):
#         # self.add_sun()
#         # self.add_ground_plane()
#         # self.add_robocanes_lab()
#         self.add_robocanes_lab_uyen()

#         # Don't modify robot name until the below note is implemented in it entirety.
#         self.add_hsr(robot_name='hsrb', robot_position=self.world_config_json['robot']['position'], robot_orientation=self.world_config_json['robot']['orientation'])

#         # Note: you can generate multiple hsr robots in the lab for testing (you'll need it for reinforcement learning), but you'll need to seperate the functionalities to work in parallel. Right now, resources are accessed based on one hsr being referred to (e.g. hsrb) for multiple you need to seperate the resources (e.g. head_camera_0, head_camera_1, etc) and the corresponding ros resources for the launch file as well.
#         # self.add_hsr(robot_name='hello', robot_position=self.world_config_json['robot']['position'], robot_orientation=self.world_config_json['robot']['orientation'])
#         # self.add_hsr(robot_name='world', robot_position=[1, 0, 0.01], robot_orientation=self.world_config_json['robot']['orientation'])
#         # self.add_hsr(robot_name='to', robot_position=[1, 1, 0.01], robot_orientation=self.world_config_json['robot']['orientation'])
#         # self.add_hsr(robot_name='you', robot_position=[2, 2, 0.01], robot_orientation=self.world_config_json['robot']['orientation'])

#         # self.add_qr_codes()

#         self.modify_viewport_camera()

#     def __init__(self):
#         self.world_config_json = {
#             'robot': {
#                 'position': [0, 0, 0.01],
#                 'orientation': [0, 0, 0],
#             }
#         }

#         print(f'worlds_config_json:', self.world_config_json)

#         # This function assumes we are constructing the world from scratch building up pre-built elements.
#         self.generate_world()

# if __name__ == '__main__':
#     world = hsr_localization_world()

#     sim_app_running = sim_app.is_running()

#     print(f'sim_app is_running: {sim_app_running}')

#     sim_world_onetime_step_trigger = True
#     world_start_time = None
#     world_start_elapsed_time = None

#     sim_world_onetime_trigger = True

#     world_start_play_time = None
#     world_start_play_elapsed_time = None

#     has_clicked_play = False

#     while sim_app.is_running():
#         sim_world.step(render=True)
#         world.hsr_instance.step()
#         # world.isaac_robot_behavior_start.step()

#         if sim_world_onetime_step_trigger:
#             print(f'ISAAC WORLD FIRST STEP PLAYING!!!')

#             world_start_time = time.time()

#             sim_world_onetime_step_trigger = False

#         # world_start_elapsed_time = time.time() - world_start_time
#         # if world_start_elapsed_time < 3:
#         #     print(f'world_start_elapsed_time:', world_start_elapsed_time)
#         #     continue

#         sim_world.play()

#         if sim_world.is_playing():
#             has_clicked_play = True

#             # One time
#             if sim_world_onetime_trigger:
#                 print(f'ISAAC WORLD ONETIME PLAYING!!!')
                
#                 sim_world_onetime_trigger = False

#                 world_start_play_time = time.time()

#             # Continuous
#             # print(f'ISAAC WORLD CONTINUOUS PLAYING!!!')

#             world_start_elapsed_time = time.time() - world_start_play_time

#             # This additional time is needed, otherwise rviz will have tf transform errors.
#             # if world_start_elapsed_time < 5:
#             #     # print(f'Skipping current play as elapsed time not met. Current elapsed time: {elapsed_time}')
#             #     continue

#             world.isaac_robot_behavior_start.start()

#             chosen_prim = stage.GetPrimAtPath('/World/hsrb/base_footprint')
#             world_translate, world_rotate, world_scale = world.get_world_transform_xform(prim=chosen_prim)
            
#             # print(f'WORLD_TRANSLATE: {world_translate}')
#             # print(f'WORLD_TRANSLATE type: {type(world_translate)}')
#             # print(f'WORLD_ROTATE: {world_rotate}')
#             # print(f'WORLD_SCALE: {world_scale}')

#             prim_world_pose = np.array([world_translate[0], world_translate[1], np.radians(world_rotate[2])])
#             world.isaac_robot_pose_pub.publish(pose=prim_world_pose)
            
#             # local_translate, local_rotate, local_scale = world.get_local_transform_xform(prim=chosen_prim)
#             # print(f'LOCAL_TRANSLATE: {local_translate}')
#             # print(f'LOCAL_ROTATE: {local_rotate}')
#             # print(f'LOCAL_SCALE: {local_scale}')

#             # local_transform = world.get_local_transform_xform(prim=chosen_prim)
#             # print(f'LOCAL TRANSFORM: {local_transform}')

#         else:
#             # print(f'ISAAC WORLD NOT PLAYING!!!')

#             if sim_world.is_stopped():
#                 # print(f'sim_world is stopped!')

#                 if has_clicked_play:
#                     print(f'We should see this only once on stop!')

#                     print(f'sim_world current time: {sim_world.current_time}')

#                     world.delete_hsr()
#                     # sim_world.reset()

#                     world.add_hsr()

#                     has_clicked_play = False
#             else:
#                 # print(f'sim_world is paused!')
#                 pass

#             sim_world_onetime_trigger = True

#     world.sim_app.close()