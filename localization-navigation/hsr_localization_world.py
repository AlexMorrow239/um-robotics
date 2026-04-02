import argparse
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
