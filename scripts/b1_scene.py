import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
import random
from pprint import pprint
from scipy.spatial.transform import Rotation as R

'''
['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper']


'''

axes_geom = gymutil.AxesGeometry(0.1)
ROOT = "/home/bikram/Documents/isaacgym/assets"


class b1Scene:

      def __init__(self,
                   control_type: str = "position",
                   control_freq: float = 40.,
                   ):
            self.gym = gymapi.acquire_gym()
            self.sim_params = gymapi.SimParams()
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
            self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -5.8)
            sim_type = gymapi.SIM_PHYSX
            compute_device_id = 0
            graphics_device_id = 0
            self.control_type = control_type
            available_control_types = ['position', 'velocity', 'torque']
            assert control_type in available_control_types, f"only available control modes are : {available_control_types}"
            
            self.sim_params.dt = 1 / control_freq
            if sim_type == gymapi.SIM_FLEX:
                  self.sim_params.substeps = 4
                  self.sim_params.flex.solver_type = 5
                  self.sim_params.flex.num_outer_iterations = 4
                  self.sim_params.flex.num_inner_iterations = 20
                  self.sim_params.flex.relaxation = 0.75
                  self.sim_params.flex.warm_start = 0.8


            elif sim_type == gymapi.SIM_PHYSX:
                  self.sim_params.substeps = 2
                  self.sim_params.physx.solver_type = 1
                  self.sim_params.physx.num_position_iterations = 10
                  self.sim_params.physx.num_velocity_iterations = 0
                  self.sim_params.physx.num_threads = 0
                  self.sim_params.physx.use_gpu = True
                  self.sim_params.physx.rest_offset = 0.001

                  ### accordint to IsaacGymEnvs
                  # self.sim_params.physx.num_threads = 6
                  # self.sim_params.physx.solver_type = 1
                  # self.sim_params.physx.num_position_iterations = 8
                  # self.sim_params.physx.num_velocity_iterations = 0
                  # self.sim_params.physx.max_gpu_contact_pairs = 8388608
                  # self.sim_params.physx.rest_offset = 0.
                  # self.sim_params.physx.use_gpu = True
                  # self.sim_params.physx.bounce_threshold_velocity = 0.02
                  # self.sim_params.physx.max_depenetration_velocity = 1000.0
                  # self.sim_params.physx.default_buffer_size_multiplier = 25.0
                  # self.sim_params.physx.contact_offset = 0.002

            self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, sim_type, self.sim_params)
            self.plane_params = gymapi.PlaneParams()

            self.plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
            self.plane_params.distance = 0
            self.plane_params.static_friction = 1
            self.plane_params.dynamic_friction = 1
            self.plane_params.restitution = 0
            self.gym.add_ground(self.sim, self.plane_params)



            b1_urdf = "urdf/b1/urdf/b1_plus_z1.urdf"



            asset_options = gymapi.AssetOptions()
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.collapse_fixed_joints = True
            asset_options.replace_cylinder_with_capsule = True
            asset_options.flip_visual_attachments  = False
            asset_options.fix_base_link = False
            asset_options.density = 0.001
            asset_options.angular_damping = 0.0
            asset_options.linear_damping = 0.0
            asset_options.armature = 0.0
            asset_options.thickness = 0.01
            asset_options.disable_gravity = False

            robot_ = self.gym.load_asset(self.sim, ROOT, b1_urdf, asset_options)
            # asset_options.flip_visual_attachments = True
            # franka_ = self.gym.load_asset(self.sim, ROOT, franka_urdf, asset_options)

            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            spacing = 2.0
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
            self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1) 
       
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(11.88, 4.0, 0.0)
            pose.r = gymapi.Quat(0, 0, 0.7068252, 0.7073883)
            
            pose.p = gymapi.Vec3(12.0, 5.0, 0.7)
            pose.r = gymapi.Quat( 0, 0, 0.7068252, 0.7073883)
            self.robot_handle = self.gym.create_actor(self.env, robot_, pose, "robot", 0, 1)
            pose.p = gymapi.Vec3(12.0, 5.0, 1.7)
            # self.franka_handle = self.gym.create_actor(self.env, franka_, pose, "robot2", 0, 1)
            robot_props = self.gym.get_asset_dof_properties(robot_)
            robot_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            robot_props["stiffness"].fill(1000.)
            robot_props["damping"].fill(220.)       
            print(robot_props)
            self.gym.set_actor_dof_properties(self.env, self.robot_handle, robot_props)







      def apply_arm_action(self, action: list):
            if self.control_type == 'position':
                  action = np.array(action)
                  dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
                  joint_pos = [dof_state['pos'] for dof_state in dof_states]
                  gripper_jts = np.array([joint_pos[7], joint_pos[8]])
                  action = np.concatenate((action, gripper_jts), axis = 0).tolist()
                  self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, action)
            elif self.control_type == 'velocity':
                  # if len(action) == 7:
                  #       act = np.array(action)
                  #       grippers = np.array([0., 0.])
                  #       action = np.concatenate((act, grippers), axis = 0).tolist()
                  self.gym.set_actor_dof_velocity_targets(self.env, self.robot_handle, action)
            else:
                  raise NotImplementedError("I will not implement it.")


      def get_state(self, handle):
            dof_states = self.gym.get_actor_rigid_body_states(self.env, handle, gymapi.STATE_ALL)
            pos = np.array([dof_states['pose']['p']['x'][0], dof_states['pose']['p']['y'][0], dof_states['pose']['p']['z'][0]])
            orien = np.array([dof_states['pose']['r']['x'][0], dof_states['pose']['r']['y'][0], dof_states['pose']['r']['z'][0], dof_states['pose']['r']['w'][0]])
            return pos.astype('float64'), orien.astype('float64')
      
      def to_robot_frame(self, pos, quat):
            robot_pos_world = np.array([12.0, 5.0, 0.7])
            robot_quat_world = np.array([0.0, 0.0, 0.707, 0.707]) 
            robot_rot_world = R.from_quat(robot_quat_world)
            T_robot_world = np.eye(4)
            T_robot_world[:3, :3] = robot_rot_world.as_matrix()
            T_robot_world[:3, 3] = robot_pos_world
            T_world_robot = np.linalg.inv(T_robot_world)
            point_world_hom = np.ones(4)
            point_world_hom[:3] = pos
            point_robot_hom = T_world_robot @ point_world_hom
            point_pos_robot = point_robot_hom[:3]
            if np.linalg.norm(quat) == 0.:
                  quat = np.array([1., 0., 0., 0.])
            point_rot_world = R.from_quat(quat)
            point_rot_robot = robot_rot_world.inv() * point_rot_world
            point_quat_robot = point_rot_robot.as_quat()  
            return point_pos_robot, point_quat_robot


      def __del__(self):
            print("scene deleted")

      def orientation_error(self, desired, current):
            cc = quat_conjugate(current)
            q_r = quat_mul(desired, cc)
            return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


      

      
      def get_viewer(self):
            return self.viewer
      
      def get_sim(self):
            return self.sim
      
      def get_gym(self):
            return self.gym
      
      def get_env(self):
            return self.env
      
      def get_robot_handle(self):
            return self.robot_handle
      

      def step(self):
            t = self.gym.get_sim_time(self.sim)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
            for evt in self.gym.query_viewer_action_events(self.viewer):
                  if evt.action == 'reset' and evt.value > 0:
                        self.gym.set_sim_rigid_body_states(self.sim, self.reset_state, gymapi.STATE_ALL)
                        # self.gym.set_actor_rigid_body_states(self.env, self.robot_handle, self.initial_state, gymapi.STATE_POS)
            return t


      def get_joint_pos_vel(self, handle) -> tuple:
            dof_states = self.gym.get_actor_dof_states(self.env, handle, gymapi.STATE_ALL)
            joint_pos = [float(dof_state['pos']) for dof_state in dof_states]
            joint_vel = [float(dof_state['vel']) for dof_state in dof_states]
            return joint_pos, joint_vel
      


      def viewer_running(self):
            return not self.gym.query_viewer_has_closed(self.viewer)


if __name__ == '__main__':
      scene = b1Scene()
      gym = scene.get_gym()
      env = scene.get_env()
      sim = scene.get_env()
      while scene.viewer_running():
            scene.step()
            print(gym.get_actor_dof_names(env, scene.robot_handle))
