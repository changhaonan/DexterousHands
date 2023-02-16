# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from unittest import TextTestRunner
from matplotlib.pyplot import axis
from PIL import Image as Im

import numpy as np
import os
import random


from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

# import torch after isaacgym modules
import torch
from utils.torch_jit_utils import *
from utils.drawing_utils import *
from utils.control_utils import *


class RobotiqGraspAndPlaceSingle(BaseTask):
    """
    This class corresponds to the GraspAndPlace task. This environment consists of one single-hand, an
    object and a bucket that requires us to pick up the object and put it into the bucket.

    Args:
        cfg (dict): The configuration file of the environment, which is the parameter defined in the
            dexteroushandenvs/cfg folder

        sim_params (isaacgym._bindings.linux-x86_64.gym_37.SimParams): Isaacgym simulation parameters
            which contains the parameter settings of the isaacgym physics engine. Also defined in the
            dexteroushandenvs/cfg folder

        physics_engine (isaacgym._bindings.linux-x86_64.gym_37.SimType): Isaacgym simulation backend
            type, which only contains two members: PhysX and Flex. Our environment use the PhysX backend

        device_type (str): Specify the computing device for isaacgym simulation calculation, there are
            two options: 'cuda' and 'cpu'. The default is 'cuda'

        device_id (int): Specifies the number of the computing device used when simulating. It is only
            useful when device_type is cuda. For example, when device_id is 1, the device used
            is 'cuda:1'

        headless (bool): Specifies whether to visualize during training

        agent_index (list): Specifies how to divide the agents of the hands, useful only when using a
            multi-agent algorithm. It contains one list, representing the right hand.
            Each list has six numbers from 0 to 5, representing the palm, middle finger, ring finger,
            tail finger, index finger, and thumb. Each part can be combined arbitrarily, and if placed
            in the same list, it means that it is divided into the same agent. The default setting is
            [[[0, 1, 2, 3, 4, 5]]], which means that the two whole hands are
            regarded as one agent respectively.

        is_multi_agent (bool): Specifies whether it is a multi-agent environment
    """

    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
        agent_index=[[[0, 1, 2, 3, 4, 5]]],
        is_multi_agent=False,
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.sub_task = self.cfg["env"]["sub_task"]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        self.hold_still_len = self.cfg["env"]["holdStillLen"]
        self.hold_still_vel_tolerance = self.cfg["env"]["holdStillVelTolerance"]
        self.hold_still_reward_scale = self.cfg["env"]["holdStillRewardScale"]
        self.hold_still_anglevel_scale = self.cfg["env"]["holdStillAngVelScale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        # assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = self.object_type == "pen"

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "pot": "mjcf/bucket/100454/mobility.urdf",
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        self.action_type = self.cfg["env"]["actionType"]

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state", "absolute_pos", "relative_pos"]):
            raise Exception("Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)

        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "point_cloud": 226 + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": 226 + self.num_point_cloud_feature_dim * 3,
            "full_state": 220 if (self.action_type == "hand_only") else 226,
        }
        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = "z"

        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states

        self.num_agents = 1
        if self.action_type == "wrist_only":
            self.cfg["env"]["numActions"] = 7
        elif self.action_type == "hand_only":
            self.cfg["env"]["numActions"] = 11
        else:
            self.cfg["env"]["numActions"] = 18

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs + self.num_object_dofs * 2)
        self.dof_force_tensor = self.dof_force_tensor[:, :24]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :,
            self.num_shadow_hand_dofs : self.num_shadow_hand_dofs + self.num_object_dofs,
        ]
        self.object_dof_pos = self.object_dof_state[..., 0]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_rigid_body_states = self.rigid_body_states.clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        # buf for grasp
        self.hold_still_count_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.disturb_force = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

        # info
        self.num_actors = self.gym.get_actor_count(self.envs[0])

        if cfg["env"]["save_success_state"]:
            # initialize success state
            self.save_success_state = True
            self.success_root_actor_state = gymtorch.wrap_tensor(actor_root_state_tensor).clone()
            self.success_dof_state = gymtorch.wrap_tensor(dof_state_tensor).clone()
        else:
            self.save_success_state = False

        # set camera
        cam_pos = gymapi.Vec3(0.4, -0.4, 1.7)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        """
        Allocates which device will simulate and which device will render the scene. Defines the simulation type to be used
        """
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """
        Adds ground plane to simulation
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Create multiple parallel isaacgym environments

        Args:
            num_envs (int): The total number of environment

            spacing (float): Specifies half the side length of the square area occupied by each environment

            num_per_row (int): Specify how many environments in a row
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        shadow_hand_asset_file = "urdf/robotiq/robotiq-3f-gripper_articulated.urdf"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = [
            "robot0:T_FFJ1c",
            "robot0:T_MFJ1c",
            "robot0:T_RFJ1c",
            "robot0:T_LFJ1c",
        ]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)

        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        # self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]
        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]  # all dofs are actuated

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        # self.sensors = []
        # sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props["lower"][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props["upper"][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        # object_asset_options.collapse_fixed_joints = True
        # object_asset_options.disable_gravity = True
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 100000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        block_asset_file = "urdf/objects/cube_multicolor1.urdf"
        block_asset = self.gym.load_asset(self.sim, asset_root, block_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        # set object dof properties
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props["lower"][i])
            self.object_dof_upper_limits.append(object_dof_props["upper"][i])

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)

        # create table asset
        table_dims = gymapi.Vec3(1.0, 1.0, 0.6)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.2, -0.2, 1.0)  # gymapi.Vec3(0.55, 0.2, 0.8)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 0, 1.57)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.2, 0.6)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0

        block_start_pose = gymapi.Transform()
        block_start_pose.p = gymapi.Vec3(0.0, -0.2, 0.6)  # 0.0, -0.2, 0.6
        block_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 1.57, 0)
        # object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
        # object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        # object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.0, 0.0, 10)
        self.goal_displacement_tensor = to_torch(
            [
                self.goal_displacement.x,
                self.goal_displacement.y,
                self.goal_displacement.z,
            ],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0.0, 0, 0)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 2 + 3 * self.num_object_bodies + 1
        max_agg_shapes = self.num_shadow_hand_shapes * 2 + 3 * self.num_object_shapes + 1

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []

        self.goal_object_indices = []
        self.table_indices = []
        self.block_indices = []
        self.block_rb_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append(
                [
                    shadow_hand_start_pose.p.x,
                    shadow_hand_start_pose.p.y,
                    shadow_hand_start_pose.p.z,
                    shadow_hand_start_pose.r.x,
                    shadow_hand_start_pose.r.y,
                    shadow_hand_start_pose.r.z,
                    shadow_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            hand_rigid_body_index = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]

            for n in self.agent_index[0]:
                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(
                            env_ptr,
                            shadow_hand_actor,
                            o,
                            gymapi.MESH_VISUAL,
                            gymapi.Vec3(colorx, colory, colorz),
                        )

            # # create fingertip force-torque sensors
            # self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            # self.gym.set_actor_scale(env_ptr, object_handle, 0.3)

            # add block
            block_handle = self.gym.create_actor(env_ptr, block_asset, block_start_pose, "block", i, 0, 0)
            block_idx = self.gym.get_actor_index(env_ptr, block_handle, gymapi.DOMAIN_SIM)
            block_rb_idx = self.gym.get_actor_rigid_body_index(env_ptr, block_handle, 0, gymapi.DOMAIN_ENV)
            self.block_rb_indices.append(block_rb_idx)
            self.block_indices.append(block_idx)

            # add goal object
            goal_handle = self.gym.create_actor(
                env_ptr,
                goal_asset,
                goal_start_pose,
                "goal_object",
                i + self.num_envs,
                0,
                0,
            )
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            # self.gym.set_actor_scale(env_ptr, goal_handle, 0.3)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr,
                    object_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )
                self.gym.set_rigid_body_color(
                    env_ptr,
                    goal_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.6, 0.72, 0.98),
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        # init for grasp
        self.goal_right_move = self.goal_states.clone()
        self.grasp_goal = self.goal_states.clone()
        self.right_wrist_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.right_wrist_rot = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.block_rb_indices = to_torch(self.block_rb_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        self.block_indices = to_torch(self.block_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        pass

    def compute_observations(self):
        # update wrist pos
        self.right_wrist_pos = self.rigid_body_states[:, 1, 0:3]
        self.right_wrist_rot = self.rigid_body_states[:, 1, 3:7]

    def compute_full_state(self, asymm_obs=False):
        pass

    def compute_point_cloud_observation(self, collect_demonstration=False):
        pass

    def reset_target_pose(self, env_ids, apply_reset=False):
        pass

    def reset(self, env_ids, goal_env_ids=None):
        pass

    def pre_physics_step(self, actions):
        """
        The pre-processing of the physics step. Determine whether the reset environment is needed,
        and calculate the next movement of Shadowhand through the given action. The 16-dimensional
        action space as shown in below:

        Index   Description
        0 - 9 	right shadow hand actuated joint (Blocked when wrist_only)
        10 - 12	right shadow hand base translation (Blocked when hand_only)
        13 - 15	right shadow hand base rotation (Blocked when hand_only)

        Args:
            actions (tensor): Actions of agents in the all environment
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # reset the environment

        # log the action
        if actions.shape[-1] == self.num_actions:
            self.actions = actions.clone().to(self.device)
        else:
            # external move control
            self.actions = actions[:, : self.num_actions].clone().to(self.device)

        # apply control
        if self.action_type == "hand_only":
            if actions.shape[-1] == self.num_actions:
                pass
            else:
                # external move control
                right_force = pid_control_transition(
                    actions[:, self.num_actions : self.num_actions + 3],
                    self.right_wrist_pos,
                    10.0,
                )
                right_force = torch.clamp(right_force, -1.0, 1.0)
                self.apply_forces[:, 1, :] = right_force * self.dt * self.transition_scale * 100000
                right_torque = pid_control_rotation(
                    actions[:, self.num_actions + 3 : self.num_actions + 7],
                    self.right_wrist_rot,
                    10.0,
                )
                right_torque = torch.clamp(right_torque, -1.0, 1.0)
                self.apply_torque[:, 1, :] = right_torque * self.dt * self.orientation_scale * 1000
                # reset goal for vis bug
                self.goal_right_move[:, :3] = actions[:, self.num_actions : self.num_actions + 3]
                self.goal_right_move[:, 3:7] = actions[:, self.num_actions + 3 : self.num_actions + 7]

            # joint control target
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions[:, :self.num_actions],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )

        rb_force = self.apply_forces

        # move control
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(rb_force),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )
        # joint control
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        all_hand_indices = torch.unique(self.hand_indices).to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        # visualize the goal
        for i in range(self.num_envs):
            if self.action_type == "hand_only":
                # move goal
                draw_6D_pose(
                    self.gym,
                    self.viewer,
                    self.envs[i],
                    self.goal_right_move[i, :3],
                    self.goal_right_move[i, 3:7],
                    color=(0, 1, 0),
                )
