""" This file is only used for test environment
"""
from ast import arg
import numpy as np
import cv2
import random
import datetime
from scipy.spatial.transform import Rotation as R

# tasks
from tasks.shadow_hand_grasp_and_place_v2_single import ShadowHandGraspAndPlaceV2Single
from tasks.robotiq_grasp_and_place_single import RobotiqGraspAndPlaceSingle
from tasks.hand_base.vec_task import (
    VecTaskCPU,
    VecTaskGPU,
    VecTaskPython,
    VecTaskPythonArm,
)

from utils.config import (
    set_np_formatting,
    set_seed,
    get_args,
    parse_sim_params,
    load_cfg,
)

import torch

# teleop related
from utils.teleop.mediapipe_hand_pose import MediapipeHandEstimator

# Pybind
import sys
import os

pybind_path = os.path.join(os.path.dirname(__file__), "../build")
sys.path.append(pybind_path)
import finger_map


class TeleopAgent:
    def __init__(self, hand_pose_estimator, num_finger_action, pre_ee_rot=np.array([])):
        self.hand_pose_estimator = hand_pose_estimator
        self.num_finger_action = num_finger_action
        self.pre_ee_rot = pre_ee_rot
        self.last_ee_pose = np.zeros(3)
        self.last_finger_pose = np.zeros(num_finger_action)
        self.move_boundary_mid = np.array([0.0, 0.0, 1.0])
        self.move_boundary_range = np.array([0.5, 0.5, 0.2])

    def read_from_teleop(self, frame_image):
        joints_3d = self.hand_pose_estimator.predict_3d_joints(frame_image)
        if len(joints_3d) < 3:
            # scale with move_boundary
            move_ee_pos = np.copy(self.last_ee_pose)
            move_ee_pos[2] = 0.0  # fix z-axis
            x_pos, y_pos = move_ee_pos[0:2]  # flip x, y
            move_ee_pos[0:2] = [y_pos, -x_pos]
            move_ee_pos[0:3] = move_ee_pos[0:3] * self.move_boundary_range + self.move_boundary_mid
            if self.pre_ee_rot.shape[0] > 0:
                move_ee_rot = self.pre_ee_rot
            else:
                # FIXME: use fixed one for now
                move_ee_rot = np.array([0.0, 0.0, 0.0, 1.0])
            move_ee = np.hstack([move_ee_pos, move_ee_rot])
            return move_ee, self.last_finger_pose
        else:
            # rescale x, y to [-1, 1]
            ee_pose_xyz = joints_3d[0:3] * 2.0 - 1.0
            self.last_ee_pose[0:3] = ee_pose_xyz
            # ee pos
            move_ee_pos = np.copy(self.last_ee_pose)
            move_ee_pos[2] = 0.0  # fix z-axis
            x_pos, y_pos = move_ee_pos[0:2]  # flip x, y
            move_ee_pos[0:2] = [y_pos, -x_pos]
            move_ee_pos[0:3] = move_ee_pos[0:3] * self.move_boundary_range + self.move_boundary_mid
            if self.pre_ee_rot.shape[0] > 0:
                move_ee_rot = self.pre_ee_rot
            else:
                # FIXME: use fixed one for now
                move_ee_rot = np.array([0.0, 0.0, 0.0, 1.0])
            move_ee = np.hstack([move_ee_pos, move_ee_rot])
            # finger pose
            finger_pose_full = finger_map.retarget(joints_3d[3:])
            # finger_pose_full = -np.ones([self.num_finger_action])
            self.last_finger_pose = finger_pose_full[0:self.num_finger_action]
            return move_ee, self.last_finger_pose


if __name__ == "__main__":
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    # create task
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
        is_multi_agent=False,
    )
    env = VecTaskPython(
        task,
        args.rl_device,
        cfg_train.get("clip_observations", 5.0),
        cfg_train.get("clip_actions", 1.0),
    )

    # prepare the teleop related
    teleop_mode = "webcam"
    video_file = ""
    if teleop_mode == "webcam":
        cap = cv2.VideoCapture(0)
    elif teleop_mode == "video":
        cap = cv2.VideoCapture(video_file)
    else:
        raise ValueError("Unknown teleop mode!")
    hand_pose_estimator = MediapipeHandEstimator()
    rot1 = R.from_euler("y", 90, degrees=True)
    rot2 = R.from_euler("z", 30, degrees=True)
    rot3 = R.from_euler("x", -90, degrees=True)
    rot = rot1 * rot2 * rot3
    teleop_agent = TeleopAgent(hand_pose_estimator, env.num_actions, pre_ee_rot=rot.as_quat())

    # reset
    current_obs = env.reset()
    # run
    while True:
        if args.use_teleop:
            if cap.isOpened():
                ret, frame_image = cap.read()
                if not ret and teleop_mode == "video":
                    print("Loop video from start.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue  # jump this frame
            else:
                raise ValueError("Cannot open video capture!")
            move_action, finger_action = teleop_agent.read_from_teleop(frame_image)
            # combine and repeat for all envs
            action = np.concatenate([finger_action, move_action])
            action = np.tile(action, (env.num_envs, 1))
            action = torch.from_numpy(action).to(args.rl_device).to(torch.float32)
        else:
            # use hard-coded action
            action = -0.5 * torch.ones([env.num_envs, env.num_actions + 7], dtype=torch.float32, device=args.rl_device)
            # set pos
            action[:, -7] = 0.0
            action[:, -6] = 0.0
            action[:, -5] = 1.5
            action[:, -4:] = torch.from_numpy(rot.as_quat())

        # step
        next_obs, reward, done, info = env.step(action)
        # update
        current_obs = next_obs
