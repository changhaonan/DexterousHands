""" This file is only used for test environment
"""
from ast import arg
import numpy as np
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

    # reset
    current_obs = env.reset()
    # run
    while True:
        # action
        action = -0.5 * torch.ones([env.num_envs, env.num_actions + 7], dtype=torch.float32, device=args.rl_device)
        # set pos
        action[:, -7] = 0.0
        action[:, -6] = 0.0
        action[:, -5] = 1.5
        # set rot: rotate 90 degree around y axis and then rotate 10 degree around z axis
        rot1 = R.from_euler("y", 90, degrees=True)
        rot2 = R.from_euler("z", 30, degrees=True)
        rot = rot1 * rot2
        action[:, -4:] = torch.from_numpy(rot.as_quat())

        # step
        next_obs, reward, done, info = env.step(action)
        # update
        current_obs = next_obs
