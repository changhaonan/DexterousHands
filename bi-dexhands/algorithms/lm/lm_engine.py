""" Runner for hierarchical reinforcement learning algorithms.
"""
from datetime import datetime
import os
import time
import numpy as np
import copy
from gym.spaces import Space

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from algorithms.rl.ppo import ActorCritic
from algorithms.lm.check_utils import *


class LM_ENGINE:
    # language maniulation engine

    def __init__(self,
                vec_env, 
                cfg_train, 
                device, 
                model_dict, 
                log_dir='run',
                is_testing=False,
                print_log=True,
                apply_reset=False,
                asymmetric=False):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]
        self.device = device
        self.asymmetric = asymmetric
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.num_transitions_per_env=learn_cfg["nsteps"]
        self.learning_rate=learn_cfg["optim_stepsize"]

        self.vec_env = vec_env
        self.model_dict = model_dict

        # parameters
        self.actor_critic_dict = {}
        for key, value in model_dict.items():
            self.actor_critic_dict[key] = ActorCritic(self.observation_space.shape, self.state_space.shape, self.action_space.shape,
                self.init_noise_std, self.model_cfg, asymmetric=asymmetric)
            self.actor_critic_dict[key].to(self.device)
        
        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset 
    
    def test(self):
        # load sub policy
        for key, value in self.actor_critic_dict.items():
            value.load_state_dict(torch.load(self.model_dict[key]))
            value.eval()

    def test_env(self, output_path):
        # run 1000 steps
        # self.vec_env.task.save_gym_states(output_path)
        # reset 
        self.vec_env.reset()
        # load 1000 steps
        self.vec_env.task.load_gym_states(output_path)
    
    def save_state(self, output_path):
        pass

    def check(self, check_command, loc):
        # conduct checking
        if check_command[1] == "POS":
            # pos check
            P0 = self.vec_env.task.get_obj_pos(check_command[2].lower())
            if len(check_command) > 4:
                P1 = self.vec_env.task.get_obj_pos(check_command[3].lower())
            check_result = eval(check_command[-1][1:-1])
        elif check_command[1] == "VEL":
            # pos check
            V0 = self.vec_env.task.get_obj_vel(check_command[2].lower())
            if len(check_command) > 4:
                V1 = self.vec_env.task.get_obj_vel(check_command[3].lower())
            check_result = eval(check_command[-1][1:-1])
        elif check_command[1] == "TIMER":
            T = loc['stage_count']
            check_result = eval(check_command[-1][1:-1])
        else:
            raise ValueError("Unknown check command")
        return check_result

    def run_command(self, command_list, num_rounds, output_path):
        current_obs = self.vec_env.reset()
        
        frame = 0
        stage = torch.zeros(self.vec_env.num_envs).to(self.device).to(torch.int64)
        stage_count = torch.zeros(self.vec_env.num_envs).to(self.device).to(torch.int64)
        success_rounds = torch.zeros(self.vec_env.num_envs).to(self.device)
        finished_rounds = torch.zeros(self.vec_env.num_envs).to(self.device)

        while True:
            with torch.no_grad():
                if self.apply_reset:
                    current_obs = self.vec_env.reset()
                # compute the action according to stage
                action_dict = {}
                for key, value in self.actor_critic_dict.items():
                    action_dict[key] = value.act_inference(current_obs)
                action_list = []
                move_list = []
                check_list = []
                num_stage = -1
                for command in command_list:
                    if command[0] == "MOVE":
                        action_list.append(action_dict[command[2]])
                        move_list.append(torch.tensor(command[1], dtype=torch.float32, device=self.device).repeat(self.vec_env.num_envs, 1))
                        # switch to next stage
                        num_stage += 1
                        check_list.append(torch.ones(self.vec_env.num_envs).to(self.device).to(torch.bool))
                    elif command[0] == "CHECK":
                        check_list[num_stage] = torch.logical_and(check_list[num_stage], self.check(command, locals()))
                num_stage = num_stage + 1
                action_list = torch.stack(action_list, dim=1)
                move_list = torch.stack(move_list, dim=1)
                check_list = torch.stack(check_list, dim=1)
                actions = action_list[torch.arange(self.vec_env.num_envs), stage, :]
                moves = move_list[torch.arange(self.vec_env.num_envs), stage, :]
                checks = check_list[torch.arange(self.vec_env.num_envs), stage]
                # combine with move
                full_actions = torch.hstack((actions, moves))
                # step the vec_environment
                next_obs, rews, dones, infos = self.vec_env.step(full_actions)

                # conduct checking
                # check_result_list = []
                # for stage, stage_check_command in enumerate(check_list):
                #     stage_check_result = torch.ones(self.vec_env.num_envs).to(self.device).to(torch.bool)
                #     for check_command in stage_check_command:
                #         stage_check_result = torch.logical_and(stage_check_result, self.check(check_command))
                #     check_result_list.append(stage_check_result)
                # check_result = torch.stack(check_result_list, dim=1)
                # checks = check_result[torch.arange(self.vec_env.num_envs), stage]
                # update stage
                success = infos["successes"]
                # stage = stage + success.to(torch.int64)  # proceed stage based success
                stage_count = stage_count + 1
                stage = stage + checks.to(torch.int64)  # proceed stage based checks
                stage_count = torch.where(checks, torch.zeros_like(stage_count), stage_count)  # reset stage count if stage proceed
                stage = torch.where(dones > 0, torch.zeros_like(stage), stage)  # reset stage if done
                stage_count = torch.where(dones > 0, torch.zeros_like(stage_count), stage_count)  # reset stage count if stage proceed
                current_obs.copy_(next_obs)

                # update finished rounds & success rounds
                success_rounds = torch.where(stage >= num_stage, success_rounds + 1, success_rounds)
                finished_rounds = torch.where(torch.logical_or(dones > 0, stage >= num_stage), finished_rounds + 1, finished_rounds)
                # reset if success
                reset_env_idx = torch.arange(self.vec_env.num_envs)[stage >= num_stage]
                if reset_env_idx.shape[0] > 0:
                    self.vec_env.task.reset(reset_env_idx)
                print("--------------------")
                print(f"Checks: {checks[0]}")
                print(f"Stage: {stage[0]}")
                print(f"Stage count: {stage_count[0]}")
                print(f"Success ratio: {success_rounds.sum() / (finished_rounds.sum() + 1e-6)}.")
                stage = torch.remainder(stage, num_stage)
                # update frame
                frame += 1

                if finished_rounds.sum() > num_rounds * self.vec_env.num_envs:
                    break
    
        # save states when leaving
        self.vec_env.task.save_success_gym_states(output_path)
        # generate the final success ratio
        print(f"Success ratio: {success_rounds.sum() / (finished_rounds.sum() + 1e-6)}.")
