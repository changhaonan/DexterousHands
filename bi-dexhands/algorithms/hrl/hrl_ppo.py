""" Runner for hierarchical reinforcement learning algorithms.
"""
from datetime import datetime
import os
import time
import numpy as np

from gym.spaces import Space

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from algorithms.rl.ppo import ActorCritic

import copy

class HRL_PPO:

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

    def run_command(self, command_list):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()
        
        frame = 0
        stage = torch.zeros(self.vec_env.num_envs).to(self.device).to(torch.int64)
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
                for command in command_list:
                    action_list.append(action_dict[command[-1]])
                    move_list.append(torch.tensor(command[0], dtype=torch.float32, device=self.device).repeat(self.vec_env.num_envs, 1))
                action_list = torch.stack(action_list, dim=1)
                move_list = torch.stack(move_list, dim=1)
                actions = action_list[torch.arange(self.vec_env.num_envs), stage, :]
                moves = move_list[torch.arange(self.vec_env.num_envs), stage, :]
                # combine with move
                full_actions = torch.hstack((actions, moves))
                # step the vec_environment
                next_obs, rews, dones, infos = self.vec_env.step(full_actions)

                # update stage
                success = infos["successes"]
                stage = stage + success.to(torch.int64)  # proceed stage based success
                stage = torch.remainder(stage, len(command_list))
                stage = torch.where(dones > 0, torch.zeros_like(stage), stage)  # reset stage if done
                print(stage)
                current_obs.copy_(next_obs)
                
                # update frame
                frame += 1