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
        while True:
            with torch.no_grad():
                if self.apply_reset:
                    current_obs = self.vec_env.reset()
                if frame < 100:
                    command = command_list[0]
                else:
                    command = command_list[1]
                
                move_command = np.array(command[0])
                move_command = torch.from_numpy(move_command).float().to(self.device).repeat(self.vec_env.num_envs).reshape(self.vec_env.num_envs, -1)
                action_command = command[-1]
                # compute the action
                actions = self.actor_critic_dict[action_command].act_inference(current_obs)
                # combine with move
                full_actions = torch.hstack((actions, move_command))
                # step the vec_environment
                next_obs, rews, dones, infos = self.vec_env.step(full_actions)
                current_obs.copy_(next_obs)
                # update frame
                frame += 1
                frame = frame % 200