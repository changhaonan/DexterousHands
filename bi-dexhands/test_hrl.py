""" Test hrl
"""
from ast import arg
import numpy as np
import random
import datetime

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.process_marl import process_MultiAgentRL, get_AgentIndex
from utils.parse_task import parse_task
from algorithms.hrl.hrl_ppo import HRL_PPO

def test():
    agent_index = get_AgentIndex(cfg)
    # parse vec task
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    model_dict = {
        "random" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_0.pt",
        "grasp" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_1400.pt"
    }
    hrl = HRL_PPO(vec_env=env,
                cfg_train = cfg_train,
                device=env.rl_device,
                model_dict=model_dict,
                log_dir=logdir,
                is_testing=True,
                print_log=True,
                apply_reset=False,
                asymmetric=(env.num_states > 0))
    hrl.test()
    hrl.run_command([["grasp"], ["random"]])

if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    # start test
    test()