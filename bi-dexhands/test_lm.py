""" Test lm_engine
"""
from ast import arg
import numpy as np
import random
import datetime

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.process_marl import process_MultiAgentRL, get_AgentIndex
from utils.parse_task import parse_task
from algorithms.lm.lm_engine import LM_ENGINE

command_file_dict = {
    "ShadowHandGraspAndPlaceSingle" : "test/grasp_place.manip",
}


def parse_command_file(command_file):
    command_list = []
    with open(command_file, "r") as f:
        for line in f:
            line = line.strip()
            line_list = line.lower().split(" ")
            # check line list and convert number to float
            for i, word in enumerate(line_list):
                if word.startswith("("):
                    number_f = [float(number) for number in word[1:-1].split(",")] 
                    line_list[i] = number_f

            command_list.append(line_list)
    return command_list


def test():
    agent_index = get_AgentIndex(cfg)
    # parse vec task
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    model_dict = {
        "release" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_0.pt",
        "grasp" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_1400.pt"
    }
    lm_engine = LM_ENGINE(vec_env=env,
                        cfg_train = cfg_train,
                        device=env.rl_device,
                        model_dict=model_dict,
                        log_dir=logdir,
                        is_testing=True,
                        print_log=True,
                        apply_reset=False,
                        asymmetric=(env.num_states > 0))
    lm_engine.test()
    # read command file
    command = parse_command_file(command_file_dict[args.task])
    lm_engine.run_command(command)


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    # start test
    test()