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

model_dict = {
    "RELEASE" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_0.pt",
    "GRASP" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_4000.pt"
}

def parse_command_file(command_file):
    command_list = []
    with open(command_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):  # comment
                continue
            line_list = line.upper().split(" ")
            # check line list and convert number to float
            for i, word in enumerate(line_list):
                if word.startswith("("):
                    number_f = [float(number) for number in word[1:-1].split(",")] 
                    line_list[i] = number_f

            command_list.append(line_list)
    return command_list


def test(program_name):
    agent_index = get_AgentIndex(cfg)
    # parse vec task
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
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
    command = parse_command_file(f"test/{program_name}.manip")
    lm_engine.run_command(command, 4, "/home/robot-learning/Projects/DexterousHands/bi-dexhands/test/gym_states.pt")
    # lm_engine.test_env("/home/robot-learning/Projects/DexterousHands/bi-dexhands/test/gym_states.pt")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    # start test
    program_name = "grasp"
    test(program_name=program_name)