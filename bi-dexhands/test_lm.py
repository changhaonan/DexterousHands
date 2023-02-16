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

model_dict = {
    "RELEASE" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_release.pt",
    "GRASP" : "logs/ShadowHandGraspAndPlaceSingle/ppo/ppo_seed-1/model_grasp.pt"
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


def test(program_name, test_option=None):
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
    # read command file
    command = parse_command_file(f"test/{program_name}.manip")

    lm_engine.init(command[0], test=True, teleop_mode="video", other_args=test_option)
    lm_engine.run_command(command, 10, "./test/gym_states.pt")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    # start test
    # program_name = "grasp"
    # program_name = "release"
    # program_name = "grasp_place"
    # program_name = "grasp_place_v2"
    program_name = "teleop"
    test_option = {"video_file" : "./data/teleop/grasp.MOV"}
    test(program_name=program_name, test_option=test_option)