# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os

from evaluator import OVMMEvaluator
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)

from home_robot.agent.ovmm_agent.ovmm_3dic_agent import OpenVocabManipAgent
from home_robot.agent.ovmm_agent.ovmm_exploration_agent import OVMMExplorationAgent
from home_robot.agent.ovmm_agent.random_agent import RandomAgent

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="/home/kmz/ovmm/home-robot-data/projects/habitat_ovmm/configs/agent/heuristic_agent.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="/home/kmz/ovmm/home-robot-data/projects/habitat_ovmm/configs/env/hssd_demo.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "random", "explore"],
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--force_step",
        type=int,
        default=20,
        help="force to switch to new episode after a number of steps",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="whether to save obseration history for data collection",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--scene_id",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    scenes = ['102816756.scene_instance', 
    '102817140.scene_instance', 
    '103997586_171030669.scene_instance', 
    '103997895_171031182.scene_instance', 
    '104348328_171513363.scene_instance', 
    '104348361_171513414.scene_instance', 
    '105515211_173104185.scene_instance', 
    '106366386_174226770.scene_instance', 
    '106366410_174226806.scene_instance', 
    '106878915_174887025.scene_instance', 
    '107733960_175999701.scene_instance', 
    '107734176_176000019.scene_instance']

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)

    # get env config
    env_config = get_omega_config(args.env_config_path)

    if args.scene_id >= 0:
        import habitat
        with habitat.config.read_write(habitat_config):
            habitat_config.habitat.dataset.content_scenes = [scenes[args.scene_id]]
        with habitat.config.read_write(env_config):
            env_config.EXP_NAME = scenes[args.scene_id].split('.')[0]

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )

    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)

    device_id = env_config.habitat.simulator.habitat_sim_v0.gpu_device_id

    # create agent
    if args.agent_type == "random":
        agent = RandomAgent(agent_config, device_id=device_id)
    elif args.agent_type == "explore":
        agent = OVMMExplorationAgent(agent_config, device_id=device_id, args=args)
    else:
        agent = OpenVocabManipAgent(agent_config, device_id=device_id)
    
    # create evaluator
    evaluator = OVMMEvaluator(env_config, data_dir=args.data_dir)

    # evaluate agent
    metrics = evaluator.evaluate(
        agent=agent,
        evaluation_type=args.evaluation_type,
        num_episodes=args.num_episodes,
    )
    print("Metrics:\n", metrics)

