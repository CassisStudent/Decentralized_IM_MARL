import os

import yaml
import sys
import torch as th
import time
from types import SimpleNamespace as SN
import numpy as np
import datetime

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.rl_video_recorder import RLVideoRecorder
from pymarlzooplus.learners.independent_ppo_learner import IndependentPPOLearner
from pymarlzooplus.utils.logging_setup import Logger

from torch.utils.tensorboard import SummaryWriter

from torch.multiprocessing import Process, Pipe
from functools import partial
import time
import copy


def read_config():
    config_dict = {}
    
    with open(os.path.join(os.path.dirname(__file__), "test_args.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    
    return config_dict

#COPIED BUT WITHOUT _LOG
def args_sanity_check(config):  #,  _log):

    # Set CUDA flags
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        print(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )
    if config["use_cuda_cnn_modules"] and not th.cuda.is_available():
        config["use_cuda_cnn_modules"] = False
        print(
            "CUDA flag use_cuda_cnn_modules was switched OFF automatically because no CUDA devices are available!"
        )
    if config["use_cuda_cnn_modules"] is False and config["use_cuda"] is True:
        config["use_cuda_cnn_modules"] = True
        print(
            "'use_cuda_cnn_modules' was turned to True since 'use_cuda' is also True!"
        )

    # Set 'centralized_image_encoding' arg
    if "centralized_image_encoding" in list(config["env_args"].keys()):
        if config["env_args"]["centralized_image_encoding"] is True and config["batch_size_run"] == 1:
            config["env_args"]["centralized_image_encoding"] = False
            print(
                "'centralized_image_encoding' was turned to False since only 1 env process is running!"
            )
    """
    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]
    """
    return config


def evaluate_agents(model_path, video_folder="eval_videos", num_episodes=5):
    if th.multiprocessing.get_start_method(allow_none=True) is None:
        th.multiprocessing.set_start_method('spawn')
    
    # 1a. Get config (TODO LETS CHECK ON SEED)
    _config = read_config()
    
    # check args sanity 
    _config = args_sanity_check(_config) #, _log)
    
    
    args = SN(**_config)
    
    num_gpus = th.cuda.device_count() if args.use_cuda else 1
    if num_gpus > 8: 
        num_gpus = 8 # Beperk tot je 8 x A16 setup
    
    args.device = "cpu"
    #if args.use_cuda and num_gpus > 1:
    #    args.device = "cuda:1"
        
    #args.device_cnn_modules = "cuda" if args.use_cuda_cnn_modules else "cpu"
    #th.backends.cudnn.benchmark = True
    
    
    # 1b. Create environment
    env = env_REGISTRY[_config["env"]](**_config["env_args"])
    env_args = [copy.deepcopy(args.env_args) for _ in range(args.buffer_size)]
    
    
    for i in range(args.buffer_size):
        env_args[i]["seed"] += i

    env_info = env.get_env_info()
    
    episode_limit = env_info["episode_limit"]
    max_episode_length = env_info["episode_limit"]
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    observation_dimension = env_info["obs_shape"]
    
    print("env_info. Render Capable: ")
    
    args.n_agents = n_agents
    args.n_actions = n_actions
    args.num_iterations = args.t_max // args.buffer_size

    

    
    # 1c. Create videorecorder.
    rl_video_recorder = RLVideoRecorder(env)

    # 2. Create per-agent PPO learners
    agents = []
    
    for i in range(n_agents):
        agent = IndependentPPOLearner(
            obs_dimension=observation_dimension,
            act_dimension=n_actions,
            args=args,
            device=args.device
        )
        
        # load_models pakt het bestand van de harde schijf en dwingt 
        # PyTorch via map_location om het direct naar cuda:1 te sturen!
        agent.load_models(model_path, agent_id=i)
        agents.append(agent)
    
    print(f"Modellen succesvol ingeladen vanuit {model_path}! Starten van {num_episodes}")
    
    
    
    # 2a. Create Buffers
    device = next(agents[0].actor.parameters()).device
    print(device)

    hidden_states_buffer = th.zeros(args.buffer_size, n_agents, max_episode_length, args.hidden_dim).pin_memory()
    
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).to("cuda:1")
        
        hidden_states = [agent.actor.init_hidden().to(agent.device) for agent in agents]
        episode_reward = 0
        step = 0
        
          
        while not done:
            actions = []
            rl_video_recorder.record_video()
          
            with th.no_grad():
                for i in range(n_agents):
                    action, value, next_hidden = agents[i].select_action_logits(obs_tensor[0, i].unsqueeze(0), hidden_states[i])
                    hidden_states[i] = next_hidden
                    actions.append(action)
            
            reward, done, extra_info = env.step(actions)
            next_obs = env.get_obs()
            
            obs_tensor = th.tensor(next_obs, dtype=th.float32).unsqueeze(0).to("cuda:1")
            episode_reward += reward
            
            
            if step % 250 == 0:
                print(step)
            
            step += 1

        rl_video_recorder.save_video(episode)    
        print(f"Episode {episode+1} voltooid | Totale Extrinsieke Score: {episode_reward:.2f} | Stappen: {step}")
    
    env.close()
    print(f"\nEvaluatie klaar! De video's (.mp4) staan klaar in de map: ....")


if __name__ == "__main__":
    # Pas dit aan naar de exacte map waar jouw .th bestanden staan
    MODEL_PATH = "./results/models/final" 
    evaluate_agents(MODEL_PATH, num_episodes=2)
    #vdisplay.stop()
    #print("Virtueel scherm netjes afgesloten.")
    #vdisplay.stop()
    print("Fake scherm netjes afgesloten.")