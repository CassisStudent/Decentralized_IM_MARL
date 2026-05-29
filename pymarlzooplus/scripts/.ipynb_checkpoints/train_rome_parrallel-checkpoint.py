# Import packages
import yaml
import os
import sys
import torch
import time
from types import SimpleNamespace as SN
import numpy as np
import asyncio

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.rl_video_recorder import RLVideoRecorder
from pymarlzooplus.learners.independent_ppo_learner import IndependentPPOLearner
from pymarlzooplus.utils.logging_setup import Logger






# #################################### API for fully cooperative tasks. ###################################
# # 'n_agns' (int) is the number of agents in the environment.
# # 'n_acts' (int) is the number of actions available to each agent (all the agents have the same actions).
# # 'reward' (float) is the sum of all agents' rewards.
# # 'done' (bool) is False if at least an agent's done is False.
# # 'extra_info' (dict) typically is an empty dictionary.
# # 'info' (dict) contains only 'TimeLimit.truncated' (bool) which is False if at least an agent's truncated is False.
# # 'obs' (tuple) contains numpy arrays, each of which corresponds to an agent:
# #               a) In the case of encoding the images, the shape of each observation is (cnn_features_dim,),
# #                  as defined by the argument 'cnn_features_dim' (default is 128).
# #               b) In the case of raw images, the shape of each observation is (3, h, w).
# # 'state' (np.ndarray) is the concatenation of all observations:
# #                      a) In the case of encoding the images, the shape is (cnn_features_dim * n_agns,).
# #                      b) In the case of raw images, the shape is (n_agns, 3, h, w).

# # Example of arguments for PettingZoo.
# # Specifically:
#   - Butterfly (except from "Knights Archers Zombies"),
#   - Atari (only "Emtombed: Cooperative" and "Space Invaders")

fakeargs = {
  "env": "pettingzoo",
  "env_args": {
      "key": "pistonball_v6",
      "time_limit": 900,  # Episode horizon.
      "render_mode": "rgb_array",  # Options: "human", "rgb_array
      "image_encoder": "ResNet18",  # Options: "ResNet18", "SlimSAM", "CLIP"
      "image_encoder_use_cuda": True,  # Whether to load image-encoder in GPU or not.
      "image_encoder_batch_size": 10,  # How many images to encode in a single pass.
      "partial_observation": False,  # Only for "Emtombed: Cooperative" and "Space Invaders"
      "trainable_cnn": False,  # Specifies whether to return image-observation or the encoded vector-observation
      "kwargs": "",
      "seed": 2024
  },
  "num_episodes": 1,
  "agent": "rnn"
}

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
    if config["use_cuda"] and not torch.cuda.is_available():
        config["use_cuda"] = False
        print(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )
    if config["use_cuda_cnn_modules"] and not torch.cuda.is_available():
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



# Define async helpers to wrap the blocking IPC methods
async def async_send(conn, message):
    # Sends are usually fast, but we wrap it to keep the loop fluid
    return await asyncio.to_thread(conn.send, message)

async def async_recv(conn):
    # Offloads the blocking recv() to a background thread
    return await asyncio.to_thread(conn.recv)



def train_ippo():
    # 1a. Get config (TODO LETS CHECK ON SEED)
    _config = read_config()
    
    # check args sanity 
    _config = args_sanity_check(_config) #, _log)
    
    
    args = SN(**_config)
    
    args.device = "cuda" if args.use_cuda else "cpu"
    args.device_cnn_modules = "cuda" if args.use_cuda_cnn_modules else "cpu"
    torch.backends.cudnn.benchmark = True
    
    
    # Make subprocesses for the envs
    parent_conns, worker_conns = zip(*[Pipe() for _ in range(args.buffer_size)])
    
    # 1b. Create environment
    env_fn = env_REGISTRY[_config["env"]]#(**_config["env_args"])
    env_args = [args.env_args.copy() for _ in range(args.buffer_size)]
    
    for i in range(args.buffer_size):
        env_args[i]["seed"] += i

    processes = [
            Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, **env_arg)),
                    image_encoder
                )
            ) for env_arg, worker_conn in zip(env_args, worker_conns)
        ]

        for p in processes:
            p.daemon = True
            p.start()
    
    parent_conns[0].send(("get_print_info", None))
    time.sleep(5)  # Wait to init the environment
    print_info = parent_conns[0].recv()
    if print_info != "None" and print_info is not None:
        print("wow printinfo:")
        print(print_info)
    
    parent_conns[0].send(("get_env_info", None))
    env_info = parent_conns[0].recv()
    episode_limit = env_info["episode_limit"]
    
    
    
    max_episode_length = env_info["episode_limit"]

    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    observation_dimension = env.get_obs_size()
    
    args.n_agents = n_agents
    args.n_actions = n_actions
    
    args.num_iterations = args.t_max // args.buffer_size
    
    # 1c. Create videorecorder.
    rl_video_recorder = RLVideoRecorder(env)

    # 2. Create per-agent PPO learners
    agents = []
    for i in range(n_agents):
        learner = IndependentPPOLearner(
            obs_dimension=observation_dimension,
            act_dimension=n_actions,
            args=args
        )
        agents.append(learner)
        
        if args.use_cuda:
            learner.cuda_new()
        #buffers.append(AgentBuffer(max_steps=args["episode_limit"]))

    
    # 2a. Create Buffers
    device = next(agents[0].actor.parameters()).device
    
    print(device)
    
    obs_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, observation_dimension, device=device)
    rewards_buffer = torch.zeros(args.buffer_size, max_episode_length, device=device)
    dones_buffer = torch.zeros(args.buffer_size, max_episode_length, device=device)
    actions_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, device=device)
    values_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, device=device)
    
    print("obs_buffer.shape: " + str(obs_buffer.shape))
    
    t = 0
    t_environment = 0
    env_steps_this_run = 0
    episode_index = 0
    
    obs = torch.zeros(args.buffer_size, n_agents, observation_dimension, device=device)
    
    while t_environment < args.t_max:
        terminated = [False for _ in range(args.buffer_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        
        # Reset the envs
        #for parent_conn in parent_conns:
        #    parent_conn.send(("reset", None))
        #obs, state = env.reset()
        #done = False
        
        # 1. Reset all envs concurrently
        await asyncio.gather(*[async_send(p_conn, ("reset", None)) for p_conn in parent_conns])
        
        #obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        
        # Init hidden state agent: (memory of of this episode)
        for agent in agents:
            agent.init_hidden_buffer(buffer_size)
        
        #for b_index, parent_conn in enumerate(parent_conns):
        #    data = parent_conn.recv()
        #    obs[b_index] = torch.from_numpy(data["obs"]).to(device)
        
        init_data_samples = await asyncio.gather(*[async_recv(p_conn) for p_conn in parent_conns])
        for b_index, data in enumerate(init_data_samples):
            obs[b_index] = obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=device) #torch.from_numpy(data["obs"]).to(device)

        for step in range(0, max_episode_length):
            actions_all = []
            logp_all = []
            value_all = []
        
            with torch.inference_mode():
                for i in range(n_agents):
                    action, logp, _, value = agents[i].select_action(obs[:, i])
                    actions_all.append(action)
                    logp_all.append(logp)
                    value_all.append(value)
                    #actions[:, i] = action.cpu().numpy()
                    #log_probs.append(logp)
                    #log_probs = logp.cpu().numpy()
                actions = torch.stack(actions_all, dim=1).cpu().numpy()   # (B, n_agents)
                logprobs = torch.stack(logp_all, dim=1)
                values = torch.stack(value_all, dim=1)

                # store buffers (still fast, no CPU sync)
                actions_buffer[:, :, step] = actions
                logprobs_buffer[:, :, step] = logprobs
                values_buffer[:, :, step] = values
                obs_buffer[:, :, step] = obs
                                        
                    #actions_buffer[:, i, step] = action.detach()
                    #logprobs_buffer[:, i, step] = logp.detach()
                    #values_buffer[:, i, step] =  value.detach()
                    #obs_buffer[:, i, step] = obs[:, i]
    
            # 3b. Send step actions concurrently
            send_tasks = []
            for idx, parent_conn in enumerate(parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:
                        #parent_conn.send(("step", actions[idx]))
                        send_tasks.append(async_send(parent_conn, ("step", actions[idx])))
                    # Rendering
                    if idx == 0 and test_mode and args.render:
                        send_tasks.append(async_send(parent_conn, ("render", None)))

            if send_tasks:
                await asyncio.gather(*send_tasks)
            
            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            if all(terminated):
                break
                
            #4. Receive environment step updates concurrently
            # Create tasks only for active, non-terminated environments
            recv_tasks = [async_recv(parent_conns[b_idx]) for b_idx in envs_not_terminated]
        
            if recv_tasks:
                step_results = await asyncio.gather(*recv_tasks)

                # Process results sequentially once all concurrent recvs finish
                for b_index, data in zip(envs_not_terminated, step_results): #parent_conn in enumerate(parent_conns):
                    if not terminated[b_index]:
                        #data = parent_conn.recv()
                        obs[b_index] = torch.from_numpy(data["obs"]).to(device)
                        rewards_buffer[b_index, step] = data["reward"]
                        dones_buffer[b_index, step] = data["terminated"]
                        terminated[b_index] = data["terminated"]

                        env_steps_this_run += 1


            #reward, done, extra_info = env.step(actions)
            #obs = env.get_obs()
            #obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            
            #rewards_buffer[episode_index, step] = reward
            #dones_buffer[episode_index, step] = done
            
            if step % 250 == 0:
                print(step)
            t += 1
            
        
        #episode_index += 1
        t_environment += env_steps_this_run

        # 4. PPO update per agent (OUTSIDE OF EPISODE LOOP)
        #if episode_index == args.buffer_size:
        for i in range(n_agents):
            agents[i].update3(
                obs_buffer[:, i], 
                actions_buffer[:, i],
                logprobs_buffer[:, i],
                values_buffer[:, i],
                rewards_buffer,
                dones_buffer
            )
        #episode_index = 0
        env_steps_this_run = 0

        obs_buffer.zero_()
        actions_buffer.zero_()
        logprobs_buffer.zero_()
        values_buffer.zero_()
        rewards_buffer.zero_()
        dones_buffer.zero_()
        
    env.close()

def env_worker(remote, env_fn, image_encoder):
    # Make environment
    env = env_fn.x()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            #avail_actions = env.get_avail_actions()
            #state = env.get_state()
            obs = env.get_obs()
            #obs = torch.as_tensor(obs, dtype=torch.float32)

            if image_encoder is not None:
                # 'obs' is tuple with a single element - a dictionary of observations, so we keep only this
                obs = image_encoder.observation(obs[0])
                #state = np.concatenate(obs, axis=0).astype(np.float32)  # Concatenate the encoded observations (vectors)
            remote.send({
                # Data for the next timestep needed to pick an action
                #"state": state,
                #"avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            #avail_actions = env.get_avail_actions()
            #state = env.get_state()
            obs = env.get_obs()
            if image_encoder is not None:
                # 'obs' is tuple with a single element - a dictionary of observations, so we let it as is since
                # the observations are in this format when coming from reset
                obs = image_encoder.observation(obs)
                # Concatenate the encoded observations (vectors)
                state = np.concatenate(obs, axis=0).astype(np.float32)
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        elif cmd == "get_print_info":
            print_info = env.get_print_info()
            if print_info is None:
                remote.send("None")
            else:
                # Simulate the message format of the logger defined in _logging.py
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                print_info = f"\n[INFO {current_time}] parallel_runner " + print_info
                remote.send(print_info)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    train_ippo()


# #########################################################################################################
