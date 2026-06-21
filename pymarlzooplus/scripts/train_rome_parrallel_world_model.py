# Import packages
import yaml
import os
import sys
import torch
import time
from types import SimpleNamespace as SN
import numpy as np
import datetime

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.rl_video_recorder import RLVideoRecorder
from pymarlzooplus.learners.independent_ppo_world_learner import IndependentPPOWorldLearner
from pymarlzooplus.utils.logging_setup import Logger

from torch.utils.tensorboard import SummaryWriter

from torch.multiprocessing import Process, Pipe
from functools import partial
import time
import copy




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


"""
# Define async helpers to wrap the blocking IPC methods
async def async_send(conn, message):
    # Sends are usually fast, but we wrap it to keep the loop fluid
    return await asyncio.to_thread(conn.send, message)

async def async_recv(conn):
    # Offloads the blocking recv() to a background thread
    return await asyncio.to_thread(conn.recv)
"""


def train_ippo():
    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method('spawn')
    
    #-----------Logging-------------------:
    log_dir = os.path.join("results", "tb_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # 1a. Get config (TODO LETS CHECK ON SEED)
    _config = read_config()
    
    # check args sanity 
    _config = args_sanity_check(_config) #, _log)
    
    
    args = SN(**_config)
    
    num_gpus = torch.cuda.device_count() if args.use_cuda else 1
    if num_gpus > 8: 
        num_gpus = 8 # Beperk tot je 8 x A16 setup
    
    args.device = "cuda" if args.use_cuda else "cpu"
    if args.use_cuda and num_gpus > 1:
        args.device = "cuda:1"

    args.device_cnn_modules = "cuda" if args.use_cuda_cnn_modules else "cpu"
    torch.backends.cudnn.benchmark = True
    
    
    # Make subprocesses for the envs
    parent_conns, worker_conns = zip(*[Pipe() for _ in range(args.buffer_size)])
    
    # 1b. Create environment
    env_fn = env_REGISTRY[_config["env"]]#(**_config["env_args"])
    env_args = [copy.deepcopy(args.env_args) for _ in range(args.buffer_size)]
    #env_args = [args.env_args.copy() for _ in range(args.buffer_size)]
    
    for i in range(args.buffer_size):
        env_args[i]["seed"] += i

    env_info = probe_env(env_fn, env_args[0])
    
    episode_limit = env_info["episode_limit"]
    max_episode_length = env_info["episode_limit"]
    n_agents = env_info["n_agents"]
    n_actions = env_info["n_actions"]
    observation_dimension = env_info["obs_shape"]
    
    args.n_agents = n_agents
    args.n_actions = n_actions
    args.num_iterations = args.t_max // args.buffer_size
    

    shared_obs = torch.zeros(args.buffer_size, n_agents, observation_dimension, dtype=torch.float32).pin_memory().share_memory_()
    shared_rewards = torch.zeros(args.buffer_size, dtype=torch.float32).pin_memory().share_memory_()
    shared_dones = torch.zeros(args.buffer_size, dtype=torch.bool).pin_memory().share_memory_()
    
    #image_encoder = None
    encoder_cfg = None
    processes = [
        Process(
            target=env_worker,
            args=(
                i,
                worker_conn,
                CloudpickleWrapper(partial(env_fn, **env_arg)),
                encoder_cfg,
                shared_obs,
                shared_rewards,
                shared_dones
            )
        ) for i, (env_arg, worker_conn) in enumerate(zip(env_args, worker_conns))
    ]
    for p in processes:
        p.daemon = True
        p.start()
    

    
    # 1c. Create videorecorder.
    #rl_video_recorder = RLVideoRecorder(env)

    # 2. Create per-agent PPO learners
    agents = []
    
    
    
    for i in range(n_agents):
        if num_gpus > 1:
            # Sla GPU 0 over! We verdelen over cuda:1 t/m cuda:3 (of t/m cuda:7 als je meer agents hebt)
            target_gpu_id = 1 + (i % (num_gpus - 1))
            target_device = f"cuda:{target_gpu_id}"
        else:
            target_device = "cuda:0"
        
        #target_device = f"cuda:{i % num_gpus}" if args.use_cuda else "cpu"
        
        print(f"Initializing Agent {i} on device: {target_device}")
        
        learner = IndependentPPOWorldLearner(
            obs_dimension=observation_dimension,
            act_dimension=n_actions,
            args=args,
            device=target_device
        )
        agents.append(learner)
    
    # 2a. Create Buffers
    device = next(agents[0].actor.parameters()).device
    print(device)
    
    obs_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, observation_dimension, device=device)
    rewards_buffer = torch.zeros(args.buffer_size, max_episode_length, device=device)
    dones_buffer = torch.zeros(args.buffer_size, max_episode_length, device=device)
    actions_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, device=device)
    values_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, device=device)
    
    hidden_states_buffer = torch.zeros(args.buffer_size, n_agents, max_episode_length, args.hidden_dim).pin_memory()
    
    obs = torch.zeros(args.buffer_size, n_agents, observation_dimension, device=device)

    print("obs_buffer.shape: " + str(obs_buffer.shape))
  
    t = 0
    t_environment = 0
    env_steps_this_run = 0
    episode_index = 0
    
    model_save_time = 0
    last_log_T = 0
    
    while t_environment < args.t_max:
        terminated = torch.zeros(args.buffer_size, dtype=torch.bool)
        
        # Reset the envs
        for conn in parent_conns:
            conn.send(("reset", None))

        for conn in parent_conns:
            conn.recv()
            
        torch.cuda.synchronize()
        obs = shared_obs.to(device)
            
        hidden_states = [agent.actor.init_hidden().expand(args.buffer_size, -1).contiguous().to(agent.device) for agent in agents]

        total_inference_time = 0
        total_env_comm_time = 0
        total_env_recv_time = 0
        total_buffer_time = 0

        for step in range(0, max_episode_length):
            actions_all = []
            logp_all = []
            value_all = []
            
            t_start_inf = time.perf_counter()
        
            hidden_states_cpu = torch.stack([h.cpu() for h in hidden_states], dim=1)
            hidden_states_buffer[:, :, step] = hidden_states_cpu

            # Select an action and save in buffers
            with torch.inference_mode():
                for i in range(n_agents):
                    action, logp, _, value, next_hidden = agents[i].select_action(obs[:, i], hidden_states[i])
                    hidden_states[i] = next_hidden
                    
                    actions_all.append(action.cpu())
                    logp_all.append(logp.cpu())
                    value_all.append(value.cpu())
                
                actions_tensor = torch.stack(actions_all, dim=1)
                logprobs = torch.stack(logp_all, dim=1)
                values = torch.stack(value_all, dim=1).squeeze(-1)
                
                actions_buffer[:, :, step] = actions_tensor.to(device)
                logprobs_buffer[:, :, step] = logprobs.to(device)
                values_buffer[:, :, step] = values.to(device)
                obs_buffer[:, :, step] = obs.to(device)
                
                actions = actions_tensor.numpy() #.cpu().numpy()   # (B, n_agents)

            torch.cuda.synchronize()
            total_inference_time += (time.perf_counter() - t_start_inf)

            active_envs = []
            t_start_env = time.perf_counter()
            
            #step in each environment
            for idx, parent_conn in enumerate(parent_conns):
                if not terminated[idx]:
                    parent_conn.send(("step", actions[idx]))
                    active_envs.append(idx)

            total_env_comm_time += (time.perf_counter() - t_start_env)
            
            #Receive all the info from the step
            t_start_recv = time.perf_counter()
            for idx in active_envs:
                parent_conns[idx].recv()
                
            total_env_recv_time += (time.perf_counter() - t_start_recv)
    
            t_start_buffer = time.perf_counter()

            #Safe information in buffers
            obs = shared_obs.to(device, non_blocking=True)
            rewards_buffer[:, step] = shared_rewards
            dones_buffer[:, step] = shared_dones
            terminated |= shared_dones
            
            env_steps_this_run += len(active_envs)
        
            total_buffer_time += (time.perf_counter() - t_start_buffer)

            if terminated.all():
                break
            
            if step % 250 == 0:
                print(step)
            t += 1
            
        print("\n========== TIMING RAPPORT PROFILER ==========")
        print(f"1. Model Inference Tijd:     {total_inference_time:.4f} sec (Gemiddeld: {total_inference_time/max_episode_length:.4f} per stap)")
        print(f"2. Omgeving (Send/Recv) Tijd: {total_env_comm_time:.4f} sec (Gemiddeld: {total_env_comm_time/max_episode_length:.4f} per stap)")
        print(f"3. RECV Verwerking:  {total_env_recv_time:.4f} sec (Gemiddeld: {total_env_recv_time/max_episode_length:.4f} per stap)")
        print(f"4. Buffer & Data Verwerking:  {total_buffer_time:.4f} sec (Gemiddeld: {total_buffer_time/max_episode_length:.4f} per stap)")
        print({
          "inf": total_inference_time,
          "recv": total_env_recv_time,
          "buf": total_buffer_time,
          "comm": total_env_comm_time
        })
        print("=============================================\n")
    
        with torch.inference_mode():
            # gemiddelde van 32 omgevingen
            mean_episode_reward = rewards_buffer.sum(dim=1).mean().item()
            
        print(f"GEMIDDELDE EPISODIC RETURN (REWARDS): {mean_episode_reward:.2f}")
        print(f"======================================================\n")

        
        #episode_index += 1
        t_environment += env_steps_this_run

        #---------------UPDATING-----------------------
        # 4. PPO update per agent (OUTSIDE OF EPISODE LOOP)
        #if episode_index == args.buffer_size:
        for i in range(n_agents):
            agents[i].update4(
                obs_buffer[:, i], 
                actions_buffer[:, i],
                logprobs_buffer[:, i],
                values_buffer[:, i],
                rewards_buffer,
                dones_buffer,
                hidden_states_buffer[:, i]
            )
        #episode_index = 0
        env_steps_this_run = 0

        #resetting buffers
        obs_buffer.zero_()
        actions_buffer.zero_()
        logprobs_buffer.zero_()
        values_buffer.zero_()
        rewards_buffer.zero_()
        dones_buffer.zero_()
        hidden_states_buffer.zero_()
        
        
        # --- MODEL SAVING ---
        if args.save_model and (t_environment - model_save_time >= args.save_model_interval or t_environment >= args.t_max):
            model_save_time = t_environment
    
            save_path = os.path.join("results", "models", str(t_environment))
            os.makedirs(save_path, exist_ok=True)

            for i in range(n_agents):
                agents[i].save_models(save_path, agent_id=i)
            print(f"[CHECKPOINT] Modellen succesvol opgeslagen bij stap {t_environment}")

        
        # ----- LOGGING --------
        if (t_environment - last_log_T) >= args.log_interval or t_environment >= args.t_max:
            last_log_T = t_environment
            
            writer.add_scalar("Train/Mean_Episode_Return", mean_episode_reward, t_environment)
            
            
            for i in range(n_agents):
                agent = agents[i]
                if hasattr(agent, 'last_pg_loss'):
                    writer.add_scalar(f"Agent_{i}/Policy_Loss", agent.last_pg_loss, t_environment)
                    writer.add_scalar(f"Agent_{i}/Actor_Loss", agent.last_actor_loss, t_environment)
                    writer.add_scalar(f"Agent_{i}/Critic_Loss", agent.last_critic_loss, t_environment)
                    writer.add_scalar(f"Agent_{i}/Entropy", agent.last_entropy, t_environment)
                    writer.add_scalar(f"Agent_{i}/Mean_Ratio", agent.last_mean_ratio, t_environment)
                    writer.add_scalar(f"Agent_{i}/Grad_Norm_Actor", agent.last_grad_norm_actor, t_environment)
                    writer.add_scalar(f"Agent_{i}/Approx_KL", agent.last_kl, t_environment)
                    
                    writer.add_scalar(f"Agent_{i}/Mean_Advantage", agent.last_mean_advantage, t_environment)
                    writer.add_scalar(f"Agent_{i}/STD_Advantage", agent.last_std_advantage, t_environment)
                    
                    writer.add_scalar(f"Agent_{i}/Explained_Variance", agent.last_explained_var, t_environment)
                    
                    writer.add_scalar(f"Agent_{i}/World_Model_Loss", agent.last_wm_loss, t_environment)
                    writer.add_scalar(f"Agent_{i}/Intrinsic_Reward_Raw_Mean", agent.last_intrinsic_raw_mean, t_environment)
                    writer.add_scalar(f"Agent_{i}/Intrinsic_Reward_Raw_Std", agent.last_intrinsic_raw_std, t_environment)
                    writer.add_scalar(f"Agent_{i}/Beta", agent.last_beta, t_environment)

            writer.flush()
            print(f"[LOG] Alle metrics (inclusief Explained Variance & Grad Norm) weggeschreven bij stap {t_environment}")
    
    
    #Final save
    final_save_path = os.path.join("results", "models", "final")
    os.makedirs(final_save_path, exist_ok=True)

    for i in range(n_agents):
        agents[i].save_models(final_save_path, agent_id=i)
    print(f"Training voltooid! Eindmodellen succesvol opgeslagen in: {final_save_path}")
        
    
    
    for conn in parent_conns:
        conn.send(("close", None))

    for p in processes:
        p.join()




def env_worker( worker_id, remote, env_fn, encoder_cfg, shared_obs, shared_rewards, shared_dones):
    # Make environment
    env = env_fn.x()
    
    # create encoder locally per process
    image_encoder = None
    if encoder_cfg is not None:
        image_encoder = ImageEncoder(
            model_type=encoder_cfg["type"],
            device="cuda" if encoder_cfg["use_cuda"] else "cpu"
        )

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            obs = env.get_obs()

            if image_encoder is not None:
                obs = image_encoder.observation(obs[0])
            
            obs = np.asarray(obs, dtype=np.float32)
            
            # WRITE DIRECTLY INTO SHARED MEMORY
            shared_obs[worker_id].copy_(
                torch.from_numpy(obs)
            )
            
            shared_rewards[worker_id] = reward
            shared_dones[worker_id] = terminated
             # tiny sync signal only
            remote.send(True)

        elif cmd == "reset":
            env.reset()
            obs = env.get_obs()

            if image_encoder is not None:
                obs = image_encoder.observation(obs)
            
            obs = np.asarray(obs, dtype=np.float32)
            
            shared_obs[worker_id].copy_(
                torch.from_numpy(obs)
            )
            shared_rewards[worker_id] = 0.0
            shared_dones[worker_id] = False

            remote.send(True)

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


def make_env(env_fn, env_arg):
    return env_fn(**env_arg)


def probe_env(env_fn, env_arg):
    env = make_env(env_fn, env_arg)
    env_info = env.get_env_info()
    env.close()
    return env_info


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

            

if __name__ == '__main__':
    train_ippo()


# #########################################################################################################
