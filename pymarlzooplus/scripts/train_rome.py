# Import packages
import yaml
import os
import sys
import torch
from types import SimpleNamespace as SN

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.rl_video_recorder import RLVideoRecorder
from pymarlzooplus.learners.independent_ppo_learner import IndependentPPOLearner
from pymarlzooplus.utils.logging_setup import Logger





##################################### API for fully cooperative tasks. ###################################
## 'n_agns' (int) is the number of agents in the environment.
## 'n_acts' (int) is the number of actions available to each agent (all the agents have the same actions).
## 'reward' (float) is the sum of all agents' rewards.
## 'done' (bool) is False if at least an agent's done is False.
## 'extra_info' (dict) typically is an empty dictionary.
## 'info' (dict) contains only 'TimeLimit.truncated' (bool) which is False if at least an agent's truncated is False.
## 'obs' (tuple) contains numpy arrays, each of which corresponds to an agent:
##               a) In the case of encoding the images, the shape of each observation is (cnn_features_dim,),
##                  as defined by the argument 'cnn_features_dim' (default is 128).
##               b) In the case of raw images, the shape of each observation is (3, h, w).
## 'state' (np.ndarray) is the concatenation of all observations:
##                      a) In the case of encoding the images, the shape is (cnn_features_dim * n_agns,).
##                      b) In the case of raw images, the shape is (n_agns, 3, h, w).

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

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config

def train_ippo():
    # 1a. Get config (TODO LETS CHECK ON SEED)
    _config = read_config()
    
    # check args sanity 
    #_config = args_sanity_check(_config) #, _log)
    
    
    args = SN(**_config)
    
    # 1b. Create environment
    env = env_REGISTRY[_config["env"]](**_config["env_args"])
    #env = env_REGISTRY[args["env"]](**args["env_args"])

    n_agents = env.get_n_agents()
    n_actions = env.get_total_actions()
    observation_dimension = env.get_obs_size()
    
    args.n_agents = n_agents
    args.n_actions = n_actions
    
    
    print(n_actions)
    print(args)
    
    
    # 1c. Create videorecorder.
    rl_video_recorder = RLVideoRecorder(env)

    # 2. Create per-agent PPO learners
    agents = []
    #buffers = []
    
    # 2a. Create Buffers
    
    #Assums no partial observation and rewards are shared among agents
    obs_buffer = torch.zeros(args.buffer_size, observation_dimension)
    rewards_buffer = torch.zeros(args.buffer_size)
    dones_buffer = torch.zeros(args.buffer_size)
    actions_buffer = torch.zeros(args.buffer_size, n_agents)
    logprobs_buffer = torch.zeros(args.buffer_size, n_agents)
    values_buffer = torch.zeros(args.buffer_size, n_agents)
    
    print("obs_buffer.shape: " + str(obs_buffer.shape))

    for i in range(n_agents):
        learner = IndependentPPOLearner(
            obs_dimension=observation_dimension,
            act_dimension=n_actions,
            args=args
        )
        agents.append(learner)
        #buffers.append(AgentBuffer(max_steps=args["episode_limit"]))
    
    # 3. Training loop
    for episode in range(args.num_episodes):
    
        # Reset the environment
        obs, state = env.reset()
        done = False
        
        # Init hidden state agent: (memory of of this episode)
        for agent in agents:
            agent.init_hidden()
        
        print(len(obs))
        print("number of agents " + str(n_agents)) 

        obs_buffer[t] = torch.tensor(obs)

        
        # Run an episode
        for step in range(0, args.buffer_size):
            # Render the environment (optional)
            #rl_video_recorder.record_video()

            # 3a. Select actions per agent
            actions = []
            log_probs = []
            
            obs_buffer[step] = obs
            dones_buffer[step] = done
            
            with torch.no_grad():
                for i in range(n_agents):
                    action, logp, value = agents[i].select_action(obs[i])
                    actions.append(action)
                    log_probs.append(logp)

                    actions_buffer[step, i] = action
                    logprobs_buffer[step, i] = logp
                    values_buffer[step, i] = value
    
            # 3b. Step environment
            reward, done, extra_info = env.step(actions)
            obs = env.get_obs()
            
            rewards_buffer[step] = reward
            
            print(step)
            # optional: maybe replace this with env.get_obs()
            #state = env.get_state()
            
            # optional: Additional Info about the environment we might need
            #info = env.get_info()
            if done is True or step > 100:
                break
                
        # 4. PPO update per agent (OUTSIDE OF EPISODE LOOP)
        for i in range(n_agents):
            agents[i].update(obs_buffer
            """
            batch = buffers[i].to_batch()
            agents[i].update(batch)
            buffers[i].reset()
            """
        # Save evaluation video
        #rl_video_recorder.save_video()
        
    # Terminate the environment
    env.close()

    
if __name__ == '__main__':
    train_ippo()
    

##########################################################################################################
