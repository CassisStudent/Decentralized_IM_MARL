import time
from functools import partial
import datetime
import numpy as np
import torch as th
from torch.multiprocessing import Pipe, Process

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.utils.image_encoder import ImageEncoder
from pymarlzooplus.utils.env_utils import check_env_installation


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class DecentralizedRunner:

    def __init__(self, args, logger):

        # Check if the requirements for the selected environment are installed
        check_env_installation(args.env, env_REGISTRY, logger)

        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Enable multithreading access to GPU
        # Check if the start method is set to spawn, if not set it to spawn
        if th.multiprocessing.get_start_method(allow_none=True) is None:
            th.multiprocessing.set_start_method('spawn')

        # In case of pettingzoo and centralized image encoding, initialize image encoder here
        image_encoder = None
        if self.args.env == 'pettingzoo' and self.args.env_args['centralized_image_encoding'] is True:
            image_encoder_args = [
                "parallel_runner",
                self.args.env_args['centralized_image_encoding'],
                self.args.env_args['trainable_cnn'],
                self.args.env_args['image_encoder'],
                self.args.env_args['image_encoder_batch_size'],
                self.args.env_args['image_encoder_use_cuda']
            ]
            image_encoder = ImageEncoder(*image_encoder_args)
            image_encoder.share_memory()  # Make model parameters shareable across processes
            self.args.env_args['given_observation_space'] = image_encoder.observation_space
            self.logger.console_logger.info(image_encoder.print_info)

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [
            Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, **env_arg)),
                    image_encoder
                )
            ) for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        # Get info from environment to be printed
        self.parent_conns[0].send(("get_print_info", None))
        time.sleep(5)  # Wait a little to initialize the environment and get the print info
        print_info = self.parent_conns[0].recv()
        
        if print_info != "None" and print_info is not None:
            self.logger.console_logger.info(print_info)
        
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t_env = 0

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        
        self.t_env = 0

        return pre_transition_data
    def run(self, test_mode=False):
        print("Dit doet het niet meer in decentralized dingen")
    
    
    def step(self, actions_batch):
        """
        actions_batch: np.array of shape [batch_size, n_agents] (of wat jouw env verwacht)
        Stuurt actions naar alle envs en geeft per-env data terug.
        """
        # actions naar envs sturen
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("step", actions_batch[idx]))

        rewards = []
        dones = []
        infos = []
        next_states = []
        next_obs_n = []
        next_avail_actions_n = []

        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            rewards.append(data["reward"])
            dones.append(data["terminated"])
            infos.append(data["info"])
            next_states.append(data["state"])
            next_avail_actions_n.append(data["avail_actions"])
            next_obs_n.append(data["obs"])

        self.t_env += len(self.parent_conns)

        return {
            "state": np.stack(next_states, axis=0),
            "obs": np.stack(next_obs_n, axis=0),
            "avail_actions": np.stack(next_avail_actions_n, axis=0),
            "reward": np.array(rewards),
            "done": np.array(dones),
            "info": infos,
        }
    
    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn, image_encoder):
    # Make environment
    env = env_fn.x()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            avail_actions = env.get_avail_actions()
            state = env.get_state()
            obs = env.get_obs()
            if image_encoder is not None:
                # 'obs' is tuple with a single element - a dictionary of observations, so we keep only this
                obs = image_encoder.observation(obs[0])
                state = np.concatenate(obs, axis=0).astype(np.float32)  # Concatenate the encoded observations (vectors)
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            avail_actions = env.get_avail_actions()
            state = env.get_state()
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

