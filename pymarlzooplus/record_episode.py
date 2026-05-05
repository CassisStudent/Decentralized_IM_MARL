import imageio
import numpy as np
import torch as th

from pymarlzooplus.envs import REGISTRY as env_REGISTRY
from pymarlzooplus.controllers import REGISTRY as mac_REGISTRY
from pymarlzooplus.utils.logging_setup import get_logger

logger = get_logger("record")

def record_episode(config_path, checkpoint_path, out_file="episode.mp4"):
    # 1. Load config
    import yaml
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 2. Create environment with rgb_array mode
    env_args = config["env_args"]
    env_args["render_mode"] = "rgb_array"
    env = env_REGISTRY[config["env"]](**env_args)

    # 3. Load trained model
    mac = mac_REGISTRY[config["mac"]](config)
    checkpoint = th.load(checkpoint_path, map_location="cpu")
    mac.load_state(checkpoint["mac"])

    # 4. Run one episode
    frames = []
    env.reset()
    mac.init_hidden(batch_size=1)

    terminated = False
    while not terminated:
        obs = env.get_obs()
        obs = th.tensor(np.array([obs]), dtype=th.float32)

        actions = mac.select_actions(obs, t=0, test_mode=True).cpu().numpy()[0]
        reward, terminated, info = env.step(actions)

        frame = env.render()
        frames.append(frame)

    # 5. Save video
    imageio.mimsave(out_file, frames, fps=20)
    print(f"Saved video to {out_file}")
