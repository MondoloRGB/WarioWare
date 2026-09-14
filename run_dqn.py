import os
import retro
import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from gymnasium.wrappers import ResizeObservation

# Microgame wrappers.
from crazycars_wrapper import CrazyCarsRewardWrapper
from dodgeballs_wrapper import DodgeBallsRewardWrapper
from mugshot_wrapper import MugShotRewardWrapper
from superwariobros_wrapper import SuperWarioBrosRewardWrapper

# Base path used to find the rest of the integration Files.
base_path = "/home/mondolo/stable-retro/retro/data/contrib/WarioWareIncMegaMicrogames-GbAdvance"



# Converts MultiBinary input (12 for GBA) into a smaller Discrete action space.
class Discretizer(gym.Wrapper):
    def __init__(self, env, combos):
        super().__init__(env)
        self._combos = combos
        self.action_space = gym.spaces.Discrete(len(combos))
        self.button_names = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", None, "L", "R"]

    def action(self, act):
        return np.array(self._combos[act], dtype=np.int8)

    def step(self, act):
        obs, reward, terminated, truncated, info = self.env.step(self.action(act))
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



# Prints reward information in real time to the WSL terminal.
class RewardPrinterCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos and isinstance(infos, list) and len(infos) > 0:
            info = infos[0]
            if "reward" in info:
                r = info["reward"]
                if abs(r) > 0.05:
                    print(f"Reward event: {r}")
        return True



# Detects Game Over screen by brightness (dark frame) and resets automatically.
class GameOverResetWrapper(gym.Wrapper):
    def __init__(self, env, threshold=5):
        super().__init__(env)
        self.threshold = threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        brightness = np.mean(obs)
        if brightness < self.threshold:
            terminated = True
        return obs, reward, terminated, truncated, info



# Creates the base Retro environment.
def create_base_env(microgame_name):
    state_path = os.path.join(base_path, f"{microgame_name}/{microgame_name}.state")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Microgame state not found: {state_path}")

    env = retro.make(
        game="WarioWareIncMegaMicrogames-GbAdvance",
        state=state_path,
        inttype=retro.data.Integrations.CONTRIB,
        render_mode=None, # "human" if you want to render gameplay
    )

    # Allowed action combos. A + RIGHT and A + LEFT were added in case "Repellion" was used.
    simple_actions = [
        [0]*12,                      # NONE
        [0,0,0,0,0,0,0,0,1,0,0,0],   # A
        [0,0,0,0,0,0,0,1,0,0,0,0],   # RIGHT
        [0,0,0,0,0,0,1,0,0,0,0,0],   # LEFT
        [0,0,0,0,0,1,0,0,0,0,0,0],   # DOWN
        [0,0,0,0,1,0,0,0,0,0,0,0],   # UP
        [0,0,0,0,0,0,0,1,1,0,0,0],   # A + RIGHT
        [0,0,0,0,0,0,1,0,1,0,0,0],   # A + LEFT
    ]
    # Observation preprocessing
    env = ResizeObservation(env, shape=(100, 100))

    # Environment action and reset wrappers
    env = GameOverResetWrapper(env)
    env = Discretizer(env, simple_actions)


    # Reward shaping wrappers. Change to whichever microgame you wish to train.
    env = CrazyCarsRewardWrapper(env)
    # env = DodgeBallsRewardWrapper(env)
    # env = MugShotRewardWrapper(env)
    # env = SuperWarioBrosRewardWrapper(env)

    env = Monitor(env)

    return env



# Create a vectorized environment for CNN input.
def create_image_env(microgame_name, n_stack=4):
    env = DummyVecEnv([lambda: create_base_env(microgame_name)])
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)
    return env



# Train DQN from Stable Baselines 3 using CnnPolicy.
# Default microgame, runs, and TPR is "Crazy Cars", 12, and 50,000 respectively.
def train_dqn(microgame_name="CrazyCars", runs=12, timesteps_per_run=50000):
    env = create_image_env(microgame_name)
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log=f"./dqn_{microgame_name.lower()}_tb/",
        policy_kwargs={"normalize_images": True}
    )

    total_steps = 0
    print(f"Starting {runs} sequential training run(s) for '{microgame_name}'...\n")

    for i in range(runs):
        print(f"=== Run {i + 1}/{runs} ===")
        model.learn(total_timesteps=timesteps_per_run, reset_num_timesteps=False)
        total_steps += timesteps_per_run
        print(f"Completed {total_steps} total timesteps\n")

    # Save the final model
    model.save(f"dqn_{microgame_name.lower()}")
    print(f"Training complete. Model saved as dqn_{microgame_name.lower()}.zip ({total_steps} timesteps total)\n")
    env.close()



# Exactly what it sounds like.
def watch_trained_agent(microgame_name):
    env = create_image_env(microgame_name)
    model = DQN.load(f"dqn_{microgame_name.lower()}", env=env)
    obs = env.reset()
    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()



# Main, which executes upon running "python3 run_dqn.py" in the WSL terminal.
# watch_trained_agent() loads the final model.
if __name__ == "__main__":
    train_dqn("CrazyCars")
    # watch_trained_agent("CrazyCars")