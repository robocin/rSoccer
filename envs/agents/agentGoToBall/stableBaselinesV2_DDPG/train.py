import gym
import rc_gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG


env = gym.make('grSimSSLGoToBall-v0')

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions), theta=float(0.15))

model = DDPG(MlpPolicy, env, verbose=1, actor_lr = 0.001, critic_lr = 0.0001, normalize_observations=True, param_noise=param_noise, action_noise=action_noise, memory_limit=100000)
model.learn(total_timesteps=100000, log_interval=10)

model.save("./models/ddpg_gotoball2")
env = model.get_env()

print("Done!")