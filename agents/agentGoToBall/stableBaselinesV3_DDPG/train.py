import gym
import gym_ssl
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make('grSimSSLGoToBall-v0')

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log='./ddpg_go_to_ball_tensorboard')
model.learn(total_timesteps=100, log_interval=10)
model.save("./models/gotoball3")
env = model.get_env()

print("Done!")

