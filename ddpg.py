import gym
import gym_ssl
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

env = gym.make('grSimSSLShootGoalie-v0')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions), theta=float(0.15))

model = DDPG(MlpPolicy, env, verbose=1, actor_lr = 0.001, critic_lr = 0.0001, param_noise=param_noise, action_noise=action_noise)

for i in range(100):
    model.learn(total_timesteps=100000)
    model.save("ddpg_shootgoalie")

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_shootgoalie")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()