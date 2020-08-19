# '''
# #
# # Reinforcement Learing Space
# # 
# # - Each gym is a type of environment with its actions, rewards and states!
# #
# '''
# import gym
# import gym_ssl


# env = gym.make('grSimSSLPenalty-v0')

# env.reset()
# for i in range(1):
#     done = False
#     reward = 0
#     while reward == 0:
#         action = env.action_space.sample()
#         next_state, reward, done, _ = env.step(0.0)
#         # print(next_state[0], next_state[1])
#         if done:
#             print(reward)
#             env.reset()



# #To test comm, uncomment the following line and comment lines above
# # while(True):
# #    print(env.client.receiveState().ball)

import gym
import gym_ssl
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

env = gym.make('grSimSSLPenalty-v0')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=2000000)
model.save("ddpg_mountain")

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_mountain")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    
    if dones:
        env.reset()
