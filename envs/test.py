# '''
# #
# # Reinforcement Learing Space
# # 
# # - Each gym is a type of environment with its actions, rewards and states!
# #
# '''

import gym
import rc_gym
import numpy as np

# Using penalty env
env = gym.make('rSimVSS3v3-v0')
# env = gym.make('grSimSSLPenalty-v0')


env.reset()
# Run for 10 episode and print reward at the end
for i in range(1000):
    done = False
    env.reset()
    while not done:
        action = np.array([0.99, 0.99])
        next_state, reward, done, _ = env.step(action)
        env.render()

