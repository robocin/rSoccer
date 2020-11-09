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
import time

# Using penalty env
env = gym.make('VSS3v3-v0')
# env = gym.make('grSimSSLPenalty-v0')


env.reset()
# Run for 10 episode and print reward at the end
for i in range(1):
    done = False
    env.reset()
    while not done:
        action = np.array([0.9, 0.9])
        next_state, reward, done, _ = env.step(action)
        print(env.frame.robots_blue[0].theta)
env.close()

