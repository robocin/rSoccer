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
for i in range(10):
    done = False
    env.reset()
    while not done:
        action = np.array([1, 1])
        next_state, reward, done, _ = env.step(action)
        env.render()
        # print(np.sqrt((env.frame.robots_blue[0].v_x * env.frame.robots_blue[0].v_x) + (env.frame.robots_blue[0].v_y * env.frame.robots_blue[0].v_y)))
        # print(env.frame.robots_blue[0].v_theta)
env.close()

