# '''
# #
# # Reinforcement Learing Space
# # 
# # - Each gym is a type of environment with its actions, rewards and states!
# #
# '''

import gym
import gym_ssl

# Using penalty env
env = gym.make('grSimSSLPenalty-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
    print(reward)