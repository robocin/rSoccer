'''
#
# Reinforcement Learing Space
# 
# - Each gym is a type of environment with its actions, rewards and states!
#
'''
import gym
import gym_ssl

env = gym.make('grSimSSL-v0')

env.reset()
for i in range(1):
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print(next_state)