'''
#
# Reinforcement Learing Space
# 
# - Each gym is a type of environment with its actions, rewards and states!
#
'''
import gym
import gym_ssl


env = gym.make('grSimSSLPenalty-v0')

env.reset()
for i in range(1):
    done = False
    reward = 0
    while reward == 0:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(0.0)
        if done:
            env.reset()



#To test comm, uncomment the following line and comment lines above
# while(True):
#    print(env.client.receiveState().ball)