# '''
# #
# # Reinforcement Learing Space
# #
# # - Each gym is a type of environment with its actions, rewards and states!
# #
# '''

from rc_gym.Utils.Utils import OrnsteinUhlenbeckAction
import gym
import rc_gym
import numpy as np
import time
import rc_gym.Utils

# Using penalty env
env = gym.make('VSS3v3-v0')
# env = gym.make('grSimSSLPenalty-v0')


def print_with_description(state):
    # print("Ball X = {}".format(state[0]))
    # print("Ball Y = {}".format(state[1]))
    # print("Ball Vx ={:.2f}".format(state[2]))
    # print("Ball Vy ={:.2f}".format(state[3]))
    print("id 0 Blue Robot X = {:.2f}".format(state[4]))
    print("id 0 Blue Robot Y = {:.2f}".format(state[5]))
    print("id 0 Blue Robot sin(theta) = {:.2f}".format(state[5]))
    print("id 0 Blue Robot cos(theta) = {:.2f}".format(state[6]))
    print("id 0 Blue Robot Vx =     {:.2f}".format(state[7]))
    print("id 0 Blue Robot Vy =     {:.2f}".format(state[8]))
    print("id 0 Blue Robot v_theta ={:.2f}".format(state[9]))
    # print("id 1 Blue Robot target_x = {:.2f}".format(state[14]))
    # print("id 1 Blue Robot target_y = {:.2f}".format(state[15]))
    # print("id 1 Blue Robot X = {:.2f}".format(state[16]))
    # print("id 1 Blue Robot Y = {:.2f}".format(state[17]))
    # print("id 1 Blue Robot sin(theta) = {:.2f}".format(state[18]))
    # print("id 1 Blue Robot cos(theta) = {:.2f}".format(state[19]))
    # print("id 1 Blue Robot Vx = {:.2f}".format(state[20]))
    # print("id 1 Blue Robot Vy = {:.2f}".format(state[21]))
    # print("id 1 Blue Robot v_theta = {:.2f}".format(state[22]))
    # print("id 2 Blue Robot target_x = {:.2f}".format(state[23]))
    # print("id 2 Blue Robot target_y = {:.2f}".format(state[24]))
    # print("id 2 Blue Robot X = {:.2f}".format(state[25]))
    # print("id 2 Blue Robot Y = {:.2f}".format(state[26]))
    # print("id 2 Blue Robot sin(theta) = {:.2f}".format(state[27]))
    # print("id 2 Blue Robot cos(theta) = {:.2f}".format(state[28]))
    # print("id 2 Blue Robot Vx = {:.2f}".format(state[29]))
    # print("id 2 Blue Robot Vy = {:.2f}".format(state[30]))
    # print("id 2 Blue Robot v_theta = {:.2f}".format(state[31]))
    # print("id 0 Yellow Robot X = {:.2f}".format(state[32]))
    # print("id 0 Yellow Robot Y = {:.2f}".format(state[33]))
    # print("id 0 Yellow Robot sin(theta) = {:.2f}".format(state[34]))
    # print("id 0 Yellow Robot cos(theta) = {:.2f}".format(state[35]))
    # print("id 0 Yellow Robot Vx = {:.2f}".format(state[36]))
    # print("id 0 Yellow Robot Vy = {:.2f}".format(state[37]))
    # print("id 0 Yellow Robot v_theta = {:.2f}".format(state[38]))
    # print("id 1 Yellow Robot X = {:.2f}".format(state[39]))
    # print("id 1 Yellow Robot Y = {:.2f}".format(state[40]))
    # print("id 1 Yellow Robot sin(theta) = {:.2f}".format(state[41]))
    # print("id 1 Yellow Robot cos(theta) = {:.2f}".format(state[42]))
    # print("id 1 Yellow Robot Vx = {:.2f}".format(state[43]))
    # print("id 1 Yellow Robot Vy = {:.2f}".format(state[44]))
    # print("id 1 Yellow Robot v_theta = {:.2f}".format(state[45]))
    # print("id 2 Yellow Robot X = {:.2f}".format(state[46]))
    # print("id 2 Yellow Robot Y = {:.2f}".format(state[47]))
    # print("id 2 Yellow Robot sin(theta) = {:.2f}".format(state[48]))
    # print("id 2 Yellow Robot cos(theta) = {:.2f}".format(state[49]))
    # print("id 2 Yellow Robot Vx = {:.2f}".format(state[50]))
    # print("id 2 Yellow Robot Vy = {:.2f}".format(state[51]))
    # print("id 2 Yellow Robot v_theta = {:.2f}".format(state[52]))
    
# env.reset()
# Run for 10 episode and print reward at the end
for i in range(10):
    done = False
    next_state = env.reset()
    # env.render()
    # print_with_description(next_state)
    epi_rew = 0.
    while not done:
        action = np.array([0., 0.])
        # action = env.action_space.sample()
        # time.sleep(0.1)
        next_state, reward, done, _ = env.step(action)
        epi_rew += reward
        # print_with_description(next_state)
        env.render()

env.close()
