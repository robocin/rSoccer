import gym
import gym_ssl
import numpy as np

from stable_baselines3 import DDPG

env = gym.make('grSimSSLPenalty-v0')
model = DDPG.load("ddpg_penalty")

while True:
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
