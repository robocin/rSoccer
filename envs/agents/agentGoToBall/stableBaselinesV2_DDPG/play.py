import gym
import rc_gym
import numpy as np

from stable_baselines import DDPG

env = gym.make('grSimSSLGoToBall-v0')
model = DDPG.load("./models/ddpg_gotoball2")

while True:
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
