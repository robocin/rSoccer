import gym
import gym_ssl

env = gym.make('grSimSSL-v0')
env.step()
env.reset()
print(env.action_space.sample())
print(env.observation_space.sample())
