import gym
import gym_vss

from gym_vss import SingleAgentSoccerEnvWrapper


env = gym.make('vss_soccer_cont-v0')
# env = SingleAgentSoccerEnvWrapper(env, simulator='sdk')
# If you want FIRASim
env = SingleAgentSoccerEnvWrapper(env, simulator='fira')
env.reset()
for i in range(1):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
env.close()
