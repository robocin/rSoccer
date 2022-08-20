# No experiment being run

import gym
import rsoccer_gym

debug = dict()

env = gym.make("VSS5v5-v0")
state = env.reset()

observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

if __name__ == '__main__':
    done = False
    info = None
    env.render()
    while not done:
        state, _, done, info = env.step(
            [1.0,1.0]
        )
        env.render()
    
    print(info)
