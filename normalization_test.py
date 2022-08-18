# This is the experiment made to obtain the maximum accumulated energy_penalty in
# the VSS5v5Env. There's only one robot in the 5v5 field, which will 
# be trained without the influence of energy_penalty

import gym
from rsoccer_gym.vss.env_vss.vss5v5_gym import VSS5v5Env

debug = dict()

env = VSS5v5Env()
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

# The found maximum accumulated energy_penalty was 30000
# experiment: https://wandb.ai/robocin/RoboCIn-RL/runs/4vdd7if2?workspace=user-icaro-nunes