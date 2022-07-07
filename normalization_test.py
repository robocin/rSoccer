# This is the experiment made to obtain the maximum energy_penalty in
# the VSS5v5Env. There's only one robot in the 5v5 field, which will 
# be trained without the influence of energy_penalty

import gym
from rsoccer_gym.vss.env_vss.vss5v5_gym import VSS5v5Env
from networks import ActorNetwork
import torch

debug = dict()

env = VSS5v5Env()
state = env.reset()

observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = ActorNetwork(observation_size, action_size)
agent.load_state_dict(torch.load('./latest_actor_model.pt'))

if __name__ == '__main__':
    done = False
    info = None
    env.render()
    while not done:
        state, _, done, info = env.step(
            agent(torch.tensor(state).float()).detach().numpy()
        )
        env.render()
    
    print(info)
    print(env.debug)

