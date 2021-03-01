import numpy as np
import ptan
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import OUNoise

DDPG_MODELS_ACTOR = {}  # add new models to the dict after class definition
DDPG_MODELS_CRITIC = {}  # add new models to the dict after class definition


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


DDPG_MODELS_ACTOR["linear"] = DDPGActor

DDPG_MODELS_CRITIC["linear"] = DDPGCritic


class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """

    def __init__(self, net, device="cpu", ou_enabled=True,
                 ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_noise = OUNoise(mu=ou_mu, theta=ou_teta,
                                max_sigma=ou_sigma, min_sigma=ou_sigma)

    def initial_state(self):
        return None

    def __call__(self, states, n_step):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        if self.ou_enabled:
            actions = self.ou_noise.get_action(actions, n_step)
        return actions
