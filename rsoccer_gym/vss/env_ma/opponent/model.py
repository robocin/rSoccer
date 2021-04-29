import torch
import torch.nn as nn


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

    def get_action(self, state):
        x = torch.FloatTensor(state)
        action = self.forward(x)
        action = action.cpu().detach().numpy()
        return action
