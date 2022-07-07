import torch.nn as nn

############################### Network Classes ###################################
class ActorNetwork(nn.Module):
  def __init__(self, observation_size, action_size):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(observation_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, action_size),
        nn.Tanh()
    )
  
  def forward(self, inp):
    return self.net(inp)
############################# End Network Classes #################################