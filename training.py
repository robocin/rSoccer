import gym
from torch.nn.modules.activation import Sigmoid
import torch, torch.functional
import torch.nn as nn, torch.optim as optimizers
from torch import tensor
from copy import deepcopy
from math import exp
import numpy as np
from tqdm import trange
from networks import ActorNetwork
import rsoccer_gym
import wandb
import random

from rsoccer_gym.vss.env_vss.vss5v5_gym import VSS5v5Env

####################################################################   Buffer implementation  #######################################
class Buffer():
  def __init__(self, batch_size=1000):
    self.capacity = batch_size * 1024
    self.items = [([0.0 for _ in range(8)], [0.0 for _ in range(2)], [0.0 for _ in range(8)], 0, 0) for _ in range(self.capacity)]
    self.size = 0
    self.cursor = 0
  
  def add(self, item):
    self.items[self.cursor] = item
    self.cursor = (self.cursor + 1) % self.capacity
    if self.size < self.capacity:
      self.size += 1

  def sample(self, batch_size):
    return random.sample(self.items[:self.size], batch_size)

  def __len__(self):
    return self.size
################################################################## End Buffer implementation  ########################################


######### HIPERPARAMETERS ###########
LEARNING_RATE = 0.001
GAMMA = 0.99
STEPS = 300000
SIGMA = 0.3
TAU = 0.001
BATCH_SIZE = 32
TRAINING_FREQ = 1
TARGET_UPDATE_RATE = 1
EPISODE_MAX_LENGTH = 2000
####### END HIPERPARAMETERS #########


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



###################################### INSTANTIATE ########################################
env = VSS5v5Env()

observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

p_net = ActorNetwork(observation_size, action_size)

target_p_net = deepcopy(p_net)

q_net = nn.Sequential(
    nn.Linear(observation_size + action_size, 400),
    nn.ReLU(),
    nn.Linear(400, 300),
    nn.ReLU(),
    nn.Linear(300, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

target_q_net = deepcopy(q_net)

p_net.to(DEVICE)
target_p_net.to(DEVICE)
q_net.to(DEVICE)
target_q_net.to(DEVICE)

p_optim = optimizers.Adam(p_net.parameters(), lr=LEARNING_RATE)

q_optim = optimizers.Adam(q_net.parameters(), lr=LEARNING_RATE)

loss_fn = nn.MSELoss()

buffer = Buffer(batch_size=BATCH_SIZE)
################################### END INSTANTIATE ######################################




###################################### TRAINING ########################################
wandb.init(
    monitor_gym=False,
    project="vss5v5env-energy-penalty-normalize-training",
    tags=[
      f"env: {'VSS5v5Env'}",
      f"actor: target_p_net",
      f"critic: target_q_net"
    ]
)

state = env.reset()

avg_rw = 0
ep_rw = 0

episode_length = -1

for i in trange(STEPS):
  episode_length += 1
  action = torch.Tensor.cpu(p_net(torch.tensor(state).float().to(DEVICE)).detach()).numpy()
  action = np.clip(np.random.normal(action, SIGMA), a_min=-1, a_max=1)

  next_state, reward, done, _ = env.step(action)

  if episode_length >= EPISODE_MAX_LENGTH:
    done = True
    episode_length = 0

  avg_rw = ((avg_rw*i) + reward)/(i+1)

  ep_rw += reward

  buffer.add( (state, action, next_state, reward, int(done)) )

  state = next_state

  if done:
    state = env.reset()
  elif i % TRAINING_FREQ != 0:
    continue

  if len(buffer) < BATCH_SIZE:
    continue

  batch = buffer.sample(BATCH_SIZE)
  state_batch = torch.tensor( np.array([i[0] for i in batch]) ).float().to(DEVICE)
  action_batch = torch.tensor( np.array([i[1] for i in batch]) ).float().to(DEVICE)
  next_state_batch = torch.tensor( np.array([i[2] for i in batch]) ).float().to(DEVICE)
  reward_batch = torch.tensor( np.array([i[3] for i in batch]) ).unsqueeze(1).float().to(DEVICE)
  done_batch = torch.tensor( np.array([i[4] for i in batch]) ).unsqueeze(1).float().to(DEVICE)

  q_batch = q_net(
    torch.cat(
      (
        state_batch,
        action_batch
      ),
      1
    )
  )

  with torch.no_grad():
    q_next = target_q_net(
      torch.cat(
        (
          next_state_batch,
          target_p_net(
            next_state_batch
          )
        ),
        1
      )
    )

    q_target = reward_batch + ((q_next*GAMMA) * (1 - done_batch))
  
  q_loss = loss_fn(q_batch, q_target)

  q_optim.zero_grad()
  q_loss.backward()
  q_optim.step()

  p_loss = -q_net(
    torch.cat(
      (
        state_batch,
        p_net(
          state_batch
        )
      ),
      1
    )
  ).mean()


  p_optim.zero_grad()
  p_loss.backward()
  p_optim.step()

  if done:
    wandb.log(dict(
        episode_reward = ep_rw,
        average_reward = avg_rw,
        q_loss = q_loss,
        policy_loss = p_loss
      )
    )
    ep_rw = 0

  if i % TARGET_UPDATE_RATE != 0:
    continue

  for target_param, param in zip(target_p_net.parameters(), p_net.parameters()):
      target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        
  for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
      target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
################################### END TRAINING ######################################


target_q_net.to(torch.device("cpu"))
target_p_net.to(torch.device("cpu"))

torch.save(target_q_net.state_dict(), 'latest_critic_model.pt')
torch.save(target_p_net.state_dict(), 'latest_actor_model.pt')