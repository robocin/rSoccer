import gym
import gym_ssl
import torch

from agents.Utils.Networks             import PolicyNetwork

# Using penalty env
env = gym.make('grSimSSLShootGoalie-v0')

use_cuda = torch.cuda.is_available()
print("use_cuda ->", use_cuda)
device   = torch.device("cuda" if use_cuda else "cpu")

max_steps   = 130 # Be Careful!

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 300

policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=device).to(device)

checkpoint = torch.load('./models/saved_networks_penalize_fix')
policy_net.load_state_dict(checkpoint['target_policy_net_dict'])

env.reset()
# Run for 100 episode and print reward at the end
for i in range(100):
    done = False
    obs = env.reset()
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        action = policy_net.get_action(obs)
        obs, reward, done, _ = env.step(action)
    print(reward)