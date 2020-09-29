import gym
import rc_gym
import torch

from agents.Utils.Networks             import PolicyNetwork

# Using penalty env
env = gym.make('grSimSSLGoToBall-v0')

use_cuda = torch.cuda.is_available()
print("use_cuda ->", use_cuda)
device   = torch.device("cuda" if use_cuda else "cpu")

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256

policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=device).to(device)

checkpoint = torch.load('./models/test_go_to_ball')
policy_net.load_state_dict(checkpoint['target_policy_net_dict'])

env.reset()
# Run for 100 episode and print reward at the end
for i in range(100):
    done = False
    obs = env.reset()
    while not done:
        action = policy_net.get_action(obs)
        obs, reward, done, _ = env.step(action)
    print(reward)