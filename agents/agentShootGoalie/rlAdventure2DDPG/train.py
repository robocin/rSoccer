import gym
import gym_ssl
import numpy as np
import os
import sys

from    torch.utils.tensorboard  import SummaryWriter
import  torch.optim  as optim
import  torch.nn     as nn
import  torch

from agents.Utils.Networks             import ValueNetwork, PolicyNetwork
from agents.Utils.Normalization        import NormalizedWrapper
from agents.Utils.ReplayBuffer         import ReplayBuffer
from agents.Utils.OUNoise              import OUNoise

use_cuda = torch.cuda.is_available()
print("use_cuda ->", use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

writer = SummaryWriter()

max_episodes  = 2000000
max_steps   = 130 # Be Careful!
episode   = 0
rewards     = []
batch_size  = 256
replay_buffer_size = 100000

# Continuous control with deep reinforcement learning
def ddpg_update(batch_size, 
           gamma = 0.99,
           min_value=-np.inf,
           max_value=np.inf,
           soft_tau=1e-2):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())


    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


if __name__ == "__main__":
        
    env = NormalizedWrapper(gym.make('grSimSSLShootGoalie-v0'))
    # env = gym.make('grSimSSLShootGoalie-v0')
    ou_noise = OUNoise(env.action_space)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 300

    value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=device).to(device)

    target_value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, device=device).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)    
        
    value_lr  = 1e-3
    policy_lr = 1e-4

    value_optimizer  = optim.Adam(value_net.parameters(),  lr=value_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

    value_criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer(replay_buffer_size)

    while episode < max_episodes:
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0
        steps_episode = 0
        goal = 0
        while steps_episode <= max_steps:
            action = policy_net.get_action(state)
            action = ou_noise.get_action(action, steps_episode)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > batch_size:
                ddpg_update(batch_size)
            
            state = next_state
            episode_reward = reward
            
            steps_episode += 1

            if done:
                if reward > 0:
                    goal = 1
                break
        
        episode += 1
        # rewards.append(episode_reward)

        writer.add_scalar('Train/Reward', episode_reward, episode)
        writer.add_scalar('Train/Steps', steps_episode, episode)
        writer.add_scalar('Episodes/Goals', goal, episode)

        if (episode % 1000) == 0:
            torch.save({
                'target_value_net_dict': target_value_net.state_dict(),
                'target_policy_net_dict': target_policy_net.state_dict()
            }, './models/saved_networks_penalize_fix')