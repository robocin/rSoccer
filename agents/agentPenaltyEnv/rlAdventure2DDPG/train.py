import gym
import gym_ssl
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch

from agents.Utils.normalization import NormalizedWrapper
from agents.Utils.networks import ValueNetwork, PolicyNetwork
from agents.Utils.OUNoise import OUNoise
from agents.Utils.replayBuffer import ReplayBuffer


class AgentDDPG:

    def __init__(self,
                 maxEpisodes=10000, maxSteps=200, batchSize=256, replayBufferSize=200000, valueLR=1e-3, policyLR=1e-4,
                 hiddenDim=256):
        # Training Parameters
        self.batchSize = batchSize
        self.maxSteps = maxSteps
        self.maxEpisodes = maxEpisodes
        self.episode = 0

        # Check if cuda gpu is available, and select it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Environment using a wrapper which scales actions and observations to [-1, 1]
        self.env = NormalizedWrapper(gym.make("grSimSSLPenalty-v0"))

        # Init action noise object
        self.ouNoise = OUNoise(self.env.action_space)

        # Init networks
        stateDim = self.env.observation_space.shape[0]
        actionDim = self.env.action_space.shape[0]
        self.valueNet = ValueNetwork(stateDim, actionDim, hiddenDim).to(self.device)
        self.policyNet = PolicyNetwork(stateDim, actionDim, hiddenDim, device=self.device).to(self.device)
        self.targetValueNet = ValueNetwork(stateDim, actionDim, hiddenDim).to(self.device)
        self.targetPolicyNet = PolicyNetwork(stateDim, actionDim, hiddenDim, device=self.device).to(self.device)
        # Same initial parameters for target networks
        for target_param, param in zip(self.targetValueNet.parameters(), self.valueNet.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.targetPolicyNet.parameters(), self.policyNet.parameters()):
            target_param.data.copy_(param.data)

        # Init optimizers
        self.valueOptimizer = optim.Adam(self.valueNet.parameters(), lr=valueLR)
        self.policyOptimizer = optim.Adam(self.policyNet.parameters(), lr=policyLR)

        # Init replay buffer
        self.replayBuffer = ReplayBuffer(replayBufferSize)

        # Tensorboard Init
        self.path = './runs/'
        # self._load
        self.writer = SummaryWriter(log_dir=self.path)

    def _update(self, batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):
        state, action, reward, next_state, done = self.replayBuffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policyLoss = self.valueNet(state, self.policyNet(state))
        policyLoss = -policyLoss.mean()

        next_action = self.targetPolicyNet(next_state)
        target_value = self.targetValueNet(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.valueNet(state, action)
        value_criterion = nn.MSELoss()
        value_loss = value_criterion(value, expected_value.detach())

        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        self.policyOptimizer.step()

        self.valueOptimizer.zero_grad()
        value_loss.backward()
        self.valueOptimizer.step()

        for target_param, param in zip(self.targetValueNet.parameters(), self.valueNet.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.targetPolicyNet.parameters(), self.policyNet.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    # Training Loop
    def train(self):
        while self.episode < self.maxEpisodes:
            state = self.env.reset()
            self.ouNoise.reset()
            episode_reward = 0
            steps_episode = 0

            for step in range(self.maxSteps):
                action = self.policyNet.get_action(state)
                action = self.ouNoise.get_action(action, step)
                next_state, reward, done, _ = self.env.step(action)

                self.replayBuffer.push(state, action, reward, next_state, done)
                if len(self.replayBuffer) > self.batchSize:
                    self._update(self.batchSize)

                state = next_state
                episode_reward += reward

                if done:
                    steps_episode = step
                    break

            self.episode += 1

            # rewards.append(episode_reward)

            self.writer.add_scalar('Train/Reward', episode_reward, self.episode)
            self.writer.add_scalar('Train/Steps', steps_episode, self.episode)

            # if (episode % 1000) == 0:
            #     torch.save({
            #         'target_value_net_dict': target_value_net.state_dict(),
            #         'target_policy_net_dict': target_policy_net.state_dict(),
            #     }, './saved_networks')

        self.writer.flush()
