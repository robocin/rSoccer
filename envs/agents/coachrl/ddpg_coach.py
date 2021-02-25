import argparse
import collections
import math
import os
import random
from pprint import pprint

import gym
import numpy as np
import ptan
import rc_gym
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import FrameStack

import wandb

# Hyperparameters
actor_lr = 0.0005
critic_lr = 0.0003
gamma = 0.99
batch_size = 32
buffer_limit = 500000
soft_tau = 0.005  # for target network soft update
device = torch.device('cuda')
torch.cuda.manual_seed(441)
np.random.seed(441)
random.seed(441)


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float),\
            torch.tensor(a_lst, dtype=torch.float), \
            torch.tensor(r_lst, dtype=torch.float), \
            torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs + 3, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x2 = F.relu(self.linear1(x))
        x3 = F.relu(self.linear2(x2))
        x4 = self.linear3(x3)
        return x4


class Actor(nn.Module):
    def __init__(self, num_inputs):
        super(Actor, self).__init__()
        self.num_input = num_inputs
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 3)

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x2 = F.relu(self.linear2(x1))
        x3 = torch.tanh(self.linear3(x2))
        return x3

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        state = state.view(1, self.num_input)
        action = self.forward(state)
        return action.detach().cpu().numpy()


class OUNoise(object):
    def __init__(self, mu=0.0, theta=0.15,
                 max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 3
        self.low = -1.25
        self.high = 1.25
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    res = math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])
    if idx > 0 and (idx == len(array) or res):
        return idx-1
    else:
        return idx


def mapped_action(action):
    action_list = ["000", "001", "002", "010", "011", "012", "020", "021",
                   "100", "101", "102", "110", "111", "112", "120", "121",
                   "200", "201", "210", "211"]
    act = ''
    for x in action:
        if x < -0.34:
            act += '0'
        elif x < 0.34:
            act += '1'
        else:
            act += '2'

    if not act in action_list:
        act = random.choice(action_list)
        action = []
        for x in act:
            if x == '0':
                action.append(-0.5)
            elif x == '1':
                action.append(0)
            else:
                action.append(0.5)
        action = np.array(action)
    return action_list.index(act), action


def train(actor, exp_queue, finish_event, load):
    wandb.init(name="CoachRL-DDPG-train", project="RC-Reinforcement")
    try:
        critic = Critic(40*60).to(device)
        critic_target = Critic(40*60).to(device)
        actor_target = Actor(40*60).to(device)
        critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
        actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
        if load:
            critic_dict = torch.load('models/DDPG_CRITIC.model')
            critic.load_state_dict(critic_dict)
            actor_optim_dict = torch.load(f'models/DDPG_ACTOR.optim')
            actor_optim.load_state_dict(actor_optim_dict)
            critic_optim_dict = torch.load(f'models/DDPG_CRITIC.optim')
            critic_optim.load_state_dict(critic_optim_dict)

        actor_target.load_state_dict(actor.state_dict())
        critic_target.load_state_dict(critic.state_dict())

        memory = ReplayBuffer()
        it = 0
        while not finish_event.is_set():
            for i in range(batch_size):
                exp = exp_queue.get()
                if exp is None:
                    break
                memory.put(exp)

            # training loop:
            while exp_queue.qsize() < batch_size/2:

                state_batch, action_batch,\
                    reward_batch, next_state_batch, done_batch = memory.sample(
                        batch_size)

                n_inputs = state_batch.size()[1]*state_batch.size()[2]
                state_batch = state_batch.view(batch_size, n_inputs)
                next_state_batch = next_state_batch.view(batch_size, n_inputs)

                state_batch = state_batch.to(device)
                next_state_batch = next_state_batch.to(device)
                action_batch = action_batch.to(device).squeeze()
                reward_batch = reward_batch.to(device)
                done_batch = done_batch.to(device)

                actor_loss = critic(state_batch, actor(state_batch))
                z = -torch.mean(actor_loss)
                actor_optim.zero_grad()
                z.backward()
                actor_optim.step()

                next_actions_target = actor_target(next_state_batch)
                q_targets = critic_target(
                    next_state_batch, next_actions_target)
                targets = reward_batch + (1.0 - done_batch)*gamma*q_targets

                q_values = critic(state_batch, action_batch)
                critic_loss = F.mse_loss(q_values, targets.detach())
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                for target_param, param in zip(critic_target.parameters(),
                                               critic.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

                for target_param, param in zip(actor_target.parameters(),
                                               actor.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

                wandb.log({'Loss/Actor': z.item(),
                           'Loss/Critic': critic_loss.item()},
                          step=it)
                it += 1

                if it % 10000 == 0:
                    torch.save(critic.state_dict(),
                               f'models/DDPG_CRITIC_{it}.model')
                    torch.save(actor.state_dict(),
                               f'models/DDPG_ACTOR_{it}.model')
                    torch.save(actor_optim.state_dict(),
                               f'models/DDPG_ACTOR_{it}.optim')
                    torch.save(critic_optim.state_dict(),
                               f'models/DDPG_CRITIC_{it}.optim')
    except KeyboardInterrupt:
        print("...Train Finishing...")
        finish_event.set()


def play(actor, exp_queue, env, test, i, finish_event):
    action_list = ["AAA", "AAZ", "AAG", "AZA", "AZZ", "AZG", "AGA", "AGZ",
                   "ZAA", "ZAZ", "ZAG", "ZZA", "ZZZ", "ZZG", "ZGA", "ZGZ",
                   "GAA", "GAZ", "GZA", "GZZ"]
    if not test:
        wandb.init(name=f"CoachRL-DDPG-{i}", project="RC-Reinforcement")
    try:
        while not finish_event.is_set():
            ori_env = gym.make('VSSCoach-v0')
            env = FrameStack(ori_env, 60)
            ou_noise = OUNoise()
            total_steps = 0
            for n_epi in range(33):
                actions_dict = {x: 0 for x in action_list}
                s = env.reset()
                s = s.__array__(dtype=np.float32)
                done = False
                score = 0.0
                epi_step = 0
                while not done:
                    a = actor.get_action(s)
                    if not test:
                        a = ou_noise.get_action(a, epi_step)[0]
                    else:
                        a = a[0]
                    action, a = mapped_action(a)
                    actions_dict[list(actions_dict.keys())[action]] += 1
                    s_prime, r, done, info = env.step(action)
                    s_prime = s_prime.__array__(dtype=np.float32)
                    done_mask = 0.0 if done else 1.0
                    exp = (s, a, r, s_prime, done_mask)
                    if not test:
                        exp_queue.put(exp)
                    # else:
                    env.unwrapped.render('human')
                    score += r
                    s = s_prime
                    total_steps += 1
                    epi_step += 1

                print(f'***********EPI {n_epi} ENDED***********')
                print(f'Total: {score}')
                print('Goal score: {}'.format(info['goal_score']))
                print(actions_dict)
                print('*****************************************')
                if not test:
                    wandb.log({'Rewards/total': score,
                               'Rewards/goal_score': info['goal_score'],
                               'Rewards/num_penalties': info['penalties'],
                               'Rewards/num_faults': info['faults'],
                               }, step=total_steps)

    except KeyboardInterrupt:
        print("...Agent Finishing...")
        finish_event.set()


def main(load_model=False, test=False):
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    finish_event = mp.Event()
    exp_queue = mp.Queue(maxsize=batch_size)
    n_inputs = 40*60
    actor = Actor(n_inputs).to(device)

    if load_model or test:
        actor_dict = torch.load('models/DDPG_ACTOR.model')
        actor.load_state_dict(actor_dict)

    actor.share_memory()
    play_threads = []
    for i in range(1):
        env = 'VSSCoach-v0'
        data_proc = mp.Process(target=play, args=(actor, exp_queue, env,
                                                  test, i, finish_event))
        data_proc.start()
        play_threads.append(data_proc)

    if not test:
        train_process = mp.Process(target=train,
                                   args=(actor, exp_queue,
                                         finish_event, load_model))
        train_process.start()
        train_process.join()
    play_threads = [t.join()
                    for t in play_threads]


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Predicts your time series')
    PARSER.add_argument('--test', default=False,
                        action='store_true', help="Test mode")
    PARSER.add_argument('--load', default=False,
                        action='store_true',
                        help="Load models from examples/models/")
    ARGS = PARSER.parse_args()
    if not os.path.exists('./models'):
        os.makedirs('models')

    main(load_model=ARGS.load, test=ARGS.test)
