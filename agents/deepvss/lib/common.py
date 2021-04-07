import math
import os
import struct
import sys
import time
from collections import deque, namedtuple
from itertools import islice

import numpy as np
import ptan
import torch


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False, dtype=np.float)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)  # the result will be masked anyway
        else:
            last_states.append(
                np.array(exp.last_state, copy=False, dtype=np.float))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class NStepTracer():
    """
    A short-term cache for n-step bootstrapping.

    Parameters
    ----------
    n : positive int

        The number of steps over which to bootstrap.

    gamma : float between 0 and 1

        The amount by which to discount future rewards.
    """

    def __init__(self, n, gamma):
        self.n = int(n)
        self.gamma = float(gamma)
        self.reset()

    def reset(self):
        self._deque_s = deque([])
        self._deque_r = deque([])
        self._done = False
        self._gammas = np.power(self.gamma, np.arange(self.n))
        self._gamman = np.power(self.gamma, self.n)

    def add(self, s, a, r, done):
        if self._done and len(self):
            # ("please flush cache (or repeatedly call popleft) before appending new transitions")
            raise Exception

        self._deque_s.append((s, a))
        self._deque_r.append(r)
        self._done = bool(done)

    def __len__(self):
        return len(self._deque_s)

    def __bool__(self):
        return bool(len(self)) and (self._done or len(self) > self.n)

    def pop(self):
        if not self:
            # ("cache needs to receive more transitions before it can be popped from")
            raise Exception

        # pop state-action (propensities) pair
        s, a = self._deque_s.popleft()

        # n-step partial return
        zipped = zip(self._gammas, self._deque_r)
        rn = sum(x * r for x, r in islice(zipped, self.n))
        self._deque_r.popleft()

        # keep in mind that we've already popped (s, a)
        if len(self) >= self.n:
            s_next, a_next = self._deque_s[self.n - 1]
            done = False
        else:
            # no more bootstrapping
            s_next, a_next, done = None, a, True

        return ptan.experience.ExperienceFirstLast(state=s,
                                                   action=a,
                                                   reward=rn,
                                                   last_state=s_next)


class RewardTracker:
    def __init__(self, writer):
        self.writer = writer
        self.last_goal = 0
        self.goal_scores = []
        self.total_rewards = []
        self.goal_rewards = []
        self.ts = time.time()
        self.ts_frame = 0
        self.ts_p = self.ts
        self.ts_processed_samples = 0

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.ts_p = self.ts
        self.ts_processed_samples = 0
        return self

    def __exit__(self, *args):
        self.writer.close()

    def track_env(self, env, frame, max_score, epsilon=None):
        #print('track_env.env.rw_goal:%f, frame:%d' % (env.rw_goal, frame))

        self.goal_scores.append(env.goal_score)
        mean_goal_score = np.mean(self.goal_scores[-100:])

        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()

        if env.my_goals > 0:
            self.writer.add_scalar(
                "steps_to_goal", (frame - self.last_goal)/env.my_goals, frame)
            self.last_goal = frame

        self.writer.add_scalar("goal_score", env.goal_score, frame)
        self.writer.add_scalar("mean_goal_score", mean_goal_score, frame)
        self.writer.add_scalar("rw_goal", env.rw_goal, frame)
        self.writer.add_scalar("rw_ball_grad", env.rw_ball_grad, frame)
        self.writer.add_scalar("speed", speed, frame)

        epsilon_str = ""
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
            epsilon_str = ", eps %.2f" % epsilon

        print("frame:%d: done %d games, score %.3f, mean:%.3f, speed %.2f f/s%s" % (
            frame, len(
                self.goal_scores), env.goal_score, mean_goal_score, speed, epsilon_str
        ))
        sys.stdout.flush()

        return mean_goal_score == max_score

    def track_agent(self, agent, frame):
        str_id = "_%d" % agent.id

        self.total_rewards.append(agent.rw_total)
        mean_total_rewards = np.mean(self.total_rewards[-100:])
        self.writer.add_scalar("rw_total_mean" + str_id,
                               mean_total_rewards, frame)

        if agent.track_rw:
            self.writer.add_scalar("rw_total" + str_id, agent.rw_total, frame)
            self.writer.add_scalar("rw_move"+str_id, agent.rw_move, frame)
            self.writer.add_scalar("rw_collision"+str_id,
                                   agent.rw_collision, frame)
            self.writer.add_scalar("rw_energy" + str_id,
                                   agent.rw_energy, frame)

    def track_training(self, processed_samples, reward, actor_loss, critic_loss=0):

        speed = (processed_samples - self.ts_processed_samples) / \
            (time.time() - self.ts_p)
        self.ts_processed_samples = processed_samples
        self.ts_p = time.time()
        self.writer.add_scalar("reward", reward, processed_samples)
        self.writer.add_scalar("actor_loss", actor_loss, processed_samples)
        self.writer.add_scalar("critic_loss", critic_loss, processed_samples)
        self.writer.add_scalar("train_speed", speed, processed_samples)

    def track_training_sac(self, processed_samples, reward, policy_loss, critic_1_loss=0, critic_2_loss=0, alpha=0, ent_loss=0):

        speed = (processed_samples - self.ts_processed_samples) / \
            (time.time() - self.ts_p)
        self.ts_processed_samples = processed_samples
        self.ts_p = time.time()
        self.writer.add_scalar("reward", reward, processed_samples)
        self.writer.add_scalar(
            'loss/critic_1', critic_1_loss, processed_samples)
        self.writer.add_scalar(
            'loss/critic_2', critic_2_loss, processed_samples)
        self.writer.add_scalar('loss/policy', policy_loss, processed_samples)
        self.writer.add_scalar('loss/entropy_loss',
                               ent_loss, processed_samples)
        self.writer.add_scalar('entropy_temprature/alpha',
                               alpha, processed_samples)
        self.writer.add_scalar("train_speed", speed, processed_samples)


class ParameterTracker:
    def __init__(self, epsilon_greedy_selector, params, phase):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_high = self.epsilon_high_start = params['epsilon_high_start']
        self.epsilon_high_final = params['epsilon_high_final']
        self.epsilon_low = params['epsilon_low']
        self.epsilon_frames = params['epsilon_frames']
        self.epsilon_decay = params['epsilon_decay']
        self.phase = phase
        self.eval = False
        self.eval_eps = 1.0
        self.frame(0)

    def set_evaluation(self, eval, eval_eps=1.0):
        self.eval = eval
        self.epsilon_greedy_selector.epsilon = self.eval_eps = eval_eps

    # This is a dumb way of calculating the current epsilons for a certain frame
    # fell free to came up with a clever way
    def set_frame(self, frames):
        modframe = int(self.phase) % self.epsilon_frames

        for frame in range(0, frames):
            modframe = int(frame + self.phase) % (self.epsilon_frames)

            if frame > 0 and modframe == 0:
                self.epsilon_high = max(
                    self.epsilon_high_final, (self.epsilon_decay * self.epsilon_high))

        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_low, self.epsilon_high -
                self.epsilon_high * (modframe / self.epsilon_frames))

    def frame(self, frame):
        modframe = int(frame+self.phase) % self.epsilon_frames

        if frame > 0 and modframe == 0:
            self.epsilon_high = max(
                self.epsilon_high_final, (self.epsilon_decay * self.epsilon_high))

        if self.eval:
            self.epsilon_greedy_selector.epsilon = self.eval_eps
        else:
            self.epsilon_greedy_selector.epsilon = \
                max(self.epsilon_low, self.epsilon_high -
                    self.epsilon_high*(modframe / self.epsilon_frames))


class PersistentExperienceReplayBuffer(ptan.experience.ExperienceReplayBuffer):

    def __init__(self, experience_source, buffer_size):
        super(PersistentExperienceReplayBuffer, self).__init__(
            experience_source, buffer_size)
        self.sync_start = 0
        self.sync_count = 0
        self.state_format = '56f'
        self.action_format = 'B'

    def clear(self):
        self.buffer = []
        self.pos = 0

    def set_state_action_format(self, state_format='56f', action_format='B'):
        self.state_format = state_format
        self.action_format = action_format

    def exp_to_bytes(self, exp):
        # 52 is the length of a state
        buf = struct.pack('='+self.state_format, *exp.state)

        if self.action_format == 'B':
            buf += struct.pack('='+self.action_format, exp.action)
        else:
            buf += struct.pack('='+self.action_format, *exp.action)

        buf += struct.pack('=f', exp.reward)

        if exp.last_state is None:
            buf += struct.pack('=?', True)
            # repeat first state as padding
            buf += struct.pack('='+self.state_format, *exp.state)
        else:
            buf += struct.pack('=?', False)
            buf += struct.pack('='+self.state_format, *exp.last_state)

        return buf

    def unpack_formats(self, fmts, data):
        result = []
        offset = 0
        for fmt in fmts:
            result.append(struct.unpack_from(fmt, data, offset))
            offset += struct.calcsize(fmt)
        return result

    def bytes_to_exp(self, exp_b):

        fmts = ('='+self.state_format, '='+self.action_format,
                '=f', '=?', '='+self.state_format)
        state, action, reward, is_none, last_state = self.unpack_formats(
            fmts, exp_b)

        reward = reward[0]
        is_none = is_none[0]

        state = np.array(state, dtype=np.float32)

        if self.action_format == 'B':
            action = int(action[0])
        else:
            action = np.array(action, dtype=np.float32)

        if is_none:
            last_state = None
        else:
            last_state = np.array(last_state, dtype=np.float32)

        return ptan.experience.ExperienceFirstLast(state, action, reward, last_state)

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.pos = len(self.buffer) - 1
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

        self.sync_count += 1
        #print("pos: %d, sync_start:%d, count:%d" % (self.pos, self.sync_start, self.sync_count))

    def sync_exps_to_file(self, file_name):

        HEADER_SIZE = 6  # bytes

        try:
            file = open(file_name, "r+b")
        except FileNotFoundError:
            file = open(file_name, "w+b")

        sync_pos = self.sync_start
        exp_b = self.exp_to_bytes(self.buffer[sync_pos])
        exp_size = len(exp_b)

        if self.sync_start == 0 and os.path.getsize(file_name) < HEADER_SIZE:
            file.write(sync_pos.to_bytes(4, byteorder='big'))
            # 4+2 = a six bytes header
            file.write(exp_size.to_bytes(2, byteorder='big'))
        else:
            file.seek(HEADER_SIZE + sync_pos * exp_size)

        print("saving %d objects from %d to %d. len:%d" %
              (self.sync_count, self.sync_start, self.pos, len(self.buffer)))

        for _ in range(self.sync_count):
            exp_b = self.exp_to_bytes(self.buffer[sync_pos])
            file.write(exp_b)
            sync_pos += 1
            if sync_pos >= self.capacity:
                sync_pos = 0
                file.seek(HEADER_SIZE + sync_pos * exp_size)

        # update position in the header
        file.seek(0)
        file.write(sync_pos.to_bytes(4, byteorder='big'))
        file.close()

        self.sync_count = 0
        self.sync_start = sync_pos
        print("done at pos: %d" % sync_pos)

    def load_exps_from_file(self, file_name):

        file = open(file_name, "rb")

        file_pos = int.from_bytes(file.read(4), byteorder='big')
        exp_size = int.from_bytes(file.read(2), byteorder='big')

        count = 0
        while True:
            exp_b = file.read(exp_size)
            if not exp_b:
                break

            exp = self.bytes_to_exp(exp_b)
            self._add(exp)
            count += 1

        self.sync_start = self.pos = file_pos
        self.sync_count = 0
        print("%d objects loaded from file which is at pos: %d" %
              (count, file_pos))

        file.close()
        return file_pos


class PersistentExperiencePrioritizedReplayBuffer(ptan.experience.PrioritizedReplayBuffer):

    def __init__(self, experience_source, buffer_size, alpha, beta):
        super(PersistentExperiencePrioritizedReplayBuffer, self).__init__(
            experience_source, buffer_size, alpha)
        self.sync_start = 0
        self.sync_count = 0
        self.state_format = '56f'
        self.action_format = 'B'
        self.init_beta = beta
        self.beta = beta
        self.beta_inc = 0.001
        self.it_capacity = 1
        while self.it_capacity < buffer_size:
            self.it_capacity *= 2

    def sample(self, batch_size):
        self.beta = np.min([1., self.beta + self.beta_inc])
        res = super().sample(batch_size, self.beta)
        return res

    def set_state_action_format(self, state_format='56f', action_format='B'):
        self.state_format = state_format
        self.action_format = action_format

    def exp_to_bytes(self, exp):
        # 52 is the length of a state
        buf = struct.pack('='+self.state_format, *exp.state)

        if self.action_format == 'B':
            buf += struct.pack('='+self.action_format, exp.action)
        else:
            buf += struct.pack('='+self.action_format, *exp.action)

        buf += struct.pack('=f', exp.reward)

        if exp.last_state is None:
            buf += struct.pack('=?', True)
            # repeat first state as padding
            buf += struct.pack('='+self.state_format, *exp.state)
        else:
            buf += struct.pack('=?', False)
            buf += struct.pack('='+self.state_format, *exp.last_state)

        return buf

    def unpack_formats(self, fmts, data):
        result = []
        offset = 0
        for fmt in fmts:
            result.append(struct.unpack_from(fmt, data, offset))
            offset += struct.calcsize(fmt)
        return result

    def bytes_to_exp(self, exp_b):

        fmts = ('='+self.state_format, '='+self.action_format,
                '=f', '=?', '='+self.state_format)
        state, action, reward, is_none, last_state = self.unpack_formats(
            fmts, exp_b)

        reward = reward[0]
        is_none = is_none[0]

        state = np.array(state, dtype=np.float32)

        if self.action_format == 'B':
            action = int(action[0])
        else:
            action = np.array(action, dtype=np.float32)

        if is_none:
            last_state = None
        else:
            last_state = np.array(last_state, dtype=np.float32)

        return ptan.experience.ExperienceFirstLast(state, action, reward, last_state)

    def _add(self, sample):
        idx = self.pos
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.pos = len(self.buffer) - 1
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity
        self.sync_count += 1
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def clear(self):
        self.buffer = []
        self.pos = 0
        self._it_sum = ptan.common.utils.SumSegmentTree(self.it_capacity)
        self._it_min = ptan.common.utils.MinSegmentTree(self.it_capacity)
        self._max_priority = 1.0
        self.beta = self.init_beta

    def sync_exps_to_file(self, file_name):

        HEADER_SIZE = 6  # bytes

        try:
            file = open(file_name, "r+b")
        except FileNotFoundError:
            file = open(file_name, "w+b")

        sync_pos = self.sync_start
        exp_b = self.exp_to_bytes(self.buffer[sync_pos])
        exp_size = len(exp_b)

        if self.sync_start == 0 and os.path.getsize(file_name) < HEADER_SIZE:
            file.write(sync_pos.to_bytes(4, byteorder='big'))
            # 4+2 = a six bytes header
            file.write(exp_size.to_bytes(2, byteorder='big'))
        else:
            file.seek(HEADER_SIZE + sync_pos * exp_size)

        print("saving %d objects from %d to %d. len:%d" %
              (self.sync_count, self.sync_start, self.pos, len(self.buffer)))

        for _ in range(self.sync_count):
            exp_b = self.exp_to_bytes(self.buffer[sync_pos])
            file.write(exp_b)
            sync_pos += 1
            if sync_pos >= self.capacity:
                sync_pos = 0
                file.seek(HEADER_SIZE + sync_pos * exp_size)

        # update position in the header
        file.seek(0)
        file.write(sync_pos.to_bytes(4, byteorder='big'))
        file.close()

        self.sync_count = 0
        self.sync_start = sync_pos
        print("done at pos: %d" % sync_pos)

    def load_exps_from_file(self, file_name):

        file = open(file_name, "rb")

        file_pos = int.from_bytes(file.read(4), byteorder='big')
        exp_size = int.from_bytes(file.read(2), byteorder='big')

        count = 0
        while True:
            exp_b = file.read(exp_size)
            if not exp_b:
                break

            exp = self.bytes_to_exp(exp_b)
            self._add(exp)
            count += 1

        self.sync_start = self.pos = file_pos
        self.sync_count = 0
        print("%d objects loaded from file which is at pos: %d" %
              (count, file_pos))

        file.close()
        return file_pos

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


# Ornstein-Uhlenbeck process
# Adding time-correlated noise to the actions taken by the deterministic policy
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

class OUNoise(object):
    def __init__(self, mu=0.0, theta=0.15,
                 max_sigma=0.2, min_sigma=0.2, decay_period=9375):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 2
        self.low = -1
        self.high = 1
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
