import gym 
import threading
import numpy as np
import random


class EnvContext:

    def __init__(self, n_agents):
        self.barrier = threading.Barrier(n_agents)
        self.commands = None
        self.commands_received = 0
        self.states = []
        self.rewards = []
        self.dones = []
        self.action_space = None
        self.observation_space = None
        self.reward_range = None


class SingleAgentSoccerEnvWrapper(gym.Env):

    def __init__(self):
        self.first = False
        self.soccer_env = None
        self.agent_id = None
        self.n_agents = None
        self.random_cmd = True

    def setup(self, soccer_env, agent_id, n_agents, random_cmd=True):
        self.soccer_env = soccer_env
        self.agent_id = agent_id
        self.n_agents = n_agents
        # This attributes set's if I should send random commands to the robots i'm not controlling with policy
        self.random_cmd = random_cmd

        if self.soccer_env.env_context is None:
            self.soccer_env.env_context = EnvContext(self.n_agents)
            self.first = True  # only the first agent actually sends the commands

    def __del__(self):
        pass

    def stop(self):
        if self.soccer_env.env_context is not None:
            self.soccer_env.env_context.barrier.abort()

    def reset(self):

        self.soccer_env.env_context.commands_received = 0

        if self.first:
            self.soccer_env.env_context.states = self.soccer_env.reset()
            self.soccer_env.env_context.action_space = self.soccer_env.action_space
            self.soccer_env.env_context.observation_space = self.soccer_env.observation_space
            self.soccer_env.env_context.reward_range = self.soccer_env.reward_range

        # print("%d waiting for reset." % self.agent_id)
        self.soccer_env.env_context.barrier.wait()
        # print("%d reset done." % self.agent_id)
        return self.soccer_env.env_context.states[self.agent_id]

    def step(self, command):

        if self.soccer_env.env_context.commands is None:
            if self.random_cmd:  # if send random commands to other agents
                cmd_array = [command]*len(self.soccer_env.team)
            else:
                cmd_array = [command]*self.n_agents

            self.soccer_env.env_context.commands = np.array(cmd_array)
            self.soccer_env.env_context.commands.fill(np.nan)

        # Collects the nth command
        # print("%d command:" % self.agent_id, command)
        self.soccer_env.env_context.commands[self.agent_id] = command

        try:
            # waits for the other commands
            # print("%d waiting for commands." % self.agent_id)
            self.soccer_env.env_context.barrier.wait()
            # print("%d command received: %d" % (self.agent_id, command))

            if self.first:  # only the first agent actually calls the environments step

                if self.random_cmd:  # set up random commands for the other agents
                    for i in range(0, len(self.soccer_env.team)):
                        if np.isnan(self.soccer_env.env_context.commands[i]).any():
                            self.soccer_env.env_context.commands[i] = self.soccer_env.env_context.action_space.sample()

                # send commands:
                self.soccer_env.env_context.states, self.soccer_env.env_context.rewards, self.soccer_env.env_context.dones, _ = self.soccer_env.step(self.soccer_env.env_context.commands)
                # clear commands:
                self.soccer_env.env_context.commands.fill(np.nan)

            if self.soccer_env.env_context.states is None:
                return None, 0, True, {}

            # waits for the other commands
            # print("%d waiting for state." % self.agent_id)
            self.soccer_env.env_context.barrier.wait()
            # print("%d state received." % self.agent_id)

            return self.soccer_env.env_context.states[self.agent_id], self.soccer_env.env_context.rewards[self.agent_id], self.soccer_env.env_context.dones[self.agent_id], {}

        except threading.BrokenBarrierError:
            return None, 0, True, {}

    def seed(self, seed=None):
        return [self.soccer_env.seed(seed)]

    def render(self, mode='human', close=False):
        self.soccer_env.render(mode, close)
