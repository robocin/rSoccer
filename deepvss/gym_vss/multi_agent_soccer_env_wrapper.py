import gym
import threading
import numpy as np
import random
import os

bin_path = '/'.join(os.path.abspath(__file__).split('/')
                    [:-1]) + '/binaries_envs/'

SDK_PARAMS = {
    'env_name': "vss_soccer_cont-v0",
    'run_name': 'experiment',  # '%' in this field sets the run_name to the parameter's file name
    'n_agents': 3,

    # simulator config:
    'simulator_path': bin_path + 'vss_sdk/VSS-SimulatorSync',
    'simulator_type': 'SimParser',
    'ip': '127.0.0.1',  # real: '224.5.23.2'
    'port': 5555,  # real: 10006
    'ia': False,
    'init_agents': 0,  # 0: default positions, 1: random positions, 2: random base positions, 3: one agent, 4: goal_keeper, 5: penalty left, 6: penalty right
    'init_ball': 2,  # 0: stopped at the center, 1: random slow, 2: towards left goal, 3: towards right goal, 4: towards a random goal
    'random_cmd': False,
    'random_init': True,
    'is_team_yellow': True,  # True here requires ia = False
    'frame_step_size': (1000 / 60.0),
    'frame_skip': 1,
    # 5min in ms, to disable time limit set time_limit_ms < 0
    'time_limit_ms':  (5 * 60 * 1000),

    # Robot configuration
    'robot_center_wheel_dist': 4.88,
    'robot_wheel_radius': 2.0,

    # actions (this parameter is optional). None for 2D continuous space:
    'actions': None,
    'range_linear': 75.,   # linear speed range
    'range_angular': 15.,  # angular speed range

    # reward shaping/scaling parameters (per step):
    'w_rescaling': 10,
    'w_goal': (1.0, 1.0),  # (pro, against)
    'w_move': (0.02, 1),  # (-1,1)
    'w_ball_grad': (0.08, 1),  # (-1,1)
    'w_collision': (0.000, 1),  # {-1,0}
    'w_ball_pot': (0.0, 1),  # (-1,0)
    'w_energy': (1e-5, 1),  # (-1,0)
    'min_dist_move_rw': 0.0,  # cm
    'collision_distance': 12,
    'alpha_base': [0.25, 0.5, 0.6],

    # Tracking and log
    'track_rewards': [True, False, False],
}

FIRA_PARAMS = {
    'env_name': "vss_soccer_cont-v0",
    'run_name': 'experiment',  # '%' in this field sets the run_name to the parameter's file name
    'n_agents': 3,

    # simulator config:
    'simulator_path': bin_path + 'fira_sim/bin/FIRASim',
    'simulator_type': 'FiraParser',
    'ip': '127.0.0.1',  # real: '224.5.23.2'
    'port': 10020,  # real: 10006
    'init_agents': 0,  # 0: default positions, 1: random positions, 2: random base positions, 3: one agent, 4: goal_keeper, 5: penalty left, 6: penalty right
    'init_ball': 2,  # 0: stopped at the center, 1: random slow, 2: towards left goal, 3: towards right goal, 4: towards a random goal
    'ia': False,
    'random_init': True,
    'random_cmd': False,
    'is_team_yellow': True,  # True here requires ia = False
    'frame_step_size': (1000 / 250.0),
    'frame_skip': 0.08,
    # 5min in ms, to disable time limit set time_limit_ms < 0
    'time_limit_ms':  (5 * 60 * 1000),
    'render': False,
    'fast_mode': True,

    # Robot configuration
    'robot_center_wheel_dist': 5.0,
    'robot_wheel_radius': 2.0/4.0,

    # actions (this parameter is optional). None for 2D continuous space:
    'actions': None,
    'range_linear': 90.,   # linear speed range
    'range_angular': 7.,  # angular speed range

    # reward shaping/scaling parameters (per step):
    'w_rescaling': 10,
    'w_goal': (1.0, 1.0),  # (pro, against)
    'w_move': (0.02, 1),  # (-1,1)
    'w_ball_grad': (0.08, 1),  # (-1,1)
    'w_collision': (0.000, 1),  # {-1,0}
    'w_ball_pot': (0.0, 1),  # (-1,0)
    'w_energy': (1e-5, 1),  # (-1,0)
    'min_dist_move_rw': 0.0,  # cm
    'collision_distance': 12,
    'alpha_base': [0., 0.0, 0.0],

    # Tracking and log
    'track_rewards': [True, False, False]
}

x_speed = 1.2
REAL_PARAMS = {
    'env_name': 'vss_soccer_real_continuous-v0',
    'run_name': 'GameVSS',  # '%' in this field sets the run_name to the parameter's file name
    'n_agents': 3,

    # 'ip': '224.5.23.2',  # SSL
    # 'port': 10006,

    'ip': '127.0.0.1',  # VSS
    'port': 54000,

    'random_cmd': False,
    'is_team_yellow': True,  # True here requires ia = False
    'frame_step_size': (1000 / 30.0),
    'frame_skip': 0,
    # 5min in ms, to disable time limit set time_limit_ms < 0
    'time_limit_ms':  (0.5 * 60 * 1000),

    # Robot configuration
    'robot_center_wheel_dist': 3.35,  # default: 3.75,
    'robot_wheel_radius': 2.6,  # default: 2.6
    # adjust linear speeds (values per agent)
    'pulse_speed_ratio': [0.22, 0.2, 0.2],

    # ctrl configuration
    'cmd_moving_average': 0.9,
    # [Kp, Ki, Kd][0.15, 0.2, 0]
    'angular_speed_pid':  [0.15, 0.01, 0.000],
    'linear_speed_pid': [0.025, 0.1, 0.],  # [Kp, Ki, Kd][0.15, 0.1, 0.001]

    # actions (this parameter is optional). None for 2D continuous space:
    'actions': None,
    'range_linear':  x_speed*75.0,  # linear speed range
    'range_angular': x_speed*15.0,  # angular speed range

    # reward shaping/scaling parameters (per step):
    'w_rescaling': 10,
    'w_goal': (1.0, 1.0),  # (pro, against)
    'w_move': (0.02, 1),  # (-1,1)
    'w_ball_grad': (0.08, 1),  # (-1,1)
    'w_collision': (0.000, 1),  # {-1,0}
    'w_ball_pot': (0.0, 1),  # (-1,0)
    'w_energy': (1e-5, 1),  # (-1,0)
    'min_dist_move_rw': 0.0,  # cm
    'collision_distance': 12,
    'alpha_base': [0.25, 0.5, 0.6],

    # Tracking and log
    'track_rewards': [False, False, True]
}


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


class MultiAgentSoccerEnvWrapper(gym.Wrapper):

    def __init__(self, env, simulator='fira', params=None):
        super().__init__(env)
        self.first = False
        self.soccer_env = env
        # This attributes set's if I should send random commands to the robots i'm not controlling with policy
        if params is None:
            if simulator in ['sdk', 'fira']:
                self.env_params = SDK_PARAMS if simulator == 'sdk' else FIRA_PARAMS
            elif simulator == 'real':
                self.env_params = REAL_PARAMS
            else:
                raise ValueError('Simulator not included in our list')
            self.n_agents = self.env_params['n_agents']
        else:
            self.n_agents = params['n_agents']
            self.env_params = params
        if self.soccer_env.env_context is None:
            self.soccer_env.set_parameters(self.env_params)
            self.soccer_env.start()
            self.observation_space, self.action_space = self.soccer_env.init_space_action()
            self.soccer_env.stop()
            self.soccer_env.env_context = EnvContext(self.n_agents)
            self.first = True  # only the first agent actually sends the commands

    def change_params(self, new_params):
        '''
        Changes environment parameters of your choice
        You always have to reset your environment after calling
        this method.

        Params -
            new_params: dict with which parameters you want to update

        Usage - 
            env.change_params({'frame_skip': 5})
            env.reset()
        '''
        self.env_params.update(new_params)
        self.soccer_env.set_parameters(self.env_params)

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
        # self.soccer_env.env_context.barrier.wait()
        # print("%d reset done." % self.agent_id)
        state = {f'agent_{i}': self.soccer_env.env_context.states[i] for i in range(self.n_agents)}
        return state

    def step(self, command):

        # if self.soccer_env.env_context.commands is None:
        #     if self.random_cmd:  # if send random commands to other agents
        #         cmd_array = [command]*len(self.soccer_env.team)
        #     else:
        #         cmd_array = [command]*self.n_agents

        #     self.soccer_env.env_context.commands = np.array(cmd_array)
        #     self.soccer_env.env_context.commands.fill(np.nan)

        # Collects the nth command
        for i in range(self.n_agents):
            self.soccer_env.env_context.commands[i] = command[i]

        try:
            # waits for the other commands
            # self.soccer_env.env_context.barrier.wait()

            if self.first:  # only the first agent actually calls the environments step

                if self.random_cmd:  # set up random commands for the other agents
                    for i in range(0, len(self.soccer_env.team)):
                        if np.isnan(self.soccer_env.env_context.commands[i]).any():
                            self.soccer_env.env_context.commands[i] = self.soccer_env.env_context.action_space.sample(
                            )

                # send commands:
                self.soccer_env.env_context.states, self.soccer_env.env_context.rewards, self.soccer_env.env_context.dones, _ = self.soccer_env.step(
                    self.soccer_env.env_context.commands)
                # clear commands:
                self.soccer_env.env_context.commands.fill(np.nan)

            if self.soccer_env.env_context.states is None:
                return None, 0, True, {}

            # waits for the other commands
            # self.soccer_env.env_context.barrier.wait()
            states = {f'agent_{i}': self.soccer_env.env_context.states[i] for i in range(self.n_agents)}
            rewards = {f'agent_{i}': self.soccer_env.env_context.rewards[i] for i in range(self.n_agents)}
            dones = {f'agent_{i}': self.soccer_env.env_context.dones[i] for i in range(self.n_agents)}
            return states, rewards, dones, {}

        except threading.BrokenBarrierError:
            return None, 0, True, {}

    def seed(self, seed=None):
        return [self.soccer_env.seed(seed)]

    def render(self):
        self.change_params({'render': True})
        self.reset()

    def close(self):
        self.stop()
