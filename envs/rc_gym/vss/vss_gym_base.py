'''
#   Environment Communication Structure
#    - Father class that creates the structure to communicate with multples setups of enviroment
#    - To create your wrapper from env to communcation, use inherit from this class! 
'''


import time
from typing import Dict, List

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot
from rc_gym.Utils import RCRender
from rc_gym.vss.Simulators.robosim.rsim import SimulatorVSS
# from rc_gym.vss.Simulators.vss_sdk.sdk import SimulatorVSS


class VSSBaseEnv(gym.Env):
    def __init__(self, field_type: int,
                 n_robots_blue: int, n_robots_yellow: int):
        self.time_step = 0.016
        self.simulator = SimulatorVSS(field_type=field_type,
                                      n_robots_blue=n_robots_blue,
                                      n_robots_yellow=n_robots_yellow,
                                      time_step_ms=int(self.time_step*1000))
        self.field_type: int = field_type
        self.n_robots_blue: int = n_robots_blue
        self.n_robots_yellow: int = n_robots_yellow
        self.field_params: Dict[str,
                                np.float64] = self.simulator.get_field_params()
        self.frame: Frame = None
        self.last_frame: Frame = None
        self.view = None
        self.steps = 0
        self.sent_commands = None

    def step(self, action):
        self.steps += 1

        # TODO talvez substituir o get commands por wrappers
        # Join agent action with environment actions
        commands: List[Robot] = self._get_commands(action)
        # Send command to simulator
        self.simulator.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.frame = self.simulator.get_frame()
        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        return observation, reward, done, {}

    def reset(self):
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None
        del(self.view)
        self.view = None
        initial_pos_frame: Frame = self._get_initial_positions_frame()
        self.simulator.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.simulator.get_frame()

        return self._frame_to_observations()

    def render(self, mode='human') -> None:
        '''
        Renders the game depending on 
        ball's and players' positions.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        if self.view == None:
            self.view = RCRender(
                self.n_robots_blue, self.n_robots_yellow, self.field_params)

        self.view.render_frame(self.frame)
        # if mode == 'human':
        #     time.sleep(0.01)

    def _get_commands(self, action):
        '''returns a list of commands of type List[Robot] from type action_space action'''
        raise NotImplementedError

    def _frame_to_observations(self):
        '''returns a type observation_space observation from a type List[Robot] state'''
        raise NotImplementedError

    def _calculate_reward_and_done(self):
        '''returns reward value and done flag from type List[Robot] state'''
        raise NotImplementedError

    def _get_initial_positions_frame(self) -> Frame:
        '''returns frame with robots initial positions'''
        raise NotImplementedError

    def close(self):
        self.simulator.stop()
