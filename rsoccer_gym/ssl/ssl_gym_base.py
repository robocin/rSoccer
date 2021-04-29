'''
#   Environment Communication Structure
#    - Father class that creates the structure to communicate with multples setups of enviroment
#    - To create your wrapper from env to communcation, use inherit from this class! 
'''


import time
from typing import Dict, List, Optional

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Field
from rsoccer_gym.Simulators.rsim import RSimSSL


class SSLBaseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }
    NORM_BOUNDS = 1.2

    def __init__(self, field_type: int,
                 n_robots_blue: int, n_robots_yellow: int, time_step: float):
        # Initialize Simulator
        self.time_step = time_step
        self.rsim = RSimSSL(field_type=field_type,
                            n_robots_blue=n_robots_blue,
                            n_robots_yellow=n_robots_yellow,
                            time_step_ms=int(self.time_step*1000))
        self.n_robots_blue: int = n_robots_blue
        self.n_robots_yellow: int = n_robots_yellow
        
        # Get field dimensions
        self.field_type: int = field_type
        self.field: Field = self.rsim.get_field_params()
        self.max_pos = max(self.field.width / 2, (self.field.length / 2) 
                                + self.field.penalty_length
                                )
        max_wheel_rad_s = (self.field.rbt_motor_max_rpm / 60) * 2 * np.pi
        self.max_v = max_wheel_rad_s * self.field.rbt_wheel_radius
        # 0.04 = robot radius (0.09) + wheel thicknees (0.005)
        self.max_w = np.rad2deg(self.max_v / 0.095)

        # Initiate 
        self.frame: Frame = None
        self.last_frame: Frame = None
        self.view = None
        self.steps = 0
        self.sent_commands = None

    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands: List[Robot] = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        return observation, reward, done, {}

    def reset(self):
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        # Close render window
        del(self.view)
        self.view = None

        initial_pos_frame: Frame = self._get_initial_positions_frame()
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()

        return self._frame_to_observations()

    def render(self, mode: Optional = 'human') -> None:
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
            from rsoccer_gym.Render import RCGymRender
            self.view = RCGymRender(self.n_robots_blue,
                                 self.n_robots_yellow,
                                 self.field,
                                 simulator='ssl')

        return self.view.render_frame(self.frame, return_rgb_array=mode == "rgb_array")

    def close(self):
        self.rsim.stop()

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
    
    def norm_pos(self, pos):
        return np.clip(
            pos / self.max_pos,
            -self.NORM_BOUNDS,
            self.NORM_BOUNDS
        )

    def norm_v(self, v):
        return np.clip(
            v / self.max_v,
            -self.NORM_BOUNDS,
            self.NORM_BOUNDS
        )

    def norm_w(self, w):
        return np.clip(
            w / self.max_w,
            -self.NORM_BOUNDS,
            self.NORM_BOUNDS
        )
