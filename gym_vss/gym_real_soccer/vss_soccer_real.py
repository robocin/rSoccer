import gym
from gym_vss.vss_player import VSSPlayer
import numpy as np
#from gym_vss.gym_real_soccer.sslparser import SSLParser
from gym_vss.gym_real_soccer.vssparser import VSSParser
from gym_vss.utils import *
#from .nrfparser import NRFParser

import os
from gym import spaces
from gym import utils
from gym.utils import seeding

import torch
import random
from collections import deque
import time
from gym_vss.vss_soccer import VSSSoccerEnv
from gym_vss.spline import SplineRandomizer

from .action_manager import ActionManager


class RealSoccerEnv(VSSSoccerEnv):
    def __init__(self):
        super(RealSoccerEnv, self).__init__()

        self.is_paused = True

        # -- Real parameters
        self.ssl_parser = None
        self.nrf_parser = None

        # -- Randomizer
        self.randomizer_left = SplineRandomizer()
        self.randomizer_right = SplineRandomizer()

        self.action_manager = ActionManager()

    def process_input(self):
        if self.action_manager.has_event:

            if self.action_manager.event_type == 'p':
                if self.action_manager.is_paused:
                    self.pause()
                else:
                    self.resume()

            elif self.action_manager.event_type == 'c':
                if self.action_manager.is_yellow:
                    self.setTeamYellow()
                else:
                    self.setTeamBlue()

            elif self.action_manager.event_type == 's':
                self.stop_all()

            self.action_manager.clearEvent()

        return False

    # Env methods
    # ----------------------------

    def start(self):
        # self.vision_parser = SSLParser(self.ip, self.port)
        self.vision_parser = VSSParser(self.ip, self.port)
        #self.ctrl_parser = NRFParser()

    def stop(self):
        pass

    def reset(self):
        self.is_first_iteration = True
        self.vision_parser.reset()
        state = self._receive_state()
        self.env_start_time = state.time
        self.last_state, reward, done = self._parse_state(state, self.cmd_wait)

        return self.last_state

    # Extension methods
    # ----------------------------

    def _receive_state(self):
        return self.vision_parser.receive()

    def _set_action(self, commands):

        for i in range(0, len(commands)):

            rbt = self.team[i]

            rbt.write_log([self.ball_x, self.ball_y])

            rbt.update_targets()

            # calculate new (target_rho, target_theta) based on the command
            rbt.target_rho = clip(rbt.target_rho + self.action_dict[commands[i]][1], -60, 60)
            rbt.target_theta = to_pi_range(rbt.target_theta + self.action_dict[commands[i]][0])

            # update target x and y regarding the new (target_rho, target_theta)
            rbt.target_x = clip(rbt.x + rbt.target_rho * math.cos(rbt.target_theta), 0, 170)
            rbt.target_y = clip(rbt.y + rbt.target_rho * math.sin(rbt.target_theta), 0, 130)

            # choose right front
            if rbt.target_rho < 0:
                rbt_theta = to_pi_range(rbt.theta + math.pi)
                cmd_theta = to_pi_range(rbt.target_theta + math.pi)
            else:
                rbt_theta = rbt.theta
                cmd_theta = rbt.target_theta

            rbt.angular_speed_desired = clip(-7.5 * smallest_angle_diff(cmd_theta, rbt_theta), -30.0, 30.0)  # * 0.7211
            rbt.linear_speed_desired = 1.0 * rbt.target_rho  # / 1.3

            # For run ctrl model
            if i == 0 and (ctrl_mode & CTRL_CORRECT):
                rbt.update_wheel_sdps_model()
            elif (ctrl_mode & CTRL_RANDOM_WALK):
                rbt.update_wheel_sdps_random()
            elif (ctrl_mode & CTRL_SPLINE):
                rbt.update_wheel_sdps_spline()
            else:
                rbt.update_wheel_sdps_analytic()

        self._send_wheel_speeds()

    def _send_wheel_speeds(self):
        pass
        #self.nrf_parser.send_speeds(int(self.team[0].left_wheel_speed), int(self.team[0].right_wheel_speed), 0)

