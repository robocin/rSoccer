import gym
import numpy as np
from envs.vss_agent import VSSAgent
from envs.gym_real_soccer.sslparser import SSLParser
from envs.gym_real_soccer.vssparser import VSSParser
from envs.utils import *
from .nrfparser import NRFParser
import math

import os
from gym import spaces
from gym import utils
from gym.utils import seeding

import torch
import random
import time
from envs.vss_soccer import VSSSoccerEnv
from envs.gym_real_soccer.PIDControl import PID

from .action_manager import ActionManager

class RealSoccerContinuousEnv(VSSSoccerEnv):
    def __init__(self):
        super(RealSoccerContinuousEnv, self).__init__()

        self.is_paused = True

        # -- Real parameters
        self.vision_parser = None
        self.ctrl_parser = None
        self.range_linear = 90.0
        self.range_angular = 10.0
        self.pulse_speed_ratio_l = [0.2, 0.2, 0.2]
        self.pulse_speed_ratio_r = [0.2, 0.2, 0.2]
        self.cmd_moving_average = 0.5

        self.angKp = 0.15
        self.angKi = 0.2
        self.angKd = 0

        self.linKp = 0.15
        self.linKi = 0.1
        self.linKd = 0.001

        self.angPID = [PID(self.angKp, self.angKi, self.angKd) for _ in range(3)]
        self.linPID = [PID(self.linKp, self.linKi, self.linKd) for _ in range(3)]

        # self.angPID = [PID(0.05, 0.2, 0.001) for _ in range(3)]
        # self.linPID = [PID(0.01, 0.5, 0.001) for _ in range(3)]

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
        #self.vision_parser = SSLParser(self.ip, self.port)
        self.vision_parser = VSSParser(self.ip, self.port)
        self.ctrl_parser = NRFParser()

    def reset(self):
        self.vision_parser.reset()
        return super(RealSoccerContinuousEnv, self).reset()

    def stop(self):
        pass

    def set_parameters(self, parameters):
        super(RealSoccerContinuousEnv, self).set_parameters(parameters)

        self.range_linear = parameters['range_linear']
        self.range_angular = parameters['range_angular']
        self.pulse_speed_ratio_l = parameters['pulse_speed_ratio_l']
        self.pulse_speed_ratio_r = parameters['pulse_speed_ratio_r']

        self.cmd_moving_average = parameters['cmd_moving_average']

        self.angKp = parameters['angular_speed_pid'][0]
        self.angKi = parameters['angular_speed_pid'][1]
        self.angKd = parameters['angular_speed_pid'][2]

        self.linKp = parameters['linear_speed_pid'][0]
        self.linKi = parameters['linear_speed_pid'][1]
        self.linKd = parameters['linear_speed_pid'][2]

        self.angPID = [PID(self.angKp, self.angKi, self.angKd) for _ in range(3)]
        self.linPID = [PID(self.linKp, self.linKi, self.linKd) for _ in range(3)]

    # Extension methods
    # ----------------------------

    def _receive_state(self):
        return self.vision_parser.receive()

    def _set_action(self, commands):

        for i in range(0, len(commands)):

            rbt = self.team[i]

            angError = rbt.angular_speed_obs - rbt.angular_speed_desired
            linError = rbt.linear_speed_obs - rbt.linear_speed_desired

            rbt.write_log([self.ball_x, self.ball_y])

            rbt.update_targets()

            command_angular_speed_desired = commands[i][0].item() * self.range_angular
            command_linear_speed_desired = commands[i][1].item() * self.range_linear

            # rbt.linear_speed_desired = command_linear_speed_desired
            # rbt.angular_speed_desired = command_angular_speed_desired

            # # Moving Average control:
            if rbt.linear_speed_desired is None:
                rbt.linear_speed_desired = command_linear_speed_desired
                rbt.angular_speed_desired = command_angular_speed_desired
            else:
                rbt.linear_speed_desired = self.cmd_moving_average * command_linear_speed_desired + (1 - self.cmd_moving_average ) * (rbt.linear_speed_desired)
                #rbt.angular_speed_desired = self.cmd_moving_average  * command_angular_speed_desired + (1 - self.cmd_moving_average ) * (rbt.angular_speed_desired)
                rbt.angular_speed_desired = command_angular_speed_desired

            # Overide control:
            # rbt.angular_speed_desired = 0
            # rbt.linear_speed_desired = 40

            # # PID control:
            # self.angPID[i].update(angError, self.running_time)
            # self.linPID[i].update(linError, self.running_time)

            #print(self.linPID[i].output, self.angPID[i].output)
            #rbt.angular_speed_desired = rbt.angular_speed_desired #+ clip(self.angPID[i].output, -10, 10)
            #rbt.linear_speed_desired = rbt.linear_speed_desired #+ clip(self.linPID[i].output, -30, 30)

            # update desired target x and y (used in parse state):
            rbt.target_rho = rbt.linear_speed_desired / 1.5
            rbt.target_theta = to_pi_range(rbt.theta + rbt.angular_speed_desired / -7.5)
            rbt.target_x = rbt.x + rbt.target_rho * math.cos(rbt.target_theta)
            rbt.target_y = rbt.y + rbt.target_rho * math.sin(rbt.target_theta)

            # calculate wheels' linear speeds:
            rbt.left_wheel_speed = rbt.linear_speed_desired - self.robot_l * rbt.angular_speed_desired
            rbt.right_wheel_speed = rbt.linear_speed_desired + self.robot_l * rbt.angular_speed_desired

            # rbt.left_wheel_speed = 45
            # rbt.right_wheel_speed = 45
            # Assumes energy consumption is proportional to wheel speeds:
            rbt.energy = abs(rbt.left_wheel_speed) + abs(rbt.right_wheel_speed)

            # For run ctrl model
            if i == 0 and (ctrl_mode & CTRL_CORRECT):
                rbt.update_wheel_sdps_model()
            elif (ctrl_mode & CTRL_RANDOM_WALK):
                rbt.update_wheel_sdps_random()
            elif (ctrl_mode & CTRL_SPLINE):
                rbt.update_wheel_sdps_spline()
            elif (ctrl_mode & CTRL_CUSTOM_SPEEDS):
                rbt.update_wheel_sdps_custom()
            #else:
                #rbt.update_wheel_sdps_analytic()

            self._send_wheel_speeds(i)

    def _send_wheel_speeds(self, idx):

        speed_ratio_l = self.pulse_speed_ratio_l[idx] * self.robot_r
        speed_ratio_r = self.pulse_speed_ratio_r[idx] * self.robot_r

        #print(self.team[idx].left_wheel_speed, self.team[idx].right_wheel_speed)
        clip_range = 50  #
        self.team[idx].left_wheel_speed = clip(int(round(speed_ratio_l*self.team[idx].left_wheel_speed)), -clip_range, clip_range)
        self.team[idx].right_wheel_speed = clip(int(round(speed_ratio_r*self.team[idx].right_wheel_speed)), -clip_range, clip_range)
        #print(self.team[idx].left_wheel_speed, self.team[idx].right_wheel_speed)

        self.ctrl_parser.send_speeds(self.team[idx].left_wheel_speed, self.team[idx].right_wheel_speed, idx)

    def stop_all(self):
        print('Stop Robots')
        for i in range(0, len(self.team)):
            rbt = self.team[i]
            rbt.left_wheel_speed = 0
            rbt.right_wheel_speed = 0

            for _ in range(0,10):
                self._send_wheel_speeds(i)
