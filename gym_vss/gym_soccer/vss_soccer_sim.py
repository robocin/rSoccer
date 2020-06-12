import os
import random
import threading

from gym_vss.vss_soccer import *

from .sim_parser import *
from .fira_parser import *


class SimSoccerEnv(VSSSoccerEnv):

    barrier = None

    def __init__(self):
        super(SimSoccerEnv, self).__init__()

        # -- Simulator parameters
        self.prev_time = 0
        self.simulator_path = None
        self.ia = False
        self.init_agents = 2
        self.init_ball = 4
        self.sim = None
        self.manage_process = True

    # Simulation methods
    # ----------------------------

    def set_parameters(self, parameters):
        super(SimSoccerEnv, self).set_parameters(parameters)
        self.simulator_type = FiraParser if parameters[
            'simulator_type'] == 'FiraParser' else SimParser
        params = ['simulator_path', 'ia', 'init_agents',
                  'init_ball', 'port', 'fast_mode', 'render']
        for param in params:
            if param in parameters.keys():
                setattr(self, param, parameters[param])

    def set_synchronize_resets(self):
        if type(self).barrier is None:
            type(self).barrier = threading.Barrier(2)

    def __del__(self):
        if self.sim is not None:
            self.sim.stop()

    # Env methods
    # ----------------------------

    def start(self, manage_process=True):
        self.manage_process = manage_process
        if isinstance(self.simulator_type, FiraParser):
            self.sim = self.simulator_type(self.ip, self.port,
                                           self.is_team_yellow,
                                           self.team_name, manage_process,
                                           self.fast_mode, self.render)
        else:
            self.sim = SimParser(self.ip, self.port,
                                 self.is_team_yellow,
                                 self.team_name, manage_process)
        self.sim.set_parameters(self.simulator_path, self.command_rate,
                                self.ia, self.init_agents, self.init_ball)
        self.sim.start()

    def stop(self):
        if self.sim is not None:
            self.sim.stop()
            self.sim = None

    def reset(self):
        if self.sim is None:
            self.start(self.manage_process)

        # slave waits manager to reset
        if not self.manage_process and type(self).barrier is not None:
            # print(self.team_name + " slave waits")
            type(self).barrier.wait()

        # manager resets first:
        self.sim.reset()

        # manager waits slave to reset
        if self.manage_process and type(self).barrier is not None:
            # print(self.team_name + " manager waits")
            type(self).barrier.wait()

        result = super(SimSoccerEnv, self).reset()

        # wait both to be good to goal
        # print(self.team_name + " will continue")
        if type(self).barrier is not None:
            type(self).barrier.wait()

        # print(self.team_name + " go!")
        return result

    # Extension methods
    # ----------------------------

    def _receive_state(self):
        state = None

        while state is None and self.sim.is_running:
            state = self.sim.receive()

        return state

    def _set_action(self, commands):

        for i in range(0, len(commands)):

            rbt = self.team[i]

            rbt.write_log([self.ball_x, self.ball_y])

            rbt.update_targets()

            # calculate new (target_rho, target_theta) based on the command
            rbt.target_rho = clip(
                rbt.target_rho + self.action_dict[commands[i]][1], -60, 60)
            rbt.target_theta = to_pi_range(
                rbt.target_theta + self.action_dict[commands[i]][0])

            # update target x and y regarding the new (target_rho, target_theta)
            rbt.target_x = clip(rbt.x + rbt.target_rho *
                                math.cos(rbt.target_theta), 0, 170)
            rbt.target_y = clip(rbt.y + rbt.target_rho *
                                math.sin(rbt.target_theta), 0, 130)

            # choose right front
            if rbt.target_rho < 0:
                rbt_theta = to_pi_range(rbt.theta + math.pi)
                cmd_theta = to_pi_range(rbt.target_theta + math.pi)
            else:
                rbt_theta = rbt.theta
                cmd_theta = rbt.target_theta

            # default control
            rbt.angular_speed_desired = clip(
                -7.5 * smallest_angle_diff(cmd_theta, rbt_theta), -30, 30)
            rbt.linear_speed_desired = 1.5 * rbt.target_rho

            # calculate wheels' linear speeds:
            rbt.left_wheel_speed = clip(
                rbt.linear_speed_desired - self.robot_l*rbt.angular_speed_desired, -100, 100)
            rbt.right_wheel_speed = clip(
                rbt.linear_speed_desired + self.robot_l*rbt.angular_speed_desired, -100, 100)

            # # random walk
            # rbt.left_wheel_speed = (0.9*rbt.left_wheel_speed + 0.1*random.uniform(-100, 100))
            # rbt.right_wheel_speed = (0.9*rbt.right_wheel_speed + 0.1*random.uniform(-100, 100))
            #
            # # Correcting desired angular and linear speeds for random walk
            # rbt.angular_speed_desired = (rbt.right_wheel_speed - rbt.left_wheel_speed) / (2.0 * self.robot_l)
            # rbt.linear_speed_desired = (rbt.right_wheel_speed + rbt.left_wheel_speed) / 2.0

            # Assumes energy consumption is proportional to wheel speeds:
            rbt.energy = abs(rbt.left_wheel_speed) + abs(rbt.right_wheel_speed)

        self.sim.send_speeds(self.team)
