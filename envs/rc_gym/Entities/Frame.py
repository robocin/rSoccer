import numpy as np
from typing import Dict
from rc_gym.Entities.Ball import Ball
from rc_gym.Entities.Robot import Robot
from dataclasses import dataclass


class Frame:
    """Units: seconds, m, m/s, degrees, degrees/s. Reference is field center."""

    def __init__(self):
        """Init Frame object."""
        self.ball = Ball()
        self.robots_blue = {}
        self.robots_yellow = {}


class FrameVSS(Frame):
    def parse(self, state, n_robots_blue=3, n_robots_yellow=3):
        """It parses the state received from grSim in a common state for environment"""
        self.ball.x = state[0]
        self.ball.y = state[1]
        self.ball.z = state[2]
        self.ball.v_x = state[3]
        self.ball.v_y = state[4]
        self.ball.x

        for i in range(n_robots_blue):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + (6 * i) + 0]
            robot.y = state[5 + (6 * i) + 1]
            robot.theta = state[5 + (6 * i) + 2]
            robot.v_x = state[5 + (6 * i) + 3]
            robot.v_y = state[5 + (6 * i) + 4]
            robot.v_theta = state[5 + (6 * i) + 5]
            self.robots_blue[robot.id] = robot

        for i in range(n_robots_yellow):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + n_robots_blue * 6 + (6 * i) + 0]
            robot.y = state[5 + n_robots_blue * 6 + (6 * i) + 1]
            robot.theta = state[5 + n_robots_blue * 6 + (6 * i) + 2]
            robot.v_x = state[5 + n_robots_blue * 6 + (6 * i) + 3]
            robot.v_y = state[5 + n_robots_blue * 6 + (6 * i) + 4]
            robot.v_theta = state[5 + n_robots_blue * 6 + (6 * i) + 5]
            self.robots_yellow[robot.id] = robot


class FrameSSL(Frame):
    def parse(self, state, n_robots_blue=3, n_robots_yellow=3):
        """It parses the state received from grSim in a common state for environment"""
        self.ball.x = state[0]
        self.ball.y = state[1]
        self.ball.z = state[2]
        self.ball.v_x = state[3]
        self.ball.v_y = state[4]
        self.ball.x

        for i in range(n_robots_blue):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + (7 * i) + 0]
            robot.y = state[5 + (7 * i) + 1]
            robot.theta = state[5 + (7 * i) + 2]
            robot.v_x = state[5 + (7 * i) + 3]
            robot.v_y = state[5 + (7 * i) + 4]
            robot.v_theta = state[5 + (7 * i) + 5]
            robot.infrared = bool(state[5 + (7 * i) + 6])
            self.robots_blue[robot.id] = robot

        for i in range(n_robots_yellow):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + n_robots_blue * 7 + (7 * i) + 0]
            robot.y = state[5 + n_robots_blue * 7 + (7 * i) + 1]
            robot.theta = state[5 + n_robots_blue * 7 + (7 * i) + 2]
            robot.v_x = state[5 + n_robots_blue * 7 + (7 * i) + 3]
            robot.v_y = state[5 + n_robots_blue * 7 + (7 * i) + 4]
            robot.v_theta = state[5 + n_robots_blue * 7 + (7 * i) + 5]
            robot.infrared = bool(state[5 + n_robots_blue * 7 + (7 * i) + 6])
            self.robots_yellow[robot.id] = robot


class FramePB(Frame):
    def parse(self, packet):
        """It parses the state received from grSim in a common state for environment"""

        self.ball.x = packet.frame.ball.x
        self.ball.y = packet.frame.ball.y
        self.ball.v_x = packet.frame.ball.vx
        self.ball.v_y = packet.frame.ball.vy

        for _robot in packet.frame.robots_blue:
            robot = Robot()
            robot.id = _robot.robot_id
            robot.x = _robot.x
            robot.y = _robot.y
            robot.theta = np.rad2deg(_robot.orientation)
            robot.v_x = _robot.vx
            robot.v_y = _robot.vy
            robot.v_theta = np.rad2deg(_robot.vorientation)

            self.robots_blue[robot.id] = robot

        for _robot in packet.frame.robots_yellow:
            robot = Robot()
            robot.id = _robot.robot_id
            robot.x = _robot.x
            robot.y = _robot.y
            robot.theta = np.rad2deg(_robot.orientation)
            robot.v_x = _robot.vx
            robot.v_y = _robot.vy
            robot.v_theta = np.rad2deg(_robot.vorientation)

            self.robots_yellow[robot.id] = robot
