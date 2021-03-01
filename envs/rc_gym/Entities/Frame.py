import numpy as np
from typing import Dict
from rc_gym.Entities.Ball import Ball
from rc_gym.Entities.Robot import Robot

class Frame:
    """Units: seconds, m, m/s, degrees, degrees/s. Reference is field center."""
    
    def __init__(self):
        """Init Frame object."""
        self.ball: Ball = Ball()
        self.robots_blue: Dict[int, Robot] = {}
        self.robots_yellow: Dict[int, Robot] = {}

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
            robot.x = state[5 + n_robots_blue*6 + (6 * i) + 0]
            robot.y = state[5 + n_robots_blue*6 + (6 * i) + 1]
            robot.theta = state[5 + n_robots_blue*6 + (6 * i) + 2]
            robot.v_x = state[5 + n_robots_blue*6 + (6 * i) + 3]
            robot.v_y = state[5 + n_robots_blue*6 + (6 * i) + 4]
            robot.v_theta = state[5 + n_robots_blue*6 + (6 * i) + 5]
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
            robot.x = state[5 + n_robots_blue*7 + (7 * i) + 0]
            robot.y = state[5 + n_robots_blue*7 + (7 * i) + 1]
            robot.theta = state[5 + n_robots_blue*7 + (7 * i) + 2]
            robot.v_x = state[5 + n_robots_blue*7 + (7 * i) + 3]
            robot.v_y = state[5 + n_robots_blue*7 + (7 * i) + 4]
            robot.v_theta = state[5 + n_robots_blue*7 + (7 * i) + 5]
            robot.infrared = bool(state[5 + n_robots_blue*7 + (7 * i) + 6])
            self.robots_yellow[robot.id] = robot
