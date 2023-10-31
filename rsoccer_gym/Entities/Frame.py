import numpy as np
from typing import Dict
from rsoccer_gym.Entities.Ball import Ball
from rsoccer_gym.Entities.Robot import Robot


class Frame:
    """Units: seconds, m, m/s, degrees, degrees/s. Reference is field center."""

    def __init__(self):
        """Init Frame object."""
        self.ball: Ball = Ball()
        self.robots_blue: Dict[int, Robot] = {}
        self.robots_yellow: Dict[int, Robot] = {}


class FrameVSS(Frame):
    def parse(self, state, n_blues=3, n_yellows=3):
        """It parses the state received from grSim in a common state for environment"""
        self.ball.x = state[0]
        self.ball.y = state[1]
        self.ball.z = state[2]
        self.ball.v_x = state[3]
        self.ball.v_y = state[4]

        rbt_obs = 6
        
        for i in range(n_blues):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + (rbt_obs*i) + 0]
            robot.y = state[5 + (rbt_obs*i) + 1]
            robot.theta = state[5 + (rbt_obs*i) + 2]
            robot.v_x = state[5 + (rbt_obs*i) + 3]
            robot.v_y = state[5 + (rbt_obs*i) + 4]
            robot.v_theta = state[5 + (rbt_obs*i) + 5]
            self.robots_blue[robot.id] = robot

        for i in range(n_yellows):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 0]
            robot.y = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 1]
            robot.theta = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 2]
            robot.v_x = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 3]
            robot.v_y = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 4]
            robot.v_theta = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 5]
            
            self.robots_yellow[robot.id] = robot


class FrameSSL(Frame):
    def parse(self, state, n_blues=3, n_yellows=3):
        """It parses the state received from grSim in a common state for environment"""
        self.ball.x = state[0]
        self.ball.y = state[1]
        self.ball.z = state[2]
        self.ball.v_x = state[3]
        self.ball.v_y = state[4]

        rbt_obs = 11
        
        for i in range(n_blues):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + (rbt_obs*i) + 0]
            robot.y = state[5 + (rbt_obs*i) + 1]
            robot.theta = state[5 + (rbt_obs*i) + 2]
            robot.v_x = state[5 + (rbt_obs*i) + 3]
            robot.v_y = state[5 + (rbt_obs*i) + 4]
            robot.v_theta = state[5 + (rbt_obs*i) + 5]
            robot.infrared = bool(state[5 + (rbt_obs*i) + 6])
            robot.v_wheel0 = state[5 + (rbt_obs*i) + 7]
            robot.v_wheel1 = state[5 + (rbt_obs*i) + 8]
            robot.v_wheel2 = state[5 + (rbt_obs*i) + 9]
            robot.v_wheel3 = state[5 + (rbt_obs*i) + 10]
            self.robots_blue[robot.id] = robot

        for i in range(n_yellows):
            robot = Robot()
            robot.id = i
            robot.x = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 0]
            robot.y = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 1]
            robot.theta = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 2]
            robot.v_x = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 3]
            robot.v_y = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 4]
            robot.v_theta = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 5]
            robot.infrared = bool(state[5 + n_blues*rbt_obs + (rbt_obs*i) + 6])
            robot.v_wheel0 = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 7]
            robot.v_wheel1 = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 8]
            robot.v_wheel2 = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 9]
            robot.v_wheel3 = state[5 + n_blues*rbt_obs + (rbt_obs*i) + 10]
            self.robots_yellow[robot.id] = robot
