import numpy as np
from rc_gym.Entities.Ball import Ball
from rc_gym.Entities.Robot import Robot


class Frame:
    """Units: seconds, m, m/s, degrees, degrees/s. Reference is field center."""
    def __init__(self):
        """Init Frame object."""
        self.ball = Ball()
        self.robots_blue = {}
        self.robots_yellow = {}
        self.time = None
        self.goals_yellow = None
        self.goals_blue = None

    def parse(self, state, status, n_robots_blue=3, n_robots_yellow=3):
        """It parses the state received from grSim in a common state for environment"""
        self.time = status['time_ms'] / 1000
        self.goals_yellow = status['goals_yellow']
        self.goals_blue = status['goals_blue']
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


class FrameSDK(Frame):
    def parse(self, packet, goal_depth=0.1):
        self.time = packet.time
        self.goals_blue = packet.goals_blue
        self.goals_yellow = packet.goals_yellow

        self.ball.x = packet.balls[0].pose.x/100 - 0.75 - goal_depth
        self.ball.y = 0.65 - packet.balls[0].pose.y/100
        # TODO GAMBIARRA POIS VELOCIDADE NO SDK TA DOBRADA
        self.ball.v_x = packet.balls[0].v_pose.x/100 / 2
        # TODO GAMBIARRA POIS VELOCIDADE NO SDK TA DOBRADA
        self.ball.v_y = -packet.balls[0].v_pose.y/100 / 2

        for i, _robot in enumerate(packet.robots_blue):
            robot = Robot()
            robot.id = i
            robot.x = _robot.pose.x/100 - 0.75 - goal_depth
            robot.y = 0.65 - _robot.pose.y/100
            robot.theta = -np.rad2deg(_robot.pose.yaw)
            # TODO GAMBIARRA POIS VELOCIDADE NO SDK TA DOBRADA
            robot.v_x = _robot.v_pose.x/100 / 2
            # TODO GAMBIARRA POIS VELOCIDADE NO SDK TA DOBRADA
            robot.v_y = -_robot.v_pose.y/100 / 2
            # TODO ta dobrado no pacote do sdk nao sabemos pq ainda
            robot.v_theta = np.rad2deg(_robot.v_pose.yaw) / 2
            self.robots_blue[robot.id] = robot

        for i, _robot in enumerate(packet.robots_yellow):
            robot = Robot()
            robot.id = i
            robot.x = _robot.pose.x/100 - 0.75 - goal_depth
            robot.y = 0.65 - _robot.pose.y/100
            robot.theta = -np.rad2deg(_robot.pose.yaw)
            # TODO GAMBIARRA POIS VELOCIDADE NO SDK TA DOBRADA
            robot.v_x = _robot.v_pose.x/100 / 2
            # TODO GAMBIARRA POIS VELOCIDADE NO SDK TA DOBRADA
            robot.v_y = -_robot.v_pose.y/100 / 2
            # TODO ta dobrado no pacote do sdk nao sabemos pq ainda
            robot.v_theta = np.rad2deg(_robot.v_pose.yaw) / 2
            self.robots_yellow[robot.id] = robot


class FramePB(Frame):
    def parse(self, packet):
        '''It parses the state received from grSim in a common state for environment'''
        self.timestamp = packet.detection.t_capture

        for _ball in packet.detection.balls:
            self.ball.x = _ball.x
            self.ball.y = _ball.y
            self.ball.v_x = _ball.vx
            self.ball.v_y = _ball.vy

        for _robot in packet.detection.robots_blue:
            robot = Robot()
            robot.id = _robot.robot_id
            robot.x = _robot.x
            robot.y = _robot.y
            robot.theta = _robot.orientation
            robot.v_x = _robot.vx
            robot.v_y = _robot.vy
            robot.v_theta = _robot.vorientation

            self.robots_blue[robot.id] = robot

        for _robot in packet.detection.robots_yellow:
            robot = Robot()
            robot.id = _robot.robot_id
            robot.x = _robot.x
            robot.y = _robot.y
            robot.theta = _robot.orientation
            robot.v_x = _robot.vx
            robot.v_y = _robot.vy
            robot.v_theta = _robot.vorientation

            self.robots_yellow[robot.id] = robot
