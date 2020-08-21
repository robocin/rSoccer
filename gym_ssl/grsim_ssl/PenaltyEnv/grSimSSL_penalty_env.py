import gym
import math
import numpy as np

from gym_ssl.grsim_ssl.Utils import distance
from gym_ssl.grsim_ssl.Entities import Ball, Frame, Robot
from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient

class GrSimSSLPenaltyEnv(GrSimSSLEnv):
    """
    Description:
        This environment controls a robot soccer goalkeeper in a Robocup 
        Small Size League penalty situation
    Observation:
        Type: Box(11)
        Num     Observation                         Min         Max
        0       Ball X   (mm)                       -7000       7000
        1       Ball Y   (mm)                       -6000       6000
        2       Ball Vx  (mm/s)                     -5000       5000
        3       Ball Vy  (mm/s)                     -5000       5000
        6       id 0 Blue Robot X       (mm)        -7000       7000
        4       id 0 Blue Robot Y       (mm)        -6000       6000
        9       id 0 Blue Robot Vx      (mm/s)      -5000       5000
        5       id 0 Blue Robot Vy      (mm/s)      -5000       5000
        6       id 0 Yellow Robot X     (mm)        -7000       7000
        7       id 0 Yellow Robot Y     (mm)        -6000       6000
        8       id 0 Yellow Robot sin(Angle)        -1          1
        9       id 0 Yellow Robot cos(Angle)        -1          1
        9       id 0 Yellow Robot Vx    (mm/s)      -5000       5000
        10      id 0 Yellow Robot Vy    (mm/s)      -5000       5000
        11      id 0 Yellow Robot Vw    (rad/s)     -10         10
    Actions:
        Type: Box(1)
        Num     Action                        Min       Max
        0       id 0 Blue Team Robot Vy       -3        3
        Note: Global reference
    Reward:
        # 1 if NOT GOAL
        # -1 * Ball distance from goalkeeper (meters) if GOAL
    Starting State:
        Ball on penalty position, goalkeeper on goal line in center of goal,
        and attacker behind ball facing goal
    Episode Termination:
        Ball crosses goal line, or ball has a velocity opposite to goal after attacker kick
    """

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)
        # Observation Space thresholds
        obsSpaceThresholds = np.array([7000, 6000, 5000, 5000, 7000, 6000, 5000, 5000, 7000, 6000,
                                       1, 1, 5000, 5000, 10], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
        self.atkState = None
        self.kickTargetAngle = None
        self.AttackerVw = None
        self.AtackerVKick = None
        self.ball_x = None
        self.ball_y = None

        print('Environment initialized')

    def reset(self):
        self.atkState = 0
        self.AttackerVw = np.random.uniform(0.4, 1.2)
        self.AtackerVKick = np.random.uniform(4.5, 6.5)
        self.ball_x = np.random.uniform(-0.015,0.05)
        self.ball_y = np.random.uniform(-0.03,0.03)
        # get a random target kick angle between -20 and 20 degrees
        kickAngle = np.random.uniform(-0.4, 0.4)
        if kickAngle < 0:
            self.kickTargetAngle = -3.14 - kickAngle
        else:
            self.kickTargetAngle = 3.14 - kickAngle
        return super().reset()

    def _parseObservationFromState(self):
        observation = []

        observation.append(self.state.ball.x)
        observation.append(self.state.ball.y)
        observation.append(self.state.ball.vx)
        observation.append(self.state.ball.vy)

        observation.append(self.state.robotsBlue[0].x)
        observation.append(self.state.robotsBlue[0].y)
        observation.append(self.state.robotsBlue[0].vx)
        observation.append(self.state.robotsBlue[0].vy)

        observation.append(self.state.robotsYellow[0].x)
        observation.append(self.state.robotsYellow[0].y)
        observation.append(math.sin(self.state.robotsYellow[0].theta))
        observation.append(math.cos(self.state.robotsYellow[0].theta))
        observation.append(self.state.robotsYellow[0].vx)
        observation.append(self.state.robotsYellow[0].vy)
        observation.append(self.state.robotsYellow[0].vw)

        return np.array(observation)

    def _getFormation(self):
        # returns robot's and ball initial position        
        # ball penalty position
        ball = Ball(x=-4.8 + self.ball_x, y=0.0 + self.ball_y)

        robotPositions = []
        # Goalkeeper penalty position
        robotPositions.append(Robot(yellow=False, id=0, x=-6, y=0, theta=0))
        # Kicker penalty position
        robotPositions.append(Robot(yellow=True, id=0, x=-4, y=0, theta=180))

        return robotPositions, ball

    def _getAttackerCommand(self):
        cmdAttacker = Robot(yellow=True, id=0, dribbler=True)

        # attacker movement speed
        vx = np.clip(((-2) * (self.state.ball.x - self.state.robotsYellow[0].x) * (0.001)), 0.2, 1.5)

        # If atkState == 0 -> move to ball
        if self.atkState == 0:
            if self.state.robotsYellow[0].x <= -4700:
                cmdAttacker.vx = 0.0
                if self.kickTargetAngle < 0:
                    self.atkState = 1
                else:
                    self.atkState = 2
            else:
                cmdAttacker.vx = vx
        # If atkState == 1 -> rotate counterclockwise until kick angle        
        if self.atkState == 1:
            self.atkState = 0
            if -(abs(self.state.robotsYellow[0].theta)) < self.kickTargetAngle:
                cmdAttacker.vw = self.AttackerVw
                cmdAttacker.vx = 0.0
            else:
                cmdAttacker.dribbler = False
                cmdAttacker.vw = 0.0
                cmdAttacker.vx = 0.0
                cmdAttacker.kickVx = self.AtackerVKick
        # If atkState == 2 -> rotate clockwise until kick angle        
        if self.atkState == 2:
            self.atkState = 0
            if abs(self.state.robotsYellow[0].theta) > self.kickTargetAngle:
                cmdAttacker.vw = -self.AttackerVw
                cmdAttacker.vx = 0.0
            else:
                cmdAttacker.dribbler = False
                cmdAttacker.vw = 0.0
                cmdAttacker.vx = 0.0
                cmdAttacker.kickVx = self.AtackerVKick

        return cmdAttacker

    def _getCommands(self, actions):
        commands = []

        cmdGoalKeeper = self._getCorrectGKCommand(actions)

        commands.append(cmdGoalKeeper)

        cmdAttacker = self._getAttackerCommand()

        commands.append(cmdAttacker)

        return commands

    def _getCorrectGKCommand(self, vy):
        '''Control goalkeeper vw and vx to keep him at goal line'''
        cmdGoalKeeper = Robot(yellow=False, id=0, vy=vy)

        # Proportional Parameters for Vx and Vw
        KpVx = 0.005
        KpVw = 1.6
        # Error between goal line and goalkeeper
        errX = -6000 - self.state.robotsBlue[0].x
        # If the error is greater than 20mm, correct the goalkeeper
        if abs(errX) > 20:
            cmdGoalKeeper.vx = KpVx * errX
        else:
            cmdGoalKeeper.vx = 0.0
        # Error between the desired angle and goalkeeper angle
        errW = 0.0 - self.state.robotsBlue[0].theta
        # If the error is greater than 0.1 rad (5,73 deg), correct the goalkeeper
        if abs(errW) > 0.1:
            cmdGoalKeeper.vw = KpVw * errW
        else:
            cmdGoalKeeper.vw = 0.0

        return cmdGoalKeeper

    def _calculateRewardsAndDoneFlag(self):
        reward = 0
        done = False

        # If ball crosses the goal line
        if self.state.ball.x < -6025:
            done = True
            # If ball crosses goal line inside goal bounds GOAL
            if self.state.ball.y < 600 and self.state.ball.y > -600:
                # Reward based on distance from keeper to ball
                reward = abs(self.state.ball.y - self.state.robotsBlue[0].y) * (-0.001)
            else:
                # NOT GOAL
                reward = 1
        # If ball is moving away from goal after attacker kick NOT GOAL
        if self.state.ball.x < -5000:
            if self.state.ball.vx > 0:
                done = True
                reward = 1

        return reward, done
