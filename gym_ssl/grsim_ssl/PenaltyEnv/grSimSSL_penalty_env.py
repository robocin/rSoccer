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
        Num     Observation                         Min                     Max
        0       Ball X   (mm)                       -7000                   7000
        1       Ball Y   (mm)                       -6000                   6000
        2       Ball Vx  (mm/s)                     -10000                  10000
        3       Ball Vy  (mm/s)                     -10000                  10000
        4       id 0 Blue Robot Y       (mm)        -6000                   6000
        5       id 0 Blue Robot Vy      (mm/s)      -10000                  10000
        6       id 0 Yellow Robot X     (mm)        -7000                   7000
        7       id 0 Yellow Robot Y     (mm)        -6000                   6000
        8       id 0 Yellow Robot Angle (rad)       -math.pi                math.pi
        9       id 0 Yellow Robot Vx    (mm/s)      -10000                  10000
        10      id 0 Yellow Robot Vy    (mm/s)      -10000                  10000
        11      id 0 Yellow Robot Vy    (rad/s)     -math.pi * 3            math.pi * 3
    Actions:
        Type: Box(1)
        Num     Action                        Min                     Max
        0       id 0 Blue Team Robot Vy       -1                      1

        Note: Global reference
    Reward:
        # TODO
    Starting State:
        Ball on penalty position, goalkeeper on goal line in center of goal,
        and attacker behind ball facing goal
    Episode Termination:
        # TODO
    """

    def __init__(self):
        super().__init__()

        self.steps = 0
        self.maxSteps = 125
        self.deterministicAttacker = None
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Observation Space thresholds
        obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, 6000, 10000, 7000, 6000,
                                       math.pi, 10000, 10000, math.pi * 3], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)        
        self.atkState = None

        print('Environment initialized')
    
    def reset(self):

        self.steps = 0
        self.atkState = 0

        # get a random target kick angle between -20 and 20 degrees
        kickAngle = np.random.uniform(-0.5,0.5)
        if kickAngle < 0:
            self.target = -3.14 - kickAngle
        else:
            self.target = 3.14 - kickAngle
        
        return super().reset()

    def _parseObservationFromState(self):
        observation = []

        observation.append(self.state.ball.x)
        observation.append(self.state.ball.y)
        observation.append(self.state.ball.vx)
        observation.append(self.state.ball.vy)

        observation.append(self.state.robotsBlue[0].y)
        observation.append(self.state.robotsBlue[0].vy)

        observation.append(self.state.robotsYellow[0].x)
        observation.append(self.state.robotsYellow[0].y)
        observation.append(self.state.robotsYellow[0].w)
        observation.append(self.state.robotsYellow[0].vx)
        observation.append(self.state.robotsYellow[0].vy)
        observation.append(self.state.robotsYellow[0].vw)

        return np.array(observation)

    def _getFormation(self):
        # ball penalty position
        ball = Ball(x=-4.8, y=0.0)

        robotPositions = []
        # Goalkeeper penalty position
        robotPositions.append(Robot(yellow=False, id=0, x=-6, y=0, w=0))
        # Kicker penalty position
        robotPositions.append(Robot(yellow=True, id=0, x=-4, y=0, w=180))

        return robotPositions, ball

    def _getAttackerCommand(self):
        cmdAttacker = Robot(yellow=True, id=0, dribbler=True)

        vw = 0.5    # attacker rotation speed   
        vx = 0.7    # attacker movement speed
        vkick = 5   # attacker kick speed

        # If atkState == 0 -> move to ball
        if self.atkState == 0:
            if self.state.robotsYellow[0].x < -4725:
                cmdAttacker.vx = 0.0
                if self.target < 0:
                    self.atkState = 1
                else:
                    self.atkState = 2
            else:
                cmdAttacker.vx = vx
        # If atkState == 1 -> rotate counterclockwise until kick angle        
        if self.atkState == 1:
            self.atkState = 0       
            if -(abs(self.state.robotsYellow[0].w)) < self.target:
                cmdAttacker.vw = vw
                cmdAttacker.vx = 0.0
            else:
                cmdAttacker.dribbler = False
                cmdAttacker.vw = 0.0
                cmdAttacker.vx = 0.0
                cmdAttacker.kickVx = vkick
        # If atkState == 2 -> rotate clockwise until kick angle        
        if self.atkState == 2:
            self.atkState = 0
            if abs(self.state.robotsYellow[0].w) > self.target:
                cmdAttacker.vw = -vw
                cmdAttacker.vx = 0.0
            else:
                cmdAttacker.dribbler = False
                cmdAttacker.vw = 0.0
                cmdAttacker.vx = 0.0
                cmdAttacker.kickVx = vkick

        return cmdAttacker

    def _getCommands(self, actions):
        commands = []

        cmdGoalKeeper = self._getCorrectGKCommand(actions)

        commands.append(cmdGoalKeeper)

        cmdAttacker = self._getAttackerCommand()
        
        commands.append(cmdAttacker)

        return commands

    def _getCorrectGKCommand(self,actions):
        cmdGoalKeeper = Robot()
        cmdGoalKeeper.id = 0
        cmdGoalKeeper.vy = actions
        cmdGoalKeeper.yellow = False
        #Proportional Parameters for Vx and Vw
        Kp_vx = 0.005
        Kp_vw = 1.6
        #Error between goal line and goalkeeper
        err_x = -6000 - self.state.robotsBlue[0].x
        #If the error is greater than 20mm, correct the goalkeeper
        if abs(err_x) > 20:
            cmdGoalKeeper.vx = Kp_vx * err_x
        else:
            cmdGoalKeeper.vx = 0.0
        #Error between the desired angle and goalkeeper angle
        err_w = 0.0 - self.state.robotsBlue[0].w
        #If the error is greater than 0.1 rad (5,73 deg), correct the goalkeeper
        if abs(err_w) > 0.1:
            cmdGoalKeeper.vw = Kp_vw * err_w
        else:
            cmdGoalKeeper.vw = 0.0

        return cmdGoalKeeper

    def _calculateRewardsAndDoneFlag(self):
        reward = 0
        done = False

        # If ball crosses the goal line
        if self.state.ball.x < -6000:
            done = True
            # If ball crosses goal line inside goal bounds GOAL
            if self.state.ball.y < 600 and self.state.ball.y > -600:
                # Reward based on distance from keeper to ball
                reward = abs(self.state.ball.y - self.state.robotsYellow[0].y)*(-0.001)
            else:
                # NOT GOAL
                reward = 1
        # If ball is moving away from goal after attacker kick NOT GOAL
        if self.state.ball.x < -5000:
            if self.state.ball.vx > -1:
                done = True
                reward = 1

        # If exceed limit of episode steps
        if self.steps > self.maxSteps:
            done = True

        return reward, done
