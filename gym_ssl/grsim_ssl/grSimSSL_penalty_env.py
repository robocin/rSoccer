import gym
import math
import numpy as np
from gym_ssl.utils import * 

from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient
from gym_ssl.grsim_ssl.Entities.Ball import Ball
from gym_ssl.grsim_ssl.Entities.Robot import Robot
from gym_ssl.grsim_ssl.Entities.Frame import Frame


class GrSimSSLPenaltyEnv(GrSimSSLEnv):
    """
    Using cartpole env description as base example for our documentation
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
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

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(self):
        super().__init__()

        self.steps = 0
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
        ball = Ball()
        ball.x = -4.8
        ball.y = 0
        ball.vx = 0
        ball.vy = 0

        robotPositions = []

        # Goalkeeper penalty position
        goalKeeper = Robot()
        goalKeeper.x = -6
        goalKeeper.y = 0
        goalKeeper.w = 0
        goalKeeper.id = 0
        goalKeeper.yellow = False
        robotPositions.append(goalKeeper)

        # Kicker penalty position
        attacker = Robot()
        attacker.x = -4
        attacker.y = 0
        attacker.w = 180
        attacker.id = 0
        attacker.yellow = True
        robotPositions.append(attacker)

        return robotPositions, ball

    def _getAttackerCommand(self):
        cmdAttacker = Robot()
        cmdAttacker.yellow = True
        cmdAttacker.id = 0
        cmdAttacker.dribbler = True

        vw = 0.5
        vx = 0.7
        vkick = 5

        if self.atkState == 0:
            if self.state.robotsYellow[0].x < -4725:
                cmdAttacker.vx = 0.0
                if self.target < 0:
                    self.atkState = 1
                else:
                    self.atkState = 2
            else:
                cmdAttacker.vx = vx        
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
            #se mandar -0.05 gira pra direita e vai de 3.14 diminuindo
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

    def render(self):
        pass


    def _getCommands(self, actions):
        commands = []

        #cmdGoalKeeper = Robot()
        #cmdGoalKeeper.id = 0
        #cmdGoalKeeper.vy = actions
        #cmdGoalKeeper.yellow = False

        cmdGoalKeeper = self._getCorrectGKCommand(actions)

        commands.append(cmdGoalKeeper)

        cmdAttacker = self._getAttackerCommand()
        # cmdAttacker = Robot()
        # cmdAttacker.id = 0
        # cmdAttacker.yellow = True
        # cmdAttacker.vx = 0.5

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

        if self.state.ball.x < -6000:
            done = True
            if self.state.ball.y < 600 and self.state.ball.y > -600:
                reward = abs(self.state.ball.y - self.state.robotsYellow[0].y)*(-0.001)
            else:
                reward = 1

        if self.state.ball.x < -5000:
            if self.state.ball.vx > -1:
                done = True
                reward = 1

        if self.steps > 125:
            done = True

        return reward, done


     # def _calculateRewardsAndDoneFlag(self):
    #     reward = 0
    #     done = False
    #     self.steps += 1

    #     # Ball cross the line
    #     if self.state.ball.x < -6000:
    #         done = True

    #         # Goal
    #         if self.state.ball.y < 600 and self.state.ball.y > -600:
    #             reward = 0.5 * ((1/distance([self.state.ball.x, self.state.ball.y], \
    #                 [self.state.robotsBlue[0].x, self.state.robotsBlue[0].y])) - 5)
    #         # Ball Out
    #         else:
    #             reward = 1

    #     # Ball inside area    
    #     if self.state.ball.x < -5000:
    #         # Defend
    #         if self.state.ball.vx > -1:
    #             done = True
    #             reward = 1

    #     if self.steps > 120:
    #         done = True
        
    #     if done:
    #         print("Steps: ", self.steps)

    #     return reward, done
