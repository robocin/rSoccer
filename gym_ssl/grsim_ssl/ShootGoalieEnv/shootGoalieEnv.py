import gym
import math
import numpy as np
import time

from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient
from gym_ssl.grsim_ssl.Utils import mod
from gym_ssl.grsim_ssl.Entities import Ball, Frame, Robot
from gym_ssl.grsim_ssl.ShootGoalieEnv import shootGoalieState


class shootGoalieEnv(GrSimSSLEnv):

  """
  Using cartpole env description as base example for our documentation
  Description:
      # TODO
  Source:
      # TODO

  Observation:
      Type: Box(16)
      Num     Observation                                       Min                     Max
      0       Ball X   (mm)                                   -7000                   7000
      1       Ball Y   (mm)                                   -6000                   6000
      2       Ball Vx  (mm/s)                                 -10000                  10000
      3       Ball Vy  (mm/s)                                 -10000                  10000
      4       Blue id 0 Robot Vw       (rad/s)                -math.pi * 3            math.pi * 3
      5       Dist Blue id0 - goal center (mm)                -10000                  10000
      6       Angle between blue id 0 and goal left (rad)     -math.pi                math.pi
      7       Angle between blue id 0 and goal left (rad)     -math.pi                math.pi
      8       Angle between blue id 0 and goal right (rad)    -math.pi                math.pi
      9       Angle between blue id 0 and goal right (rad)    -math.pi                math.pi
      10      Angle between blue id 0 and goalie center(rad)  -math.pi                math.pi
      11      Angle between blue id 0 and goalie center(rad)  -math.pi                math.pi
      12      Angle between blue id 0 and goalie left (rad)   -math.pi                math.pi
      13      Angle between blue id 0 and goalie left (rad)   -math.pi                math.pi
      14      Angle between blue id 0 and goalie right (rad)  -math.pi                math.pi
      15      Angle between blue id 0 and goalie right (rad)  -math.pi                math.pi

  Actions:
      Type: Box(2)
      Num     Action                        Min                     Max
      0       Blue id 0 Vw (rad/s)        -math.pi * 3            math.pi * 3
      1       Blue Kick Strength (m/s)        -6.5                   6.5
  Reward:
      Reward is 1 for success, -1 to fails. 0 otherwise.

  Starting State:
      All observations are assigned a uniform random value in [-0.05..0.05]
      # TODO

  Episode Termination:
    # TODO
  """
  def __init__(self):
    super().__init__()
    ## Action Space
    actSpaceThresholds = np.array([math.pi * 3, 6.5], dtype=np.float32)
    self.action_space = gym.spaces.Box(low=-actSpaceThresholds, high=actSpaceThresholds)

    # Observation Space thresholds
    obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, math.pi * 3, 10000, math.pi, math.pi,
                                   math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.shootGoalieState = None

    print('Environment initialized')
  
  def _getCommands(self, actions):
    commands = []
    cmdAttacker = Robot(id=0, yellow=False, vw=actions[0], kickVx=actions[1], dribbler=True)
    # cmdAttacker = Robot(id=0, yellow=False, vw=0, kickVx=0, dribbler=True)
    
    commands.append(cmdAttacker)

    # TODO GOALIE
    # cmdAttacker = self._getAttackerCommand()

    # commands.append(cmdAttacker)

    return commands

  def _parseObservationFromState(self):
    observation = []

    self.shootGoalieState = shootGoalieState()
    observation = self.shootGoalieState.getObservation(self.state)

    return np.array(observation)

  def _getFormation(self):
    # ball penalty position
    ball = Ball(x=-4.1, y=0, vx=0, vy=0)
    
    # Goalkeeper penalty position
    goalKeeper = Robot(id=0, x=-6, y=0, theta=0, yellow = True)

    # Kicker penalty position
    attacker = Robot(id=0, x=-4, y=0, theta=180, yellow = False)

    return [goalKeeper, attacker], ball
    
  def _calculateRewardsAndDoneFlag(self):
    reward = 0
    done = False

    if self.state.ball.x < -6000:
      # the ball out the field limits
      done = True
      if self.state.ball.y < 600 and self.state.ball.y > -600:
          # ball entered the goal
          reward = 1
      else:
          # the ball went out the bottom line
          reward = -1
    elif self.state.ball.x < -5000 and self.state.ball.vx > -1:
        # goalkeeper caught the ball
      done = True
      reward = -1
    elif mod(self.state.ball.vx, self.state.ball.vy) < 10 and self.steps > 15: # 1 cm/s
      done = True
      reward = -1
    elif  self.steps > 125:
      done = True
      reward = 0
    return reward, done
