import gym
from math import *
import numpy as np
import time
import random

from rc_gym.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from rc_gym.grsim_ssl.Communication.grSimClient import grSimClient
from rc_gym.grsim_ssl.Entities import Robot, Ball, Frame
from rc_gym.grsim_ssl.GoToBallEnv import goToBallState
from rc_gym.Utils import *

class goToBallEnv(GrSimSSLEnv):
  """
  Using cartpole env description as base example for our documentation
  Description:
      # TODO
  Source:
      # TODO

  Observation:
      Type: Box(14)
      Num     Observation                                       Min                     Max
      0       Ball X   (mm)                                   -7000                   7000
      1       Ball Y   (mm)                                   -6000                   6000
      2       Blue id 0 Vx  (mm/s)                            -10000                  10000
      3       Blue id 0 Vy  (mm/s)                            -10000                  10000
      4       Blue id 0 Robot Vw       (rad/s)                -math.pi * 3            math.pi * 3
      5       Dist Blue id0 - ball (mm)                       -10000                  10000
      6       Position Wall Top x                             -10000                  10000
      7       Position Wall Top y                             -10000                  10000
      8       Position Wall Bottom x                          -10000                  10000
      9       Position Wall Bottom y                          -10000                  10000
      10      Position Wall Left x                            -10000                  10000
      11      Position Wall Left y                            -10000                  10000
      12      Position Wall Right x                           -10000                  10000
      13      Position Wall Right y                           -10000                  10000
      

  Actions:
      Type: Box(2)
      Num     Action                        Min                     Max
      0       Blue id 0 Vx                  -10                      10
      1       Blue id 0 Vy                  -10                      10
      2       Blue id 0 Vw (rad/s)        -math.pi * 3            math.pi * 3
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
    actSpaceThresholds = np.array([10, 10, math.pi * 3], dtype=np.float32)
    self.action_space = gym.spaces.Box(low=-actSpaceThresholds, high=actSpaceThresholds)

    # Observation Space thresholds
    obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, math.pi * 3, 10000, 10000, 10000,
                                   10000, 10000, 10000, 10000, 10000, 10000], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.goToballState = None
    self.distAnt = 1000
    self.angleAnt = 0
    self.timestampAnt = 0
    print('Environment initialized')
  
  def _getCommands(self, actions):
    commands = []
    cmdAttacker = Robot(id=0, yellow=False, vx=actions[0], vy=actions[1], vw=actions[2])
    # cmdAttacker = Robot(id=0, yellow=False, vw=0, kickVx=0, dribbler=True)
    
    commands.append(cmdAttacker)

    # TODO GOALIE
    # cmdAttacker = self._getAttackerCommand()

    # commands.append(cmdAttacker)

    return commands

  def _parseObservationFromState(self):
    observation = []

    self.goToBallState = goToBallState()
    observation = self.goToBallState.getObservation(self.state)

    return np.array(observation)

  def _getFormation(self):
    #To CHANGE: 
    # ball penalty position
    #ball = Ball(x=random.uniform(-4, 0), y=random.uniform(-4, 4), vx=0, vy=0)
    ball = Ball(x=random.uniform(-4.5,-3.5), y=random.uniform(-1.5, 1.5), vx=0, vy=0)
    
    # Goalkeeper penalty position
    #goalKeeper = Robot(id=0, x=-6, y=0, theta=0, yellow = True)
    goalKeeper = Robot(id=0, x=6, y=1, theta=0, yellow = True)

    # Kicker penalty position
    #attacker = Robot(id=0, x=random.uniform(-3.5, 0), y=random.uniform(-4, 4), theta=180, yellow = False)
    attacker = Robot(id=0, x=random.uniform(-3,-1), y=random.uniform(-2.5, 2.5), theta=180, yellow = False)

    return [goalKeeper, attacker], ball
    
  def _calculateRewardsAndDoneFlag(self):
    reward = 0
    rewardContact = 0
    rewardDistance = 0
    rewardAngle =0
    done = False
    #print(self.state.timestamp)

    
    dt = self.state.timestamp - self.timestampAnt

    if(dt==0):
      dt =1

    rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance*0.001)**2 + self.goToBallState.angle_relative**2) / 2) - 2
    #rewardDistance -= (0.1*(self.goToBallState.distance - self.distAnt))/dt
    #rewardAngle -=(0.02*(self.goToBallState.angle_relative - self.angleAnt))/dt

    self.distAnt = self.goToBallState.distance
    self.timestampAnt = self.state.timestamp
    self.angleAnt = self.goToBallState.angle_relative

    if self.state.ball.x < -6000 or self.state.ball.y > 4500 or self.state.ball.y < -4500:
      # the ball out the field limits
      done = True
      rewardContact += 0
      
    elif  self.steps > 250:
      #finished the episode
      done = True
      if (self.goToBallState.distance*0.001 <= 0.130 and abs(self.goToBallState.angle_relative)<0.5):
        #print("OI")
        rewardContact += 100
      #rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance*0.001)**2 + self.goToBallState.angle_relative**2) / 2) - 2
    else:
      # the ball in the field limits
      if (self.goToBallState.distance*0.001 <= 0.130 and abs(self.goToBallState.angle_relative)<0.5):
        #print("OI")
        rewardContact += 100
      #rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance*0.001)**2 + self.goToBallState.angle_relative**2) / 2) - 2 

    reward = rewardContact + rewardDistance + rewardAngle
    #print(reward)
    return reward, done