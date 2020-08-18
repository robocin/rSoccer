import gym
import math
import numpy as np
import time
import gym_ssl.grsim_ssl.Communication.pb.grSim_Packet_pb2 as packet_pb2

from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient
from gym_ssl.grsim_ssl.Entities.Ball import Ball
from gym_ssl.grsim_ssl.Entities.Robot import Robot
from gym_ssl.grsim_ssl.Entities.Frame import Frame
from gym_ssl.grsim_ssl.goalieState import GoalieState


class shootGoalieEnv(GrSimSSLEnv):
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
      12       Angle between blue id 0 and goalie left (rad)   -math.pi                math.pi
      13       Angle between blue id 0 and goalie left (rad)   -math.pi                math.pi
      14      Angle between blue id 0 and goalie right (rad)  -math.pi                math.pi
      15      Angle between blue id 0 and goalie right (rad)  -math.pi                math.pi

  Actions:
      Type: Box(2)
      Num     Action                        Min                     Max
      0       Blue id 0 Vw (rad/s)        -math.pi * 3            math.pi * 3
      1       Blue Kick Strength (m/s)          -6.5                   6.5 <-------------------- VERIFY!!!!

      Note: The amount the velocity that is reduced or increased is not
      fixed; it depends on the angle the pole is pointing. This is because
      the center of gravity of the pole increases the amount of energy needed
      to move the cart underneath it
  Reward:
      Reward is 1 for success, -1 to fails. 0 otherwise.

  Starting State:
      All observations are assigned a uniform random value in [-0.05..0.05]
      # Algulo aleatorio na posicao do penalty

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
    actSpaceThresholds = np.array([math.pi * 3, 6.5], dtype=np.float32)
    self.action_space = gym.spaces.Box(low=-actSpaceThresholds, high=actSpaceThresholds)
    # Observation Space thresholds
    obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, math.pi * 3, 10000, math.pi, math.pi,
                                   math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.state = None
    self.shootGoalieState = None
    print('Environment initialized')
  
  def _getCommands(self, actions):
    commands = []
    cmdAttacker = Robot(id=0, yellow=False, vw=actions[0], kickVx=actions[1], dribbler=True)
    
    commands.append(cmdAttacker)

    # TODO GOALIE
    # cmdAttacker = self._getAttackerCommand()

    # commands.append(cmdAttacker)

    return commands

  def _parseObservationFromState(self):
    observation = []

    self.shootGoalieState = GoalieState()
    observation = self.shootGoalieState.getObservation(self.state)

    return np.array(observation)

  def _getFormation(self):
    # ball penalty position
    ball = Ball(x=-4.2, y=0, vx=1, vy=0)
    
    # Goalkeeper penalty position
    goalKeeper = Robot(id=0, x=-6, y=0, w=0, yellow = True)

    # Kicker penalty position
    attacker = Robot(id=0, x=-4, y=0, w=180, yellow = False)

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

    if self.state.ball.x < -5000:
        # goalkeeper caught the ball
        if self.state.ball.vx > -1:
            done = True
            reward = -1

    return reward, done
