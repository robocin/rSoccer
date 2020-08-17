import gym
import math
import numpy as np
import gym_ssl.grsim_ssl.Communication.pb.grSim_Packet_pb2 as packet_pb2

from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient
from gym_ssl.grsim_ssl.Entities.Ball import Ball
from gym_ssl.grsim_ssl.Entities.Robot import Robot
from gym_ssl.grsim_ssl.Entities.Frame import Frame
from gym_ssl.grsim_ssl.shootGoalie.State import State


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
      Type: Box(3)
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
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    # Observation Space thresholds
    obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, 6000, 10000, 7000,
                                   math.pi, 6000, 1000, math.pi * 3], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.state=None
    print('Environment initialized')
  
  def _getCommands(self, actions):
    pass
  def _parseObservationFromState(self, frame):
    self.state = State()
    self.state.ball_x = frame.ball.x
    self.state.ball_y = frame.ball.y
    self.state.ball_vx = frame.ball.vx
    self.state.ball_vy = frame.ball.vy
    
    self.state.distance = self.state.getDistance(frame)
    self.state.robot_w = frame.robots_blue[0].vw
    self.state.theta_l_sen, self.state.theta_l_cos = self.state.get_l_angle(frame)
    self.state.theta_r_sen, self.state.theta_r_cos = self.state.get_r_angle(frame)
    self.state.theta_goalie_c_sen, self.state.theta_goalie_c_cos = self.state.get_goalie_c_angle(frame)
    self.state.theta_goalie_l_sen, self.state.theta_l_cos = self.state.get_goalie_l_angle(frame)
    self.state.theta_goalie_r_sen, self.state.theta_r_cos = self.state.get_goalie_r_angle(frame)
    return self.state
  def _getFormation(self):
    # ball penalty position
    ball = Ball()
    ball.x = -4.8
    ball.y = 0
    ball.vx = 0
    ball.vy = 
    # Goalkeeper penalty position
    goalKeeper = Robot()
    goalKeeper.x = -6
    goalKeeper.y = 0
    goalKeeper.w = 0
    goalKeeper.id = 0
    goalKeeper.yellow = Tru
    # Kicker penalty position
    attacker = Robot()
    attacker.x = -4
    attacker.y = 0
    attacker.w = 180
    attacker.id = 0
    attacker.yellow = Tru
    return [[goalKeeper, attacker], ball]
    
  def _calculateRewardsAndDoneFlag(self):
    return 0, False
