from gym_ssl.util import *


CENTER_GOAL_X = -6.0
CENTER_GOAL_Y = 0.0

class State:
  def __init__(self):
    """Init Frame object."""
    self.ball_x = None
    self.ball_y = None
    self.ball_vx = None
    self.ball_vy = None
    self.robot_w = None
    self.distance = None
    self.theta_l_sen = None
    self.theta_l_cos = None
    self.theta_r_sen = None
    self.theta_r_cos = None
    self.theta_goalie_c_sen = None
    self.theta_goalie_c_cos = None
    self.theta_goalie_l_sen = None
    self.theta_goalie_l_cos = None
    self.theta_goalie_r_sen = None
    self.theta_goalie_r_cos = None
  
  def getDistance(self, frame):
    return mod(abs(frame.robots_blue[0].x-CENTER_GOAL_X), abs(frame.robots_blue[0].y-CENTER_GOAL_Y))

  def get_l_angle(self, frame):
    pass

  def get_r_angle(self, frame):
    pass

  def get_goalie_c_angle(self, frame):
    pass

  def get_goalie_l_angle(self, frame):
    pass

  def get_goalie_r_angle(self, frame):
    pass