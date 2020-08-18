from dataclasses import dataclass
from gym_ssl.util import *


CENTER_GOAL_X = -6000
CENTER_GOALY = 0

LEFT_GOAL_X = -6000
LEFT_GOALY = -600

RIGHT_GOAL_X = -6000
RIGHT_GOALY = -600

ROBOT_RADIUS = 90

@dataclass
class GoalieState:
  """Init Frame object."""
  ball_x: float = None
  ballY: float = None
  ball_vx: float = None
  ball_vy: float = None
  robot_w: float = None
  distance: float = None
  theta_l_sen: float = None
  theta_l_cos: float = None
  theta_r_sen: float = None
  theta_r_cos: float = None
  theta_goalie_c_sen: float = None
  theta_goalie_c_cos: float = None
  theta_goalie_l_sen: float = None
  theta_goalie_l_cos: float = None
  theta_goalie_r_sen: float = None
  theta_goalie_r_cos: float = None

  def getDistance(self, frame) -> float:
    return float(mod(abs(frame.robotsBlue[0].x-CENTER_GOAL_X), abs(frame.robotsBlue[0].y-CENTER_GOALY)))

  def get_l_angle(self, frame):
    dist_left = [frame.robotsBlue[0].x - LEFT_GOAL_X, frame.robotsBlue[0].y - LEFT_GOALY]
    angle_left = angle(dist_left[0], dist_left[1]) + (math.pi - frame.robotsBlue[0].w)
    return math.sin(angle_left), math.cos(angle_left)

  def get_r_angle(self, frame):
    dist_right = [frame.robotsBlue[0].x - RIGHT_GOAL_X, frame.robotsBlue[0].y - RIGHT_GOALY]
    angle_right = angle(dist_right[0], dist_right[1]) - (math.pi - frame.robotsBlue[0].w)
    return math.sin(angle_right), math.cos(angle_right)

  def get_goalie_c_unify_angle(self, frame):
    angle_c = angle(frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - frame.robotsYellow[0].y)
    return angle_c
  
  def get_goalie_c_angle(self, frame):
    angle_c = angle(frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - frame.robotsYellow[0].y)
    return math.sin(angle_c), math.cos(angle_c)

  def get_goalie_l_angle(self, frame):
    dist_left = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - (frame.robotsYellow[0].x - ROBOT_RADIUS)]
    angle_left = angle(dist_left[0], dist_left[1]) - (math.pi - self.get_goalie_c_unify_angle(frame))
    return math.sin(angle_left), math.cos(angle_left)

  def get_goalie_r_angle(self, frame):
    dist_right = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - (frame.robotsYellow[0].x + ROBOT_RADIUS)]
    angle_right = angle(dist_right[0], dist_right[1]) - (math.pi - self.get_goalie_c_unify_angle(frame))
    return math.sin(angle_right), math.cos(angle_right)
  
  def getObservation(self, frame):
    self.ball_x = frame.ball.x
    self.ballY = frame.ball.y
    self.ball_vx = frame.ball.vx
    self.ball_vy = frame.ball.vy
    
    self.distance = self.getDistance(frame)
    self.robot_w = frame.robotsBlue[0].vw
    self.theta_l_sen, self.theta_l_cos = self.get_l_angle(frame)
    self.theta_r_sen, self.theta_r_cos = self.get_r_angle(frame)
    self.theta_goalie_c_sen, self.theta_goalie_c_cos = self.get_goalie_c_angle(frame)
    self.theta_goalie_l_sen, self.theta_l_cos = self.get_goalie_l_angle(frame)
    self.theta_goalie_r_sen, self.theta_r_cos = self.get_goalie_r_angle(frame)
    
    observation = []

    observation.append(self.ball_x) 
    observation.append(self.ballY) 
    observation.append(self.ball_vx) 
    observation.append(self.ball_vy) 
    observation.append(self.robot_w)
    observation.append(self.distance)
    observation.append(self.theta_l_sen)
    observation.append(self.theta_l_cos)
    observation.append(self.theta_r_sen) 
    observation.append(self.theta_r_cos)
    observation.append(self.theta_goalie_c_sen)
    observation.append(self.theta_goalie_c_cos)
    observation.append(self.theta_goalie_l_sen)
    observation.append(self.theta_l_cos)
    observation.append(self.theta_goalie_r_sen)
    observation.append(self.theta_r_cos)
    return observation
