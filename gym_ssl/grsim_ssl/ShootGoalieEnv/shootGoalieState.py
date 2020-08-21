from dataclasses import dataclass
from gym_ssl.grsim_ssl.Utils import *


CENTER_GOAL_X = -6000
CENTER_GOALY = 0

LEFT_GOAL_X = -6000
LEFT_GOALY = -600

RIGHT_GOAL_X = -6000
RIGHT_GOALY = 600

ROBOT_RADIUS = 90

@dataclass
class shootGoalieState:
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

  def getLeftPoleAngle(self, frame):
    dist_left = [frame.robotsBlue[0].x - LEFT_GOAL_X, frame.robotsBlue[0].y - LEFT_GOALY]
    angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robotsBlue[0].theta))
    return math.sin(angle_left), math.cos(angle_left)

  def getRightPoleAngle(self, frame):
    dist_right = [frame.robotsBlue[0].x - RIGHT_GOAL_X, frame.robotsBlue[0].y - RIGHT_GOALY]
    angle_right = toPiRange(angle(dist_right[0], dist_right[1]) + (math.pi - frame.robotsBlue[0].theta))
    return math.sin(angle_right), math.cos(angle_right)

  def getGoalieCenterUnifiedAngle(self, frame):
    dist_g = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - frame.robotsYellow[0].y]
    angle_g = toPiRange(angle(dist_g[0], dist_g[1]) + (math.pi - frame.robotsBlue[0].theta))
    return angle_g
  
  def getGoalieCenterAngle(self, frame):
    angle_c = self.getGoalieCenterUnifiedAngle(frame)
    return math.sin(angle_c), math.cos(angle_c)

  def getGoalieLeftAngle(self, frame):
    dist_left = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - (frame.robotsYellow[0].y - ROBOT_RADIUS)]
    angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robotsBlue[0].theta))
    return math.sin(angle_left), math.cos(angle_left)

  def getGoalieRightAngle(self, frame):
    dist_right = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - (frame.robotsYellow[0].y + ROBOT_RADIUS)]
    angle_right = toPiRange(angle(dist_right[0], dist_right[1]) + (math.pi - frame.robotsBlue[0].theta))
    return math.sin(angle_right), math.cos(angle_right)
  
  def getBallLocalCoordinates(self, frame):
    robot_ball = [frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
    robot_ball_x = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_y = mod_to_ball* math.sin(angle_to_ball)
    return robot_ball_x, robot_ball_y
  
  def getBallLocalSpeed(self, frame):
    robot_ball = [frame.robotsBlue[0].vx - frame.ball.vx, frame.robotsBlue[0].vy - frame.ball.vy]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
    robot_ball_vx = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_vy = mod_to_ball* math.sin(angle_to_ball)
    return robot_ball_vx, robot_ball_vy

  
  def getObservation(self, frame):

    self.ball_x, self.ballY = self.getBallLocalCoordinates(frame)
    self.ball_vx, self.ball_vy = self.getBallLocalSpeed(frame)
    
    self.distance = self.getDistance(frame)
    self.robot_w = frame.robotsBlue[0].vw
    self.theta_l_sen, self.theta_l_cos = self.getLeftPoleAngle(frame)
    self.theta_r_sen, self.theta_r_cos = self.getRightPoleAngle(frame)
    self.theta_goalie_c_sen, self.theta_goalie_c_cos = self.getGoalieCenterAngle(frame)
    self.theta_goalie_l_sen, self.theta_l_cos = self.getGoalieLeftAngle(frame)
    self.theta_goalie_r_sen, self.theta_r_cos = self.getGoalieRightAngle(frame)
    
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