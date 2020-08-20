from dataclasses import dataclass
from gym_ssl.grsim_ssl.Utils import *


TOP_FIELD = 4.5
BOTTOM_FIELD = -4.5
LEFT_FIELD = -6
RIGHT_FIELD = 6

@dataclass
class goToBallState:
  """Init Frame object."""
  ball_x: float = None
  ball_y: float = None
  robot_vx: float = None
  robot_vy: float = None
  robot_w: float = None
  distance: float = None
  wall_top_x: float = None
  wall_top_y: float = None
  wall_bottom_x: float = None
  wall_bottom_y: float = None
  wall_left_x: float = None
  wall_left_y: float = None
  wall_right_x: float = None
  wall_right_y: float = None
  angle_relative: float = None
  

  def getDistance(self, frame) -> float:
    #print(float(mod(abs(frame.robotsBlue[0].x-frame.ball.x), abs(frame.robotsBlue[0].y-frame.ball.y))))
    return float(mod(abs(frame.robotsBlue[0].x-frame.ball.x), abs(frame.robotsBlue[0].y-frame.ball.y)))

  def getTopPosition(self, frame):
    diff_y = TOP_FIELD - frame.robotsBlue[0].y
    pos_x = math.sin(frame.robotsBlue[0].theta) * diff_y
    pos_y = math.cos(frame.robotsBlue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y

  def getBottomPosition(self, frame):
    diff_y = BOTTOM_FIELD - frame.robotsBlue[0].y
    pos_x = math.sin(frame.robotsBlue[0].theta) * diff_y
    pos_y = math.cos(frame.robotsBlue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y

  def getLeftPosition(self, frame):
    diff_y = LEFT_FIELD - frame.robotsBlue[0].x
    pos_x = math.cos(frame.robotsBlue[0].theta) * diff_y
    pos_y = -math.sin(frame.robotsBlue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y 

  def getRightPosition(self, frame):
    diff_y = RIGHT_FIELD - frame.robotsBlue[0].x
    pos_x = math.cos(frame.robotsBlue[0].theta) * diff_y
    pos_y = -math.sin(frame.robotsBlue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y  

  def getRelativeRobotToBallAngle(self, frame):
    #print(toPiRange(angle(frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y)))
    return toPiRange(angle(frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y))
  
  def getObservation(self, frame):
    self.ball_x = frame.ball.x
    self.ball_y = frame.ball.y
    self.robot_vx = frame.robotsBlue[0].vx
    self.robot_vy = frame.robotsBlue[0].vy
    
    self.distance = self.getDistance(frame)
    self.robot_w = frame.robotsBlue[0].vw
    self.wall_top_x, self.wall_top_y = self.getTopPosition(frame)
    self.wall_bottom_x, self.wall_bottom_y = self.getBottomPosition(frame)   
    self.wall_left_x, self.wall_left_y = self.getLeftPosition(frame)
    self.wall_right_x, self.wall_right_y = self.getRightPosition(frame)
    self.angle_relative = self.getRelativeRobotToBallAngle(frame)

    
    
    observation = []

    observation.append(self.ball_x) 
    observation.append(self.ball_y) 
    observation.append(self.robot_vx) 
    observation.append(self.robot_vy) 
    observation.append(self.robot_w)
    observation.append(self.distance)
    observation.append(self.wall_top_x)
    observation.append(self.wall_top_y)
    observation.append(self.wall_bottom_x) 
    observation.append(self.wall_bottom_y)
    observation.append(self.wall_left_x)
    observation.append(self.wall_left_y)
    observation.append(self.wall_right_x)
    observation.append(self.wall_right_y)
    
    return observation