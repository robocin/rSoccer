from dataclasses import dataclass
from rc_gym.Utils import *


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
    #dist_left = [abs(frame.ball.x - frame.robotsBlue[0].x), abs(frame.ball.y - frame.robotsBlue[0].y)]
    #angle_ball = angle(dist_left[0], dist_left[1])
    # print(angle_ball)
    # if frame.robotsBlue[0].theta * frame.ball.y >=0:
    #   # the direction of the robot and the position of the ball are both in the same side of the fild (top or bottom)
    #   angle_relative = 
    # print(frame.robotsBlue[0].theta)
    #sign = lambda x: (1, -1)[x<0]
    #if frame.ball.x <= frame.robotsBlue[0].x:
    #
    #  if sign(frame.ball.y) ^ sign(frame.robotsBlue[0].y) >= 0:
    #    #if both values have the same signal
    #    angle_relative = abs(angle_ball - (math.pi - abs(frame.robotsBlue[0].theta)))
    #  else:
    #    angle_relative = abs(angle_ball + (math.pi - abs(frame.robotsBlue[0].theta)))
    #else:
    #  if sign(frame.ball.y) ^ sign(frame.robotsBlue[0].y) >= 0: 
    #    angle_relative = abs(abs(frame.robotsBlue[0].theta) - angle_ball)
    #  else:
    #    angle_relative = abs(abs(frame.robotsBlue[0].theta) + angle_ball)
    #
    #print(angle_relative)
    #
    robot_ball = [frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y]
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
    #print(angle_to_ball)
    return angle_to_ball

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
    self.ball_x, self.ball_y = self.getBallLocalCoordinates(frame)
    self.ball_vx, self.ball_vy = self.getBallLocalSpeed(frame)
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