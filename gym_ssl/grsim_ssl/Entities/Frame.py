from gym_ssl.grsim_ssl.Entities.Ball import Ball
from gym_ssl.grsim_ssl.Entities.Robot import Robot
import gym_ssl.grsim_ssl.Communication.pb.messages_robocup_ssl_wrapper_pb2 as wrapper_pb2
import gym_ssl.grsim_ssl.Communication.pb.grSim_Packet_pb2 as packet_pb2



class Frame:
  def __init__(self):
    """Init Frame object."""
    self.ball = Ball()
    self.robotsBlue = {}
    self.robotsYellow = {}
    self.timestamp = None

  def parse(self, packet):
    '''It parses the state received from grSim in a common state for environment'''
    self.timestamp = packet.detection.t_capture

    for _ball in packet.detection.balls:
      self.ball.x = _ball.x
      self.ball.y = _ball.y 
      self.ball.vx = _ball.vx
      self.ball.vy = _ball.vy
    
    for _robot in packet.detection.robots_blue:
      robot = Robot()
      robot.id = _robot.robot_id
      robot.x = _robot.x
      robot.y = _robot.y
      robot.w = _robot.orientation
      robot.vx = _robot.vx
      robot.vy = _robot.vy
      robot.vw = _robot.vorientation

      self.robotsBlue[robot.id] = robot

    for _robot in packet.detection.robots_yellow:
      robot = Robot()
      robot.id = _robot.robot_id
      robot.x = _robot.x
      robot.y = _robot.y
      robot.w = _robot.orientation
      robot.vx = _robot.vx
      robot.vy = _robot.vy
      robot.vw = _robot.vorientation

      self.robotsYellow[robot.id] = robot

