import gym_ssl.grsim_ssl.Communication.pb.messages_robocup_ssl_wrapper_pb2 as wrapper_pb2
import gym_ssl.grsim_ssl.Communication.pb.grSim_Packet_pb2 as packet_pb2


class Frame:

  def __init__(self):
    """Init Frame object."""
    self.ball = Ball()
    self.robots_blue = [Robot(i) for i in range(1)]
    self.robots_yellow = [Robot(i) for i in range(1)]
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
      idx = _robot.robot_id
      self.robots_blue[idx].id = _robot.robot_id
      self.robots_blue[idx].x = _robot.x
      self.robots_blue[idx].y = _robot.y
      self.robots_blue[idx].orientation = _robot.orientation
      self.robots_blue[idx].vx = _robot.vx
      self.robots_blue[idx].vy = _robot.vy
      self.robots_blue[idx].vorientation = _robot.vorientation

    for _robot in packet.detection.robots_yellow:
      idx = _robot.robot_id
      self.robots_yellow[idx].id = _robot.robot_id
      self.robots_yellow[idx].x = _robot.x
      self.robots_yellow[idx].y = _robot.y
      self.robots_yellow[idx].orientation = _robot.orientation
      self.robots_yellow[idx].vx = _robot.vx
      self.robots_yellow[idx].vy = _robot.vy
      self.robots_yellow[idx].vorientation = _robot.vorientation


class Ball:
  """Init Ball object."""
  def __init__(self):
    self.x = None
    self.y = None
    self.vx = None
    self.vy = None

class Robot:
  """Init Robot object."""
  def __init__(self,id):
    self.id = id
    self.x = None
    self.y = None
    self.orientation = None
    self.vx = None
    self.vy = None
    self.vorientation = None


