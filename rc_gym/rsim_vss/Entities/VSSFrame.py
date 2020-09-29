from rc_gym.rsim_vss.Entities.Ball import Ball
from rc_gym.rsim_vss.Entities.VSSRobot import VSSRobot


class VSSFrame:
  def __init__(self):
    """Init Frame object."""
    self.ball = Ball()
    self.robots_blue = {}
    self.robots_yellow = {}
    self.time = None
    self.goals_yellow = None
    self.goals_blue = None

  def parse(self, state, status, n_robots):
    '''It parses the state received from grSim in a common state for environment'''
    self.time = status['time']
    self.goals_yellow = status['goals_yellow']
    self.goals_blue = status['goals_blue']

    self.ball.x = state[0]
    self.ball.y = state[1]
    self.ball.z = state[2]
    self.ball.vx = state[3]
    self.ball.vy = state[4]
    
    for i in range(n_robots * 2):
      robot = VSSRobot()
      robot.id = i % n_robots
      robot.x = state[5 + (4 * i) + 0]
      robot.y = state[5 + (4 * i) + 1]
      robot.vx = state[5 + (4 * i) + 2]
      robot.vy = state[5 + (4 * i) + 3]
      
      if i < n_robots:
        self.robots_blue[robot.id] = robot
      else:
        self.robots_yellow[robot.id] = robot
