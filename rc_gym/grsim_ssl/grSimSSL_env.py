'''
#   Environment Communication Structure
#    - Father class that creates the structure to communicate with multples setups of enviroment
#    - To create your wrapper from env to communcation, use inherit from this class! 
'''


import gym
from rc_gym.grsim_ssl.Communication.grSimClient import grSimClient
from rc_gym.grsim_ssl.Entities import Robot

class GrSimSSLEnv(gym.Env):
    def __init__(self):
        self.client = grSimClient()
        self.action_space = None
        self.observation_space = None
        self.state = None
        self.steps = 0

    def step(self, action):
        self.steps += 1
        # Sends actions
        commands = self._getCommands(action)
        self.client.sendCommandsPacket(commands) 

        # Update state
        self.state = self.client.receiveState()

        # Calculate environment observation, reward and done condition
        observation = self._parseObservationFromState()
        reward, done = self._calculateRewardsAndDoneFlag()

        return observation, reward, done, {}

    def reset(self):
        self.steps = 0
        # Place robots on reset positions
        resetRobotPositions, resetBallPosition = self._getFormation()
        self.client.sendReplacementPacket(robotPositions=resetRobotPositions, ballPosition=resetBallPosition) 
        # Update state and observation
        self.state = self.client.receiveState()
        observation = self._parseObservationFromState()

        return observation

    def _getCommands(self, action):
        '''returns a list of commands of type List[Robot] from type action_space action'''
        raise NotImplementedError

    def _getFormation(self):
        '''returns a positioning formation of type List[Robot]'''
        raise NotImplementedError

    def _parseObservationFromState(self):
        '''returns a type observation_space observation from a type List[Robot] state'''
        raise NotImplementedError

    def _calculateRewardsAndDoneFlag(self):
        '''returns reward value and done flag from type List[Robot] state'''
        raise NotImplementedError

