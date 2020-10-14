'''
#   Environment Communication Structure
#    - Father class that creates the structure to communicate with multples setups of enviroment
#    - To create your wrapper from env to communcation, use inherit from this class! 
'''


import gym
import robosim
import numpy as np
from rc_gym.Entities import Frame
from rc_gym.Utils.Render import RCRender
from typing import List, Dict


class rSimVSSEnv(gym.Env):
    def __init__(self, field_type: int,
                 n_robots_blue: int, n_robots_yellow: int):
        self.simulator = robosim.SimulatorVSS(field_type=field_type,
                                              n_robots_blue=n_robots_blue,
                                              n_robots_yellow=n_robots_yellow)
        self.field_type: int = field_type
        self.n_robots_blue: int = n_robots_blue
        self.n_robots_yellow: int = n_robots_yellow
        self.field_params: Dict[str,
                                np.float64] = self.simulator.get_field_params()
        self.frame: Frame = None
        self.view = None
        self.steps = 0

    def step(self, action):
        self.steps += 1

        # TODO talvez substituir o get commands por wrappers
        # Join agent action with environment actions
        commands: List[Robot] = self._get_commands(action)
        # Convert commands to simulator commands format
        sim_commands: np.ndarray = self.commands_to_sim_commands(commands)
        # step simulation
        self.simulator.step(sim_commands)

        # Get status and state from simulator
        state = self.simulator.get_state()
        status = self.simulator.get_status()
        # Update frame with new status and state
        self.frame = Frame()
        self.frame.parse(state, status, self.n_robots_blue,
                         self.n_robots_yellow)

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        return observation, reward, done, {}

    def reset(self):
        self.steps = 0

        # Reset simulator
        self.simulator.reset()

        # Sets robots positions
        initial_pos_frame: Frame = self._get_initial_positions_frame()
        replacement_pos = self.frame_to_replacement(initial_pos_frame)
        self.simulator.replace(**replacement_pos)
        
        # Get status and state from simulator
        state = self.simulator.get_state()
        status = self.simulator.get_status()
        # Update frame with new status and state
        self.frame = Frame()
        self.frame.parse(state, status, self.n_robots_blue,
                         self.n_robots_yellow)

        return self._frame_to_observations()

    def render(self) -> None:
        '''
        Renders the game depending on 
        ball's and players' positions.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        if self.view == None:
            self.view = RCRender(
                self.n_robots_blue, self.n_robots_yellow, self.field_params)

        self.view.render_frame(self.frame)

    def frame_to_replacement(self, replacement_frame: Frame) -> Dict[str, np.ndarray]:
        '''
        Returns a dict with robot position array to be used 
        in a replacement command

        Parameters
        ----------
        Frame
            replacement_frame

        Returns
        -------
        Dict
            replacement_pos

        '''
        replacement_pos: Dict[str, np.ndarray] = {}

        ball_pos: List[float] = [
            replacement_frame.ball.x, replacement_frame.ball.y]
        
        replacement_pos['ball_pos'] = np.array(ball_pos)

        blue_pos: List[List[float]] = []
        for robot in replacement_frame.robots_blue.values():
            robot_pos: List[float] = [robot.x, robot.y]
            blue_pos.append(robot_pos)

        replacement_pos['blue_pos'] = np.array(blue_pos)

        yellow_pos: List[List[float]] = []
        for robot in replacement_frame.robots_yellow.values():
            robot_pos: List[float] = [robot.x, robot.y]
            yellow_pos.append(robot_pos)

        replacement_pos['yellow_pos'] = np.array(yellow_pos)

        return replacement_pos

    def commands_to_sim_commands(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            sim_commands[rbt_id][0] = cmd.v_wheel1 * 100
            sim_commands[rbt_id][1] = cmd.v_wheel2 * 100

        return sim_commands

    def _get_commands(self, action):
        '''returns a list of commands of type List[Robot] from type action_space action'''
        raise NotImplementedError

    def _frame_to_observations(self):
        '''returns a type observation_space observation from a type List[Robot] state'''
        raise NotImplementedError

    def _calculate_reward_and_done(self):
        '''returns reward value and done flag from type List[Robot] state'''
        raise NotImplementedError

    def _get_initial_positions_frame(self) -> Frame:
        '''returns frame with robots initial positions'''
        raise NotImplementedError