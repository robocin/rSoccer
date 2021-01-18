import math
import random
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot
from rc_gym.Utils import normVx, normX
from rc_gym.vss.vss_gym_base import VSSBaseEnv


class VSSReachEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0 + (4 * i)     id i Blue Robot X
            1 + (4 * i)     id i Blue Robot Y
            2 + (4 * i)     id i Blue Robot Vx
            3  + (4 * i)    id i Blue Robot Vy
            12 + (4 * i)    id i Yellow Robot X
            13 + (4 * i)    id i Yellow Robot Y
            14 + (4 * i)    id i Yellow Robot Vx
            15 + (4 * i)    id i Yellow Robot Vy
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Objective:
            Coord (x,y) of the field
        Reward:
            Sum of Rewards:
                Move to Point
                Energy Penalty
                Reach point
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    __square = np.array([[0.4, 0.3, 180], [-0.4, 0.3, -90],
                         [-0.4, -0.3, 0], [0.4, -0.3, 90]])
    __hexagon = np.array([[0.4, 0.3, 135], [0, 0.45, 225],
                          [-0.4, 0.3, -90], [-0.4, -0.3, -45],
                          [0, -0.45, 45], [0.4, -0.3, 90]])

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=3,
                         n_robots_yellow=3, time_step=0.032)

        low_obs_bound = [-1.2, -1.2, -1.25, -1.25]*6
        high_obs_bound = [1.2, 1.2, 1.25, 1.25]*6
        low_obs_bound = np.array(low_obs_bound)
        high_obs_bound = np.array(high_obs_bound)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low_obs_bound,
                                                high=high_obs_bound,
                                                shape=(24, ), dtype=np.float32)

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.objective = [self.__square, 0]
        self.auto_objective = True
        self.last_obj_time = 0
        self.v_wheel_deadzone = 0.05

        print('Environment initialized')

    def set_objective(self, obj: np.array):
        assert obj.shape == self.objective[0].shape, 'Incorrect Goal shape'
        self.objective = (obj, 0)
        self.auto_objective = False

    def __get_objective(self):
        return [random.choice([self.__square, self.__hexagon]), 0]

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        observation = super().reset()
        self.last_obj_time = 0
        if self.auto_objective:
            self.objective = self.__get_objective()
        return observation, self.objective[0][0]

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        info = {'reward_shaping': self.reward_shaping_total}
        info['objective'] = self.objective[0][self.objective[1]]
        return observation, reward, done, info

    def _frame_to_observations(self):

        observation = []

        for i in range(self.n_robots_blue):
            observation.append(normX(self.frame.robots_blue[i].x))
            observation.append(normX(self.frame.robots_blue[i].y))
            observation.append(normVx(self.frame.robots_blue[i].v_x))
            observation.append(normVx(self.frame.robots_blue[i].v_y))

        for i in range(self.n_robots_yellow):
            observation.append(normX(self.frame.robots_yellow[i].x))
            observation.append(normX(self.frame.robots_yellow[i].y))
            observation.append(normVx(self.frame.robots_yellow[i].v_x))
            observation.append(normVx(self.frame.robots_yellow[i].v_y))

        return np.array(observation)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions
        v_wheel1, v_wheel2 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))

        # Send random commands to the other robots
        for i in range(1, 3):
            actions = self.action_space.sample()
            self.actions[i] = actions
            v_wheel1, v_wheel2 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=False, id=i, v_wheel1=v_wheel1,
                                  v_wheel2=v_wheel2))
        for i in range(3):
            actions = self.action_space.sample()
            v_wheel1, v_wheel2 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=True, id=i, v_wheel1=v_wheel1,
                                  v_wheel2=v_wheel2))

        return commands

    def __move_reward(self):
        '''Calculate Move to point reward

        Cosine between the robot vel vector and the vector robot -> point.
        This indicates rather the robot is moving towards the point or not.
        '''

        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_x,
                              self.frame.robots_blue[0].v_y])
        robot_objective = self.objective[0][self.objective[1]][:-1] - robot
        robot_objective = robot_objective/np.linalg.norm(robot_objective)

        move_reward = np.dot(robot_objective, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -1.0, 1.0)
        return move_reward

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel1)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel2)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        energy_penalty /= self.simulator.robot_wheel_radius
        return energy_penalty

    def __time_penalty(self):
        dt = self.last_obj_time - self.frame.time
        penalty = np.tanh(dt/150)
        return penalty

    def __reached_objective(self):
        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        obj = self.objective[0][self.objective[1]][:-1]
        reached = np.linalg.norm(robot - obj) < 0.01
        return reached

    def _calculate_reward_and_done(self):
        reward = 0
        w_move = 0.2
        w_energy = 2e-4
        w_time = 2e-3
        w_angle = 5/180
        done = False
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'reach_score': 0,
                                         'move': 0,
                                         'energy': 0,
                                         'time': 0}

        # Check if goal ocurred
        if self.__reached_objective():
            desired_angle = self.objective[0][self.objective[1]][-1]
            self.objective[1] = (self.objective[1]
                                 + 1) % len(self.objective[0])

            angle_diff = self.frame.robots_blue[0].theta - desired_angle
            angle_diff = (angle_diff + 180) % 360 - 180

            reward += 10 - angle_diff/w_angle
            self.reward_shaping_total['reach_score'] += 1
            done = self.objective[1] == 0
        else:

            if self.last_frame is not None:
                # Calculate Move ball
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()
                # Calculate Time penalty
                time_penalty = self.__time_penalty()

                reward = w_move * move_reward\
                    + w_time * time_penalty\
                    + w_energy * energy_penalty

                self.reward_shaping_total['move'] += w_move * move_reward
                self.reward_shaping_total['energy'] += w_energy * energy_penalty  # noqa
                self.reward_shaping_total['time'] += w_time * time_penalty

        done = done or self.steps * self.time_step >= 300

        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the inicial frame'''
        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(-180, 180)

        pos_frame: Frame = Frame()

        pos_frame.ball.x = x()
        pos_frame.ball.y = y()
        pos_frame.ball.v_x = 0.
        pos_frame.ball.v_y = 0.

        pos_frame.robots_blue[0] = Robot(x=0, y=0, theta=theta())
        pos_frame.robots_blue[1] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_blue[2] = Robot(x=x(), y=y(), theta=theta())

        pos_frame.robots_yellow[0] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_yellow[1] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_yellow[2] = Robot(x=x(), y=y(), theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.simulator.linear_speed_range
        right_wheel_speed = actions[1] * self.simulator.linear_speed_range

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -2.6, 2.6)

        return left_wheel_speed, right_wheel_speed
