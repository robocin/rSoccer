import math
import random
from rc_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLGoToBallIREnv(SSLBaseEnv):
    """The SSL robot needs to reach the ball 


        Description:
            One blue robot and a ball are randomly placed on a div B field,
            the episode ends when the robots infrared is activated, the ir
            is activated when the ball touches the robot kicker
        Observation:
            Type: Box(4 + 7*n_robots_blue + 5*n_robots_yellow)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->10    id 0 Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta]
            +5*i     id i Yellow Robot [X, Y, v_x, v_y, v_theta]
        Actions:
            Type: Box(3, )
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
        Reward:
            1 if ball is reached
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            Ball is reached or 30 seconds (1200 steps)
    """

    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(field_type=field_type, n_robots_blue=1, 
                         n_robots_yellow=n_robots_yellow, time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ), dtype=np.float32)
        
        n_obs = 4 + 7*self.n_robots_blue + 5*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        print('Environment initialized')

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        commands.append(Robot(yellow=False, id=0, v_x=actions[0],
                              v_y=actions[1], v_theta=actions[2]))

        return commands

    def _calculate_reward_and_done(self):
        reward = 0

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        # Check if robot infrared is activated
        if robot.infrared:
            reward = 1

        done = reward

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
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

        agents = []
        for i in range(self.n_robots_blue):
            pos_frame.robots_blue[i] = Robot(x=x(), y=y(), theta=theta())
            agents.append(pos_frame.robots_blue[i])

        for i in range(self.n_robots_yellow):
            pos_frame.robots_yellow[i] = Robot(x=x(), y=y(), theta=theta())
            agents.append(pos_frame.robots_yellow[i])

        def same_position_ref(obj, ref, radius):
            if obj.x >= ref.x - radius and obj.x <= ref.x + radius and \
                    obj.y >= ref.y - radius and obj.y <= ref.y + radius:
                return True
            return False

        radius_ball = 0.03
        radius_robot = 0.1

        for i in range(len(agents)):
            while same_position_ref(agents[i], pos_frame.ball, radius_ball):
                agents[i] = Robot(x=x(), y=y(), theta=theta())
            for j in range(i):
                while same_position_ref(agents[i], agents[j], radius_robot):
                    agents[i] = Robot(x=x(), y=y(), theta=theta())

        for i in range(self.n_robots_blue):
            pos_frame.robots_blue[i] = agents[i]

        for i in range(self.n_robots_yellow):
            pos_frame.robots_yellow[i] = agents[i+self.n_robots_blue]

        return pos_frame
