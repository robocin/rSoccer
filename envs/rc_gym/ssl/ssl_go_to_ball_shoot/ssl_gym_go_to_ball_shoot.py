import math
import random
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot, Ball
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLGoToBallShootEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal


        Description:
            One blue robot and a ball are placed on fixed position on a half 
            div B field, the robot is rewarded if it makes a goal
        Observation:
            Type: Box(4 + 7*n_robots_blue + 5*n_robots_yellow)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->10    id 0 Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta]
            +5*i     id i Yellow Robot [X, Y, v_x, v_y, v_theta]
        Actions:
            Type: Box(5, )
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
            3       id 0 Blue Kick x Speed  (%)
            4       id 0 Blue Dribbler  (%) (true if % is positive)
            
        Reward:
            1 if goal
        Starting State:
            Robot and ball on half opponent field size in different y.
        Episode Termination:
            Goal, ball leaves bounds or 60 seconds (2400 steps)
    """

    def __init__(self, field_type=1, random_init=False, enter_goal_area=False):
        super().__init__(field_type=field_type, n_robots_blue=1, 
                         n_robots_yellow=0, time_step=0.025)
        self.random_init = random_init
        self.enter_goal_area= enter_goal_area
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(5, ), dtype=np.float32)
        
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
                              v_y=actions[1], v_theta=actions[2],
                              kick_v_x=1. if actions[3] > 0 else 0., 
                              dribbler=True if actions[4] > 0 else False))

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        done = False
        
        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        half_goal_wid = self.field.goal_width / 2
        
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid
        
        # Check if robot exited field right side limits
        if robot.x < 0 or abs(robot.y) > half_wid:
            done = True
        # If flag is set, end episode if robot enter gk area
        elif not self.enter_goal_area and robot_in_gk_area(robot):
            done = True
        # Check ball for ending conditions
        elif ball.x < 0 or abs(ball.y) > half_wid:
            done = True
        elif ball.x > half_len:
            done = True
            reward = 1 if abs(ball.y) < half_goal_wid else 0

        done = done

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        if self.random_init:
            half_len = self.field.length / 2
            half_wid = self.field.width / 2
            penalty_len = self.field.penalty_length
            def x(): return random.uniform(0.3, half_len - penalty_len - 0.3)
            def y(): return random.uniform(-half_wid + 0.1, half_wid - 0.1)
            def theta(): return random.uniform(-180, 180)
        else:
            def x(): return self.field.length / 4
            def y(): return self.field.width / 8
            def theta(): return 0

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        pos_frame.robots_blue[0] = Robot(x=x(), y=-y(), theta=theta())

        return pos_frame
