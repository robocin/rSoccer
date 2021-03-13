import math
import random
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot, Ball
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLHWDribblingEnv(SSLBaseEnv):
    """The SSL robot needs navigate a course while keeping the ball


        Description:
            The robot must navigate through a field with robots as obstacles,
            while keeping the ball. The obstacles are placed on a straight line,
            the course will be in a zigzag configuration, having checkpoints
            defined as the space between two robots, the last checkpoint needs
            to be passed three times
        Observation:
            Type: Box(4 + 7*n_robots_blue + 2*n_robots_yellow)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->10    id 0 Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta]
            +5*i     id i Yellow Robot [X, Y]
        Actions:
            Type: Box(4,)
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
            3       id 0 Blue Dribbler  (%) (true if % is positive)
            
        Reward:
            1 every time the robot passes a checkpoint
        Starting State:
            Robot starts with the ball and obstacles are spaced with 
            pre defined values.
        Episode Termination:
            Course completed, 2 minutes, robot exits course limits or robot 
            reverse a checkpoint
    """

    def __init__(self, field_type=2):
        super().__init__(field_type=field_type, n_robots_blue=1, 
                         n_robots_yellow=0, time_step=0.032)
        self.random_init = random_init
        self.enter_goal_area= enter_goal_area
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(5, ), dtype=np.float32)
        
        n_obs = 4 + 7*self.n_robots_blue + 5*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)
        
        self.checkpoints_count = 0
        # Checkpoint dimensions
        # Checkpoints dimensions
        gate_0 = 0.5    # Distance between initial position and first obstacle
        gate_1 = 0.25
        gate_2 = 0.3
        gate_3 = 0.4
        
        
        print('Environment initialized')

    def reset(self):
        self.checkpoints_count = 0
        return super().reset()

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
        
        # Checkpoints dimensions
        gate_0 = 0.5    # Distance between initial position and first obstacle
        gate_1 = 0.25
        gate_2 = 0.3
        gate_3 = 0.4
        gates_sum = gate_0 + gate_1 + gate_2 + gate_3
        field_margin = 1
        
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        def robot_out_of_bounds(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid
        
        if robot_out_of_bounds(robot):
            done = True
        # TODO
        # else self.last_frame is not None:
        #     if self.checkpoints_count == 0:
        #     elif self.checkpoints_count == 1:
        #     elif self.checkpoints_count == 2:
        #     elif self.checkpoints_count == 3:

        done = done or self.steps * self.time_step >= 120

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        # TODO

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=0., y=0.)

        pos_frame.robots_blue[0] = Robot(x=0., y=0., theta=0.)

        return pos_frame
