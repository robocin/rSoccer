import math
import random
from typing import Dict

import gymnasium as gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv


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
            +2*i     id i Yellow Robot [X, Y]
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
            Course completed, 2 minutes (4800 steps), robot exits course limits or robot 
            reverse a checkpoint
    """

    def __init__(self, render_mode=None):
        super().__init__(field_type=2, n_robots_blue=1, 
                         n_robots_yellow=4, time_step=0.025, render_mode=render_mode)
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(4, ), dtype=np.float32)
        
        n_obs = 5 + 8*self.n_robots_blue + 2*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)
        
        self.checkpoints_count = 0
        # Checkpoints nodes positions
        self.node_0 = -0.5 
        self.node_1 = -1.
        self.node_2 = -1.5
        self.node_3 = -2.
        self.field_margin = 1
        
        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        
        print('Environment initialized')

    def reset(self, *, seed=None, options=None):
        self.checkpoints_count = 0
        return super().reset(seed=seed, options=options)

    def _frame_to_observations(self):

        observation = []

        observation.append(((self.checkpoints_count/6)*2)-1)
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
            observation.append(1 if self.frame.robots_blue[i].infrared else -1)

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
        cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta,
                    dribbler=True if actions[3] > 0 else False)
        commands.append(cmd)

        return commands

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        reward = 0
        done = False
        
        ball = self.frame.ball
        last_ball = None or self.last_frame.ball
        robot = self.frame.robots_blue[0]
        
        # End episode in case of collision
        for rbt in self.frame.robots_yellow.values():
            if abs(rbt.v_x) > 0.05 or abs(rbt.v_y) > 0.05:
                done = True
        
        def robot_out_of_bounds(rbt):
            if rbt.x < self.node_3 - self.field_margin or rbt.x > self.field_margin:
                return True
            if abs(rbt.y) > self.field_margin:
                return True
            return False
        
        if robot_out_of_bounds(robot):
            done = True
        elif last_ball:
            if self.checkpoints_count == 0:
                if ball.x < self.node_0 and ball.x > self.node_1:
                    if last_ball.y >= 0 and ball.y < 0:
                        reward = 1
                        self.checkpoints_count += 1
            elif self.checkpoints_count == 1:
                if ball.x < self.node_1 and ball.x > self.node_2:
                    if last_ball.y < 0 and ball.y >= 0:
                        reward = 1
                        self.checkpoints_count += 1
            elif self.checkpoints_count >= 2:
                if self.checkpoints_count % 2 == 0:
                    if ball.x < self.node_2 and ball.x > self.node_3:
                        if last_ball.y >= 0 and ball.y < 0:
                            reward = 1
                            self.checkpoints_count += 1
                            if self.checkpoints_count == 7:
                                done = True
                        elif last_ball.y < 0 and ball.y >= 0:
                            done = True
                else:
                    if ball.x > self.node_3 - self.field_margin and ball.x < self.node_3:
                        if last_ball.y < 0 and ball.y >= 0:
                            reward = 1
                            self.checkpoints_count += 1

        done = done

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        # TODO

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=-0.1, y=0.)

        pos_frame.robots_blue[0] = Robot(x=0., y=0., theta=180.)
        
        pos_frame.robots_yellow[0] = Robot(x=self.node_0, y=0., theta=180.)
        pos_frame.robots_yellow[1] = Robot(x=self.node_1, y=0., theta=180.)
        pos_frame.robots_yellow[2] = Robot(x=self.node_2, y=0., theta=180.)
        pos_frame.robots_yellow[3] = Robot(x=self.node_3, y=0., theta=180.)

        return pos_frame
