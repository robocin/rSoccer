import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.Kinematics import Kinematics


class SSLGoToBallEnv(SSLBaseEnv):
    """The SSL robot needs to reach the ball 


        Description:
            One blue robot and a ball are randomly placed on a div B field,
            the episode ends when the robots is closer than 0.2m from the ball
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

    def __init__(self, field_type=1, n_robots_blue=1, n_robots_yellow=0, 
                 mm_deviation=0, angle_deviation=0):
        super().__init__(field_type=field_type, n_robots_blue=n_robots_blue, 
                         n_robots_yellow=n_robots_yellow, time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ), dtype=np.float32)
        
        n_obs = 4 + 7*self.n_robots_blue + 2*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)
        
        # Set robot kinematics
        self.kinematics = self._make_robots_kinematics(mm_deviation, angle_deviation)

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        print('Environment initialized')

    def _make_robots_kinematics(self, mm_deviation, angle_deviation):
        self.original_kinematics = self._get_robot_kinematics(robot_id=15,
                                                              mm_deviation=0,
                                                              angle_deviation=0)

        robots_kinematics = []
        for i in range(self.n_robots_blue):
            if i==0:
                kin = self.original_kinematics
            else:
                kin = self._get_robot_kinematics(robot_id=i,
                                                mm_deviation=mm_deviation,
                                                angle_deviation=angle_deviation)
            robots_kinematics.append(kin)
        
        return robots_kinematics

    def _get_robot_kinematics(self, robot_id, mm_deviation=0, angle_deviation=0):
        robot_kinematics = Kinematics.Robot(number_of_wheels=4,
                                            id = robot_id,
                                            wheel_radius = 0.02475,
                                            axis_length = 0.081,
                                            wheels_alphas = [60, 135, -135, -60],
                                            mm_deviation = mm_deviation,
                                            angle_deviation = angle_deviation)
        return robot_kinematics

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
        v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(0))
        for id in range(self.n_robots_blue):
            v_x, v_y, v_theta = self.apply_kinematics_transformation(id, v_x, v_y, v_theta)
            cmd = Robot(yellow=False, id=id, v_x=v_x, v_y=v_y, v_theta=v_theta)
            commands.append(cmd)

        return commands

    def apply_kinematics_transformation(self, id, v_x, v_y, v_theta):
        v = np.array([v_x, v_y, v_theta])

        # WHEELS' DESIRED ROTATION
        phi = self.original_kinematics.get_J2_inv()@self.original_kinematics.get_J1()@v

        # ROBOT MOVEMENT
        v_real = self.kinematics[id].get_J1_inv()@self.kinematics[id].get_J2()@phi

        return v_real

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
    
        for id in range(self.n_robots_blue):
            robot = self.frame.robots_blue[id]
            # Check if robot is out of field
            if abs(robot.x) > 3 or abs(robot.y) > 2:
                reward = 1

        done = reward

        return reward, done
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=3, y=2)

        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = [-2.5, 0.5*i-1.5]
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=0)

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame
