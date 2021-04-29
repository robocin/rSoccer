import math
import random
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLPassEnduranceMAEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal with contested possession


        Description:

        Observation:
            Type: Box(4 + 6*n_robots_blue)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->13    id N Blue [X, Y, sin(theta), cos(theta), V_theta]


        Actions:
            Type: Box(3,)
            Num     Action
            0       id i Blue Angular Speed  (%)
            1       id i Blue Kick x Speed  (%)
            2       id i Blue Dribbler  (%) (true if % is positive)

        Reward:

        Starting State:

        Episode Termination:
            30 seconds (1200 steps) or wrong pass
    """

    def __init__(self):
        super().__init__(field_type=2, n_robots_blue=2,
                         n_robots_yellow=0, time_step=0.025)
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(self.n_robots_blue, 5),
                                           dtype=np.float32)

        n_obs = 4 + 9*self.n_robots_blue
        self.stopped_steps = 0
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(
                                                    self.n_robots_blue, n_obs),
                                                dtype=np.float32)
        self.receiver_id = 1
        self.shooter_id = 0
        self.ball_grad_scale = np.linalg.norm([self.field.width/2,
                                               self.field.length/2])/4
        wheel_max_rad_s = 160
        max_steps = 1200
        self.energy_scale = ((wheel_max_rad_s * 4) * max_steps)

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        self.max_kick_x = 5.0

        print('Environment initialized')

    def get_rotated_obs(self):
        robots_dict = dict()
        for i in range(self.n_robots_blue):
            robots_dict[i] = list()
            robots_dict[i].append(self.norm_pos(self.frame.robots_blue[i].x))
            robots_dict[i].append(self.norm_pos(self.frame.robots_blue[i].y))
            robots_dict[i].append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots_dict[i].append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots_dict[i].append(self.norm_v(self.frame.robots_blue[i].v_x))
            robots_dict[i].append(self.norm_v(self.frame.robots_blue[i].v_y))
            robots_dict[i].append(self.norm_w(
                self.frame.robots_blue[i].v_theta))
            robots_dict[i].append(
                1 if self.frame.robots_blue[i].infrared else 0)
            robots_dict[i].append(1 if i == self.shooter_id else 0)

        rotaded_obs = list()
        for i in range(self.n_robots_blue):
            aux_dict = {}
            aux_dict.update(robots_dict)
            rotated = list()
            rotated = rotated + aux_dict.pop(i)
            teammates = list(aux_dict.values())
            for teammate in teammates:
                rotated = rotated + teammate
            rotaded_obs.append(rotated)

        return rotaded_obs

    def _frame_to_observations(self):

        observations = list()
        robots = self.get_rotated_obs()
        for idx in range(self.n_robots_blue):
            observation = []
            observation.append(self.norm_pos(self.frame.ball.x))
            observation.append(self.norm_pos(self.frame.ball.y))
            observation.append(self.norm_v(self.frame.ball.v_x))
            observation.append(self.norm_v(self.frame.ball.v_y))
            observation += robots[idx]
            observations.append(np.array(observation, dtype=np.float32))

        observations = np.array(observations)
        return observations

    def reset(self):
        self.reward_shaping_total = None
        state = super().reset()
        self.stopped_steps = 0
        self.shooter_id, self.receiver_id = 0, 1
        return state

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _get_commands(self, actions):
        commands = []
        actions[0][3] = actions[0][3] if abs(actions[0][3]) > 0.5 else 0
        actions[1][3] = actions[1][3] if abs(actions[1][3]) > 0.5 else 0


        for i in range(self.n_robots_blue):
            angle = self.frame.robots_blue[i].theta
            v_x, v_y, v_theta = self.convert_actions(actions[i], np.deg2rad(angle))
            cmd = Robot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta,
                        kick_v_x=actions[i][3] * self.max_kick_x,
                        dribbler=True if actions[i][4] > 0 else False)
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
        w_ball_grad = 1/self.ball_grad_scale
        w_energy = 1/self.energy_scale
        reward = {f'robot_{i}': 0 for i in range(self.n_robots_blue)}
        done = False
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'n_passes': 0,
                                         'ball_grad': 0}
            for i in range(self.n_robots_blue):
                self.reward_shaping_total[f'robot_{i}'] = {'energy': 0}

        if self.frame.robots_blue[self.receiver_id].infrared:
            for i in range(self.n_robots_blue):
                reward[f'robot_{i}'] = 10
            self.reward_shaping_total['n_passes'] += 1
            self.stopped_steps = 0
            self.shooter_id, self.receiver_id = self.receiver_id, self.shooter_id
        else:
            rw_ball_grad = w_ball_grad * self.__ball_grad_rw()
            reward[f'robot_{self.shooter_id}'] += rw_ball_grad
            reward[f'robot_{self.receiver_id}'] += rw_ball_grad
            self.reward_shaping_total['ball_grad'] += rw_ball_grad

            for i in range(self.n_robots_blue):
                rw_energy = w_energy*self.__energy_pen(i)
                reward[f'robot_{i}'] += rw_energy
                self.reward_shaping_total[f'robot_{i}']['energy'] += rw_energy

        if self.__bad_state():
            for i in range(self.n_robots_blue):
                reward[f'robot_{i}'] = -1
            done = True
        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        pos_frame: Frame = Frame()
        def x(): return random.uniform(-2, 2)
        def y(): return random.uniform(1.5, -1.5)

        pos_frame.ball = Ball(x=x(), y=y())
        factor = (pos_frame.ball.y/abs(pos_frame.ball.y))
        offset = 0.09*factor
        angle = 270 if factor > 0 else 90
        pos_frame.robots_blue[0] = Robot(
            x=pos_frame.ball.x, y=pos_frame.ball.y+offset, theta=angle
        )
        ball = np.array([pos_frame.ball.x,
                         pos_frame.ball.y])
        recv_x = x()
        while abs(recv_x - pos_frame.ball.x) < 1.5:
            recv_x = x()
        receiver = np.array([recv_x, -pos_frame.ball.y])
        vect = receiver - ball
        recv_angle = np.rad2deg(np.arctan2(vect[1], vect[0]) + np.pi)

        pos_frame.robots_blue[1] = Robot(x=receiver[0],
                                         y=receiver[1],
                                         theta=recv_angle)

        return pos_frame

    def __bad_state(self):
        # Check if dist between robots > 1.5
        recv = np.array([self.frame.robots_blue[self.receiver_id].x,
                         self.frame.robots_blue[self.receiver_id].y])
        shooter = np.array([self.frame.robots_blue[self.shooter_id].x,
                            self.frame.robots_blue[self.shooter_id].y])
        min_dist = np.linalg.norm(recv - shooter) > 1.5

        # Check if ball is in this rectangle
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])
        inside = -2 < ball[0] < 2 and -1.5 < ball[1] < 1.5

        # Check if ball is stopped for too long
        last_dist = np.linalg.norm(last_ball - recv)
        dist = np.linalg.norm(ball - recv)
        stopped = abs(last_dist - dist) < 0.01
        if stopped:
            self.stopped_steps += 1
        else:
            self.stopped_steps = 0
        return self.stopped_steps > 20 or not inside or not min_dist

    def __ball_grad_rw(self):
        assert(self.last_frame is not None)

        # Goal pos
        goal = np.array([self.frame.robots_blue[self.receiver_id].x,
                         self.frame.robots_blue[self.receiver_id].y])

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        ball = self.frame.ball
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = np.linalg.norm(goal - last_ball_pos)

        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(goal - ball_pos)

        ball_dist_rw = last_ball_dist - ball_dist
        return np.clip(ball_dist_rw, -1, 1)

    def __energy_pen(self, idx):
        robot = self.frame.robots_blue[idx]

        # Sum of abs each wheel speed sent
        energy = abs(robot.v_wheel0)\
            + abs(robot.v_wheel1)\
            + abs(robot.v_wheel2)\
            + abs(robot.v_wheel3)

        return energy
