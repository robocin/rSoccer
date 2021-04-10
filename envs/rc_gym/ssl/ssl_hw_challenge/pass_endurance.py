import math
import random
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot, Ball
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLPassEnduranceEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal with contested possession


        Description:

        Observation:
            Type: Box(4 + 5*n_robots_blue)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->N    id 0 Blue [X, Y, sin(theta), cos(theta), receiver]

        Actions:
            Type: Box(2, 3)
            Num     Action
            0       id i Blue Angular Speed  (%)
            1       id i Blue Kick x Speed  (%)
            2       id i Blue Dribbler  (%) (true if % is positive)

        Reward:

        Starting State:

        Episode Termination:
            30 seconds (1200 steps) or wrong pass
    """
    original_vec = None
    max_dist = 100
    actions = {}

    def __init__(self):
        super().__init__(field_type=2, n_robots_blue=2,
                         n_robots_yellow=0, time_step=0.025)
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(self.n_robots_blue, 3),
                                           dtype=np.float32)

        n_obs = 4 + 5*self.n_robots_blue
        self.holding_steps = 0
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(self.n_robots_blue,
                                                       n_obs),
                                                dtype=np.float32)
        self.receiver_id = 1

        print('Environment initialized')

    def _frame_to_observations(self):
        observations = [None]*self.n_robots_blue

        ball = []

        ball.append(self.norm_pos(self.frame.ball.x))
        ball.append(self.norm_pos(self.frame.ball.y))
        ball.append(self.norm_v(self.frame.ball.v_x))
        ball.append(self.norm_v(self.frame.ball.v_y))
        robots = [list() for _ in range(self.n_robots_blue)]
        for i in range(self.n_robots_blue):
            robots[i].append(self.norm_pos(self.frame.robots_blue[i].x))
            robots[i].append(self.norm_pos(self.frame.robots_blue[i].y))
            robots[i].append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots[i].append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robots[i].append(self.receiver_id == i)
        for i in range(self.n_robots_blue):
            observations[i] = ball + robots[i]
            for k in [j for j in range(self.n_robots_blue) if j != i]:
                observations[i] = observations[i] + robots[k]
        return np.array(observations, dtype=np.float32)

    def reset(self):
        self.reward_shaping_total = None
        state = super().reset()
        self.actions = {}
        self.original_vec = self.__get_shooter_receiver_vec()
        self.holding_steps = 0
        return state

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _get_commands(self, actions):
        shooter_id = int(not self.receiver_id)
        commands = []
        self.actions = {shooter_id: actions[0], self.receiver_id: actions[1]}
        cmd = Robot(yellow=False, id=shooter_id, v_x=0,
                    v_y=0, v_theta=actions[0][0],
                    kick_v_x=1 if actions[0][1] > 0 else 0,
                    dribbler=True if actions[0][2] > 0 else False)
        commands.append(cmd)
        cmd = Robot(yellow=False, id=self.receiver_id, v_x=0,
                    v_y=0, v_theta=actions[1][0],
                    kick_v_x=abs(actions[1][1]),
                    dribbler=True if actions[1][2] > 0 else False)
        commands.append(cmd)

        return commands

    def _calculate_reward_and_done(self):

        w_en = 0.1
        w_angle = 0.2
        w_angle_sht = 0.5
        w_dist = 0.2
        reward = [0, 0]
        done = self.__wrong_ball()
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'pass_score': 0, 'angle_sht': 0,
                                         'energy_sht': 0, 'energy_rec': 0,
                                         'angle_rec': 0, 'dist_ball': 0}
        if self.frame.robots_blue[self.receiver_id].infrared:
            self.receiver_id = 1 - self.receiver_id
            reward[0] += 1
            reward[1] += 1
            self.reward_shaping_total['pass_score'] += 1
        else:
            shooter_id = int(not self.receiver_id)
            rw_angle_shooter, receiver_rw = self.__angle_reward()
            en_shooter, en_receiver = self.__energy_rw()
            rw_en_receiver = w_en*en_receiver
            rw_angle_receiver = w_angle * receiver_rw
            ball = np.array([self.frame.ball.x, self.frame.ball.y])
            recv = self.frame.robots_blue[self.receiver_id]
            recv = np.array([recv.x, recv.y])
            reward[self.receiver_id] += rw_en_receiver + rw_angle_receiver

            rw_en_shooter = 0
            dist_ball = 0
            _, kick, _ = self.actions[shooter_id]
            rw_angle_shooter *= w_angle_sht if kick else w_angle
            if kick > 0:
                reward[shooter_id] = rw_angle_shooter
            else:
                dist_ball = -np.linalg.norm(recv - ball)*w_dist
                rw_en_shooter = w_en*en_shooter
                reward[shooter_id] += rw_en_shooter + \
                    rw_angle_shooter + dist_ball

            self.reward_shaping_total['dist_ball'] += dist_ball
            self.reward_shaping_total['energy_sht'] += rw_en_shooter
            self.reward_shaping_total['energy_rec'] += rw_en_receiver
            self.reward_shaping_total['angle_sht'] += rw_angle_shooter
            self.reward_shaping_total['angle_rec'] += rw_angle_receiver
        if done:
            reward[0] -= 1
            reward[1] -= 1

        return reward, done

    def __get_shooter_receiver_vec(self):
        shooter_id = 0 if self.receiver_id else 1
        receiver = np.array([self.frame.robots_blue[self.receiver_id].x,
                             self.frame.robots_blue[self.receiver_id].y])
        shooter = np.array([self.frame.robots_blue[shooter_id].x,
                            self.frame.robots_blue[shooter_id].y])
        shooter_receiver = receiver - shooter
        shooter_receiver = shooter_receiver/np.linalg.norm(shooter_receiver)
        return shooter_receiver

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        pos_frame: Frame = Frame()
        def x(): return random.uniform(-1.5, 1.5)

        pos_frame.ball = Ball(x=0, y=-1.)
        pos_frame.robots_blue[0] = Robot(
            x=pos_frame.ball.x, y=pos_frame.ball.y-0.1, theta=90
        )
        pos_frame.robots_blue[1] = Robot(x=0, y=-pos_frame.ball.y, theta=270)

        return pos_frame

    def __wrong_ball(self):
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        recv = np.array([self.frame.robots_blue[self.receiver_id].x,
                         self.frame.robots_blue[self.receiver_id].y])
        line_check = ball[1] > recv[1] if self.receiver_id else ball[1] < recv[1]
        dist_ball = np.linalg.norm(ball - recv)
        recv_check = dist_ball <= 0.11 and not self.frame.robots_blue[self.receiver_id].infrared
        return line_check or recv_check

    def __energy_rw(self):
        shooter_id = 0 if self.receiver_id else 1
        _, kick, dribbling = self.actions[shooter_id]
        kick = 1 if kick > 0 else 0
        if self.last_frame.robots_blue[shooter_id].infrared:
            self.holding_steps += 1
            kick = 0
        else:
            self.holding_steps = 0
        dribbling = 0.05 if dribbling > 0 else 0
        shooter_energy = kick + dribbling
        _, kick, dribbling = self.actions[self.receiver_id]
        kick = 1 if kick > 0 else 0
        dribbling = 0.5 if dribbling > 0 else 0
        receiver_energy = kick + dribbling
        return -shooter_energy, -receiver_energy

    def __angle_reward(self):
        shooter_id = 0 if self.receiver_id else 1
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        shooter = np.array([self.last_frame.robots_blue[shooter_id].x,
                            self.last_frame.robots_blue[shooter_id].y])
        receiver = np.array([self.frame.robots_blue[self.receiver_id].x,
                             self.frame.robots_blue[self.receiver_id].y])
        receiver_angle = self.frame.robots_blue[self.receiver_id].theta
        receiver_ball = receiver - ball
        receiver_ball = receiver_ball/np.linalg.norm(receiver_ball)
        receiver_ball_angle = np.arctan2(
            receiver_ball[1], receiver_ball[0]) + np.pi
        min_diff = np.rad2deg(receiver_ball_angle) - receiver_angle
        receiver_angle_reward = (min_diff + 180) % 360 - 180
        receiver_angle_reward = np.cos(np.deg2rad(receiver_angle_reward))

        ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])
        shooter_ball = ball - shooter
        dist_ball = np.linalg.norm(shooter_ball)
        shooter_ball /= dist_ball
        cos_shooter = np.dot(shooter_ball, self.original_vec)
        cos_shooter *= 1 if self.receiver_id else -1
        shooter_angle_reward = 0
        if dist_ball > 0.11:
            if self.holding_steps != 0:
                if np.rad2deg(np.arccos(cos_shooter)) > 1:
                    shooter_angle_reward = -1
                else:
                    shooter_angle_reward = 1
        else:
            if self.holding_steps < 20:
                shooter_angle_reward = cos_shooter
            else:
                shooter_angle_reward -= 0.5
        return shooter_angle_reward, receiver_angle_reward
