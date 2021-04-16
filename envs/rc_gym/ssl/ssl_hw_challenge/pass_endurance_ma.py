import math
import random
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Ball, Frame, Robot
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv


class SSLPassEnduranceMAEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal with contested possession


        Description:

        Observation:
            Type: Box(4 + 8*n_robots_blue)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized  
            0->3     Ball [X, Y, V_x, V_y]
            4->N    id 0 Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta, is_receiver]
        Actions:
            Type: Box(N, 5)
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
            3       id i Blue Kick x Speed  (%)
            4       id i Blue Dribbler  (%) (true if % is positive)

        Reward:

        Starting State:

        Episode Termination:
            30 seconds (1200 steps) or wrong pass
    """
    original_vec = None
    max_dist = 100
    actions = {}
    shooted = False

    def __init__(self):
        super().__init__(field_type=2, n_robots_blue=2,
                         n_robots_yellow=0, time_step=0.025)
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(self.n_robots_blue, 5),
                                           dtype=np.float32)

        n_obs = 4 + 8*self.n_robots_blue
        self.holding_steps = 0
        self.stopped_steps = 0
        self.recv_angle = 270
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(self.n_robots_blue,
                                                       n_obs),
                                                dtype=np.float32)
        self.shooter_id = 0
        self.receiver_id = 1
        # scale max dist rw to 1 Considering that max possible move rw if ball and robot are in opposite corners of field
        self.move_scale = np.linalg.norm([self.field.width,
                                          self.field.length/2])
        self.ball_grad_scale = np.linalg.norm([self.field.width/2,
                                               self.field.length/2])/4
        self.energy_scale = ((160 * 4) * 1200)

        print('Environment initialized')

    def _frame_to_observations(self):
        observations = list()
        ball = []

        ball.append(self.norm_pos(self.frame.ball.x))
        ball.append(self.norm_pos(self.frame.ball.y))
        ball.append(self.norm_v(self.frame.ball.v_x))
        ball.append(self.norm_v(self.frame.ball.v_y))

        robots = list()
        for i in range(self.n_robots_blue):
            robot = list()
            robot.append(self.norm_pos(self.frame.robots_blue[i].x))
            robot.append(self.norm_pos(self.frame.robots_blue[i].y))
            robot.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robot.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            robot.append(self.norm_v(self.frame.robots_blue[i].v_x))
            robot.append(self.norm_v(self.frame.robots_blue[i].v_y))
            robot.append(self.norm_w(self.frame.robots_blue[i].v_theta))
            robot.append(int(self.receiver_id == i))
            robots.append(robot)

        for i in range(self.n_robots_blue):
            observations.append(ball + robots[i])
            for j in range(self.n_robots_blue):
                if j == i:
                    continue
                observations[i] += robots[j]
        return np.array(observations, dtype=np.float32)

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        pos_frame: Frame = Frame()
        def x(): return random.uniform(-1.5, 1.5)
        def y(): return random.uniform(1.5, -1.5)

        pos_frame.ball = Ball(x=x(), y=y())
        factor = (pos_frame.ball.y/abs(pos_frame.ball.y))
        offset = 0.115*factor
        angle = 270 if factor > 0 else 90
        pos_frame.robots_blue[self.shooter_id] = Robot(
            x=pos_frame.ball.x, y=pos_frame.ball.y+offset, theta=angle
        )
        recv_x = x()
        while abs(recv_x - pos_frame.ball.x) < 1:
            recv_x = x()
        receiver = np.array([recv_x, -pos_frame.ball.y])

        self.objective = np.array([x(), receiver[1]])

        pos_frame.robots_blue[self.receiver_id] = Robot(x=receiver[0],
                                                        y=receiver[1],
                                                        theta=270)

        return pos_frame

    def reset(self):
        self.reward_shaping_total = None
        state = super().reset()
        self.actions = {}
        self.stopped_steps = 0
        return state

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _get_commands(self, actions):
        commands = []
        self.actions = {}
        for idx in [self.shooter_id, self.receiver_id]:
            if abs(actions[idx][3]) > 0.5:
                actions[idx][3] = abs(actions[idx][3])
            else:
                actions[idx][3] = 0
            self.actions[idx] = actions[idx]
            cmd = Robot(yellow=False, id=idx,
                        v_x=actions[idx][0],
                        v_y=actions[idx][1],
                        v_theta=actions[idx][2],
                        kick_v_x=actions[idx][3],
                        dribbler=True if actions[idx][4] > 0 else False)
            commands.append(cmd)

        return commands

    def __wrong_ball(self):
        done = False
        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])

        recv = np.array([self.frame.robots_blue[self.receiver_id].x,
                         self.frame.robots_blue[self.receiver_id].y])

        ball = self.frame.ball
        robot = self.frame.robots_blue[self.receiver_id]

        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid

        # If flag is set, end episode if robot enter gk area
        if robot_in_gk_area(robot):
            done = True
            self.reward_shaping_total['rbt_in_gk_area'] += 1
        # Check ball for ending conditions
        elif abs(ball.x) > half_len or abs(ball.y) > half_wid:
            done = True
            self.reward_shaping_total['done_ball_out'] += 1

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        last_dist = np.linalg.norm(last_ball - recv)
        dist = np.linalg.norm(ball - recv)
        stopped = abs(last_dist - dist) < 0.01
        if stopped:
            self.stopped_steps += 1
        else:
            self.stopped_steps = 0
        return self.stopped_steps > 20 or done

    def _calculate_reward_and_done(self):
        w_move = 1/self.move_scale
        w_ball_grad = 1/self.ball_grad_scale
        w_energy = 1/self.energy_scale
        done = False
        reward = {f'robot_{i}': 0 for i in range(self.n_robots_blue)}
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'pass_score': 0,
                                         'ball_grad': 0,
                                         'move': 0,
                                         'done_ball_out': 0,
                                         'done_ball_out_right': 0,
                                         'done_rbt_out': 0,
                                         }
            for i in range(self.n_robots_blue):
                self.reward_shaping_total[f'robot_{i}'] = {'energy': 0}

        recv = self.frame.robots_blue[self.receiver_id]
        recv_pos = np.array([recv.x, recv.y])
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        
        grad_rw = w_ball_grad*self.__ball_grad_rw()
        move_rw = w_move*self.__move_rw()
        def energy_rw(i): return w_energy*self.__energy_pen(i)

        if recv.infrared:
            if np.linalg.norm(recv_pos - ball) <= 0.05:
                for i in range(self.n_robots_blue):
                    reward[f'robot_{i}'] = 1
                self.reward_shaping_total['pass_score'] += 1
                done = True
            else:
                reward[f'robot_{self.shooter_id}'] = -grad_rw*0.2
                reward[f'robot_{self.receiver_id}'] = move_rw*1.5
                self.reward_shaping_total['move'] += move_rw*1.5
                self.reward_shaping_total['ball_grad'] -= grad_rw*0.2

        elif np.linalg.norm(self.objective - ball) <= 0.05:
            if np.linalg.norm(recv_pos - self.objective) > 0.1:
                reward[f'robot_{self.receiver_id}'] = -move_rw*0.2
                reward[f'robot_{self.shooter_id}'] = grad_rw
                self.reward_shaping_total['move'] -= move_rw*0.2
                self.reward_shaping_total['ball_grad'] += grad_rw       
        else:
            reward[f'robot_{self.shooter_id}'] = grad_rw
            reward[f'robot_{self.receiver_id}'] = move_rw
            self.reward_shaping_total['move'] += move_rw
            self.reward_shaping_total['ball_grad'] += grad_rw

        reward[f'robot_{self.shooter_id}'] += w_energy*energy_rw(self.shooter_id)
        reward[f'robot_{self.receiver_id}'] += w_energy*energy_rw(self.receiver_id)
        for i in range(self.n_robots_blue):
            self.reward_shaping_total[f'robot_{i}'] = {
                'energy': w_energy*energy_rw(i)
            }
        if self.__wrong_ball() or self.holding_steps > 15:
            for i in range(self.n_robots_blue):
                reward[f'robot_{i}'] = -1
            done = True

        return reward, done

    def __move_rw(self):
        assert(self.last_frame is not None)

        # recv pos
        recv = self.frame.robots_blue[self.receiver_id]
        recv = np.array([recv.x, recv.y])
        prev_recv = self.last_frame.robots_blue[self.receiver_id]
        prev_recv = np.array([prev_recv.x, prev_recv.y])

        # Calculate previous dist
        last_dist = np.linalg.norm(self.objective - prev_recv)

        # Calculate new dist
        dist = np.linalg.norm(self.objective - recv)

        dist_rw = last_dist - dist

        return np.clip(dist_rw, -1, 1)

    def __ball_grad_rw(self):
        assert(self.last_frame is not None)

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        ball = self.frame.ball
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_ball_dist = np.linalg.norm(self.objective - last_ball_pos)

        # Calculate new ball dist
        ball_pos = np.array([ball.x, ball.y])
        ball_dist = np.linalg.norm(self.objective - ball_pos)

        ball_dist_rw = last_ball_dist - ball_dist

        return np.clip(ball_dist_rw, -1, 1)

    def __energy_pen(self, index):
        robot = self.frame.robots_blue[index]

        # Sum of abs each wheel speed sent
        energy = abs(robot.v_wheel0)\
            + abs(robot.v_wheel1)\
            + abs(robot.v_wheel2)\
            + abs(robot.v_wheel3)

        return energy
