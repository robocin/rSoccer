import math
import random

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot
from rc_gym.vss.vss_gym_base import VSSBaseEnv
from rc_gym.Utils import distance, normVt, normVx, normX


class VSS3v3Env(VSSBaseEnv):
    """
    Description:
        This environment controls a single robot soccer in VSS League 3v3 match
    Observation:
        Type: Box(53)
        Num     Observation normalized
        0        Ball X
        1        Ball Y
        2        Ball Vx
        3        Ball Vy
        x        id i Blue Robot target_x
        x        id i Blue Robot target_y
        x        id i Blue Robot X
        x        id i Blue Robot Y
        x        id i Blue Robot sin(theta)
        x        id i Blue Robot cos(theta)
        x        id i Blue Robot Vx
        x        id i Blue Robot Vy
        x        id i Blue Robot v_theta
        x        id i Yellow Robot X
        x        id i Yellow Robot Y
        x        id i Yellow Robot sin(theta)
        x        id i Yellow Robot cos(theta)
        x        id i Yellow Robot Vx
        x        id i Yellow Robot Vy
        x        id i Yellow Robot v_theta
    Actions:
        Type: Box(2, )
        Num     Action
        0       id 0 Blue Angular Speed (%)
        1       id 0 Blue Linear Speed (%)
    Reward:
    Starting State:
        TODO
    Episode Termination:
        Match time or goal
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-0, high=1, shape=(53, ), dtype=np.float32)
        print('Environment initialized')

    def _frame_to_observations(self):

        observation = []

        width = 1.3/2.0
        lenght = (1.5/2.0) + 0.1
        observation.append(1 - (self.frame.time / 300))
        observation.append(normX(lenght + self.frame.ball.x))
        observation.append(normX(width - self.frame.ball.y))
        observation.append(normVx(self.frame.ball.v_x))
        observation.append(normVx(-self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            if self.actions != None:
                target_theta = (np.deg2rad(self.frame.robots_blue[i].theta) + (
                    self.actions[i][0] * self.simulator.angular_speed_range
                    * self.time_step * 5.3))

                target_x = self.frame.robots_blue[i].x + (
                    self.actions[i][1] * self.simulator.linear_speed_range
                    * np.cos(target_theta)) * 0.52  # * self.time_step

                target_y = self.frame.robots_blue[i].y + (
                    self.actions[i][1] * self.simulator.linear_speed_range
                    * np.sin(target_theta)) * 0.52  # * self.time_step
            else:
                target_x = self.frame.robots_blue[i].x
                target_y = self.frame.robots_blue[i].y
            observation.append(normX(lenght + target_x))
            observation.append(normX(width - target_y))
            observation.append(normX(lenght + self.frame.robots_blue[i].x))
            observation.append(normX(width - self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(-self.frame.robots_blue[i].theta)))
            observation.append(
                np.cos(np.deg2rad(-self.frame.robots_blue[i].theta)))
            observation.append(normVx(self.frame.robots_blue[i].v_x))
            observation.append(normVx(-self.frame.robots_blue[i].v_y))
            observation.append(normVt(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(normX(lenght + self.frame.robots_yellow[i].x))
            observation.append(normX(width - self.frame.robots_yellow[i].y))
            observation.append(
                np.sin(np.deg2rad(-self.frame.robots_yellow[i].theta)))
            observation.append(
                np.cos(np.deg2rad(-self.frame.robots_yellow[i].theta)))
            observation.append(normVx(self.frame.robots_yellow[i].v_x))
            observation.append(normVx(-self.frame.robots_yellow[i].v_y))
            observation.append(normVt(self.frame.robots_yellow[i].v_theta))

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

    def _calculate_reward_and_done(self):
        goal_score = 0
        done = False
        reward = 0

        w_move = 10e-5
        w_ball_grad = 10e-3
        w_energy = 10e-6
        # w_ball_pot = 10e-5

        # Check if a goal has ocurred
        if self.last_frame is not None:
            
            if self.frame.ball.x > (self.field_params['field_length'] / 2):
                goal_score = 1
            if self.frame.ball.x < -(self.field_params['field_length'] / 2):
                goal_score = -1

            # If goal scored reward = 1 favoured, and -1 if against
            if goal_score != 0:
                reward = goal_score
            else:
                # Move Reward : Reward the robot for moving in ball direction
                ball = np.array([self.frame.ball.x, self.frame.ball.y])
                robot = np.array([self.frame.robots_blue[0].x,
                                  self.frame.robots_blue[0].y])
                robot_ball = ball - robot
                
                if np.linalg.norm(robot_ball) != 0:
                    robot_ball = robot_ball/np.linalg.norm(robot_ball)

                robot_vel = np.array([self.frame.robots_blue[0].v_x,
                                      self.frame.robots_blue[0].v_y])
                if np.linalg.norm(robot_vel) != 0:
                    robot_vel = robot_vel/np.linalg.norm(robot_vel)
                # move reward = cosine between those two unit vectors above
                move_reward = np.dot(robot_ball, robot_vel)
                
                half_field_length = (self.field_params['field_length'] / 2)
                prev_dist_ball_enemy_goal_center = distance(
                    (self.last_frame.ball.x, self.last_frame.ball.y),
                    (half_field_length, 0)
                )
                dist_ball_enemy_goal_center = distance(
                    (self.frame.ball.x, self.frame.ball.y),
                    (half_field_length, 0)
                )

                prev_dist_ball_own_goal_center = distance(
                    (self.last_frame.ball.x, self.last_frame.ball.y),
                    (-half_field_length, 0)
                )

                dist_ball_own_goal_center = distance(
                    (self.frame.ball.x, self.frame.ball.y),
                    (-half_field_length, 0)
                )

                ball_grad = (dist_ball_own_goal_center
                             - prev_dist_ball_own_goal_center) \
                    + (prev_dist_ball_enemy_goal_center -
                       dist_ball_enemy_goal_center)

                energy_penalty = - \
                    (abs(self.sent_commands[0].v_wheel1) +
                     abs(self.sent_commands[0].v_wheel2))

                reward = w_move * move_reward + \
                    w_ball_grad * ball_grad + \
                    w_energy * energy_penalty

        self.last_frame = self.frame
        done = self.frame.time >= 300 or goal_score != 0
        reward = reward*100
        return reward, done

    def _get_initial_positions_frame(self):
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

        pos_frame.robots_blue[0] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_blue[1] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_blue[2] = Robot(x=x(), y=y(), theta=theta())

        pos_frame.robots_yellow[0] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_yellow[1] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_yellow[2] = Robot(x=x(), y=y(), theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        linear_speed_desired = actions[1] * self.simulator.linear_speed_range
        angular_speed_desired = actions[0] * self.simulator.angular_speed_range
        robot_radius = 0.038

        left_wheel_speed = linear_speed_desired - \
            (robot_radius * angular_speed_desired)
        right_wheel_speed = linear_speed_desired + \
            (robot_radius * angular_speed_desired)

        # Deadzone
        if -0.208 < left_wheel_speed < 0.208:
            left_wheel_speed = 0

        if -0.208 < right_wheel_speed < 0.208:
            right_wheel_speed = 0

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -2.6, 2.6)

        return left_wheel_speed, right_wheel_speed
