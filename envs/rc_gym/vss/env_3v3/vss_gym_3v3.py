import math
import random

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot
from rc_gym.vss.vss_gym_base import VSSBaseEnv
from rc_gym.Utils import distance, normVt, normVx, normX, normY


class VSS3v3Env(VSSBaseEnv):
    """
    Description:
        This environment controls a single robot soccer in VSS League 3v3 match
    Observation:
        Type: Box(41)
        Num     Observation units in meters
        0        Ball X
        1        Ball Y
        2        Ball Vx
        3        Ball Vy
        x        id i Blue Robot X
        x        id i Blue Robot Y
        x        id i Blue Robot sin(theta)
        x        id i Blue Robot cos(theta)
        x        id i Blue Robot Vx
        x        id i Blue Robot Vy
        x        id i Blue Robot v_theta
        x        id i Blue Robot last_command[0]
        x        id i Blue Robot last_command[1]
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
        0       id 0 Blue Linear Speed (%)
        1       id 0 Blue Angular 2 Speed (%)
    Reward:
        1 if Blue Team Goal
        -1 if Yellow Team Goal
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
            low=-0, high=1, shape=(53, ) dtype=np.float32)
        print('Environment initialized')

    def _frame_to_observations(self):

        observation = []

        width = 1.3/2.0
        lenght = (1.5/2.0) + 0.1
        
        observation.append(1 - (self.frame.time / 300))
        observation.append(normX(lenght + self.frame.ball.x))
        observation.append(normX(width - self.frame.ball.y))
        observation.append(normVx(self.frame.ball.v_x))
        observation.append(normVx(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(normX(lenght + self.frame.robots_blue[i].x)) # TODO
            observation.append(normX(width - self.frame.robots_blue[i].y))  # TODO
            observation.append(normX(lenght + self.frame.robots_blue[i].x))
            observation.append(normX(width - self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(-self.frame.robots_blue[i].theta)))
            observation.append(
                np.cos(np.deg2rad(-self.frame.robots_blue[i].theta)))
            observation.append(normVx(self.frame.robots_blue[i].v_x))
            observation.append(normVx(self.frame.robots_blue[i].v_y))
            observation.append(normVt(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(normX(lenght + self.frame.robots_yellow[i].x))
            observation.append(normX(width - self.frame.robots_yellow[i].y))
            observation.append(
                np.sin(np.deg2rad(-self.frame.robots_yellow[i].theta)))
            observation.append(
                np.cos(np.deg2rad(-self.frame.robots_yellow[i].theta)))
            observation.append(normVx(self.frame.robots_yellow[i].v_x))
            observation.append(normVx(self.frame.robots_yellow[i].v_y))
            observation.append(normVt(self.frame.robots_yellow[i].v_theta))

        return np.array(observation)

    def _get_commands(self, actions):
        commands = []

        v_wheel1, v_wheel2 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))

        # Send random commands to the other robots
        v_wheel1, v_wheel2 = self._actions_to_v_wheels(
            self.action_space.sample())
        commands.append(Robot(yellow=False, id=1, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))
        v_wheel1, v_wheel2 = self._actions_to_v_wheels(
            self.action_space.sample())
        commands.append(Robot(yellow=False, id=2, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))
        v_wheel1, v_wheel2 = self._actions_to_v_wheels(
            self.action_space.sample())
        commands.append(Robot(yellow=True, id=0, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))
        v_wheel1, v_wheel2 = self._actions_to_v_wheels(
            self.action_space.sample())
        commands.append(Robot(yellow=True, id=1, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))
        v_wheel1, v_wheel2 = self._actions_to_v_wheels(
            self.action_space.sample())
        commands.append(Robot(yellow=True, id=2, v_wheel1=v_wheel1,
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
            self.previous_ball_potential = None
            # if self.frame.goals_blue > self.last_frame.goals_blue:
            #     goal_score = 1
            # if self.frame.goals_yellow > self.last_frame.goals_yellow:
            #     goal_score = -1
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
                # Ball Potential Reward : Reward the ball for moving in
                # the opponent goal direction and away from team goal
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
                ball_potential = dist_ball_own_goal_center \
                    - dist_ball_enemy_goal_center

                ball_grad = (dist_ball_own_goal_center
                             - prev_dist_ball_own_goal_center) \
                    + (prev_dist_ball_enemy_goal_center -
                       dist_ball_enemy_goal_center)

                energy_penalty = - \
                    (abs(self.sent_commands[0].v_wheel1) +
                     abs(self.sent_commands[0].v_wheel2))

                # Colisions Reward : Penalty when too close to teammates TODO
                reward = w_move * move_reward + \
                    w_ball_grad * ball_grad + \
                    w_energy * energy_penalty
                # w_ball_pot * ball_potential + \
                # + w_collision * collisions

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
        linear_speed_desired = actions[0] * self.simulator.linear_speed_range
        angular_speed_desired = actions[1] * self.simulator.angular_speed_range

        # calculate wheels' linear speeds:
        if linear_speed_desired + angular_speed_desired > self.simulator.linear_speed_range:
            right_wheel_overshoot = linear_speed_desired + \
                angular_speed_desired - self.simulator.linear_speed_range
            left_wheel_speed = linear_speed_desired - \
                angular_speed_desired - right_wheel_overshoot
            right_wheel_speed = self.simulator.linear_speed_range
        elif linear_speed_desired + angular_speed_desired < -self.simulator.linear_speed_range:
            right_wheel_overshoot = linear_speed_desired + \
                angular_speed_desired + self.simulator.linear_speed_range
            left_wheel_speed = linear_speed_desired - \
                angular_speed_desired - right_wheel_overshoot
            right_wheel_speed = -self.simulator.linear_speed_range
        elif linear_speed_desired - angular_speed_desired > self.simulator.linear_speed_range:
            left_wheel_overshoot = linear_speed_desired - \
                angular_speed_desired - self.simulator.linear_speed_range
            left_wheel_speed = self.simulator.linear_speed_range
            right_wheel_speed = linear_speed_desired + \
                angular_speed_desired - left_wheel_overshoot
        elif linear_speed_desired - angular_speed_desired < -self.simulator.linear_speed_range:
            left_wheel_overshoot = linear_speed_desired - \
                angular_speed_desired + self.simulator.linear_speed_range
            left_wheel_speed = -self.simulator.linear_speed_range
            right_wheel_speed = linear_speed_desired + \
                angular_speed_desired - left_wheel_overshoot
        else:
            left_wheel_speed = linear_speed_desired - angular_speed_desired
            right_wheel_speed = linear_speed_desired + angular_speed_desired

        return left_wheel_speed, right_wheel_speed
