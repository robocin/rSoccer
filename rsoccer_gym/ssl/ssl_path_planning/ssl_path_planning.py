import random

from utility import GoToPointEntry, go_to_point, Point2D, RobotMove, dist

from typing import Final, List

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree


class SSLPathPlanningEnv(SSLBaseEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(self, field_type=1, n_robots_yellow=0):
        super().__init__(field_type=field_type, n_robots_blue=1,
                         n_robots_yellow=n_robots_yellow, time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1, # hyp tg.
                                           shape=(3, ), dtype=np.float32)

        n_obs = 3 + 7*self.n_robots_blue + 2*self.n_robots_yellow
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10

        self.target_point: Point2D = Point2D(0, 0)
        self.target_angle: float = 0.0

        print('Environment initialized')

    def _frame_to_observations(self):

        observation: List[float] = []

        observation.append(self.norm_pos(self.target_point.x))
        observation.append(self.norm_pos(self.target_point.y))
        observation.append(self.target_angle)

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

    def _get_commands(self, action):
        entry: GoToPointEntry = GoToPointEntry()
        entry.target = Point2D(action[0] * 1000.0,
                               action[1] * 1000.0
                               )
        entry.target_angle = action[2] # (target.position() - ally.position()).angle()
        entry.using_prop_velocity = True

        angle: float = np.deg2rad(self.frame.robots_blue[0].theta)
        position: Point2D = Point2D(x=self.frame.robots_blue[0].x * 1000.0,
                                    y=self.frame.robots_blue[0].y * 1000.0)
        velocity: Point2D = Point2D(x=self.frame.robots_blue[0].v_x * 1000.0,
                                    y=self.frame.robots_blue[0].v_y * 1000.0)

        robot_move: RobotMove = go_to_point(entry=entry,
                                            agent_angle=angle,
                                            agent_position=position,
                                            agent_velocity=velocity)

        return [
            Robot(
                yellow=False,
                id=0,
                v_x=robot_move.velocity.x / 1000.0,
                v_y=robot_move.velocity.y / 1000.0,
                v_theta=robot_move.velocity_theta
            )
        ]

    def _calculate_reward_and_done(self):
        reward = 0

        target_pos: Point2D = self.target_point
        robot_pos: Point2D = Point2D(x=self.frame.robots_blue[0].x,
                                     y=self.frame.robots_blue[0].y)

        dist_robot_target = dist(target_pos, robot_pos)

        if dist_robot_target < 0.2:
            reward = 1

        done = reward

        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length: Final[float] = self.field.length / 2
        field_half_width: Final[float] = self.field.width / 2

        def get_random_x():
            return random.uniform(-field_half_length + 0.1,
                                  field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1,
                                  field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        self.target_point = Point2D(x=get_random_x(), y=get_random_y())
        self.target_angle = np.deg2rad(get_random_theta())

        min_dist = 0.2

        places = KDTree()

        # place ball outside field
        pos_frame.ball.x = 0.0
        pos_frame.ball.y = field_half_width + 0.5

        for i in range(self.n_robots_blue):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=get_random_theta())

        for i in range(self.n_robots_yellow):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(
                x=pos[0], y=pos[1], theta=get_random_theta())

        return pos_frame