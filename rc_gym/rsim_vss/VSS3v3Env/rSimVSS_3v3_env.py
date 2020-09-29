import gym
import math
import numpy as np

from rc_gym.Utils import distance
from rc_gym.Entities import Robot
from rc_gym.rsim_vss.rSimVSS_env import rSimVSSEnv


class rSimVSS3v3Env(rSimVSSEnv):
    """
    Description:
        This environment controls a robot soccer in VSS League 3v3 match
    Observation:
        Type: Box(29)
        Num     Observation units in meters
        0       Ball X
        1       Ball Y
        2       Ball Z
        3       Ball Vx
        4       Ball Vy
        5       id 0 Blue Robot X
        6       id 0 Blue Robot Y
        7       id 0 Blue Robot Vx
        8       id 0 Blue Robot Vy
        9       id 1 Blue Robot X
        10      id 1 Blue Robot Y
        11      id 1 Blue Robot Vx
        12      id 1 Blue Robot Vy
        13      id 2 Blue Robot X
        14      id 2 Blue Robot Y
        15      id 2 Blue Robot Vx
        16      id 2 Blue Robot Vy
        17      id 0 Yellow Robot X
        18      id 0 Yellow Robot Y
        19      id 0 Yellow Robot Vx
        20      id 0 Yellow Robot Vy
        21      id 1 Yellow Robot X
        22      id 1 Yellow Robot Y
        23      id 1 Yellow Robot Vx
        24      id 1 Yellow Robot Vy
        25      id 2 Yellow Robot X
        26      id 2 Yellow Robot Y
        27      id 2 Yellow Robot Vx
        28      id 2 Yellow Robot Vy
    Actions:
        Type: Box(1, 2)
        Num     Action
        0       id 0 Blue Robot Wheel 1 Speed (%)
        1       id 0 Blue Robot Wheel 2 Speed (%)
    Reward:
        1 if Blue Team Goal
        -1 if Yellow Team Goal
    Starting State:
        TODO
    Episode Termination:
        Match time
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=2, n_robots_yellow=1)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1, 2), dtype=np.float32)

        # Define observation space bound
        bound_x = (self.field_params['field_length'] /
                   2) + self.field_params['goal_depth']
        bound_y = self.field_params['field_width'] / 2
        bound_v = 2
        # ball bounds
        obs_bounds = [bound_x, bound_y] + [0.2] + [bound_v, bound_v]
        # concatenate robot bounds
        obs_bounds = obs_bounds + self.n_robots_blue * \
            [bound_x, bound_y, bound_v, bound_v]\
            + self.n_robots_yellow * [bound_x, bound_y, bound_v, bound_v]
        obs_bounds = np.array(obs_bounds, dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-obs_bounds, high=obs_bounds, dtype=np.float32)

        self.last_frame = None
        print('Environment initialized')

    def _frame_to_observations(self):
        observation = []

        observation.append(self.frame.ball.x)
        observation.append(self.frame.ball.y)
        observation.append(self.frame.ball.z)
        observation.append(self.frame.ball.v_x)
        observation.append(self.frame.ball.v_y)

        for i in range(self.n_robots_blue):
            observation.append(self.frame.robots_blue[i].x)
            observation.append(self.frame.robots_blue[i].y)
            observation.append(self.frame.robots_blue[i].v_x)
            observation.append(self.frame.robots_blue[i].v_y)

        for i in range(self.n_robots_yellow):
            observation.append(self.frame.robots_yellow[i].x)
            observation.append(self.frame.robots_yellow[i].y)
            observation.append(self.frame.robots_yellow[i].v_x)
            observation.append(self.frame.robots_yellow[i].v_y)

        return np.array(observation)

    def _get_commands(self, actions):
        commands = []

        commands.append(Robot(yellow=False, id=0, v_wheel1=actions[0][0],
                              v_wheel2=actions[0][1]))

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        done = False

        if self.last_frame is not None:
            if self.last_frame.goals_yellow > self.frame.goals_yellow:
                reward += 1
            if self.last_frame.goals_blue > self.frame.goals_blue:
                reward -= 1

        self.last_frame = self.frame

        done = self.frame.time >= 300000

        return reward, done
