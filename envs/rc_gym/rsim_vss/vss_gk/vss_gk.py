import math
import os
import random

import gym
import numpy as np
import torch
from rc_gym.Entities import Frame, Robot
from rc_gym.rsim_vss.rSimVSS_env import rSimVSSEnv
from rc_gym.rsim_vss.vss_gk.attacker.models import DDPGActor, GaussianPolicy
from rc_gym.Utils import distance, normVt, normVx, normX


class rSimVSSGK(rSimVSSEnv):
    """
    Description:
        This environment controls a single robot soccer in VSS League 3v3 match
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
        1 if time > 30000
        -1 if Yellow Team Goal
    Starting State:
        TODO
    Episode Termination:
        Match time
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2, ), dtype=np.float32)

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
        self.energy_penalty = 0
        self.attacker = None
        print('Environment initialized')

    # def load_atk(self):
    #     device = torch.device('cpu')
    #     atk_path = os.path.dirname(os.path.realpath(
    #         __file__)) + '/attacker/attacker_model.pth'
    #     self.attacker = GaussianPolicy(53, 2)
    #     atk_checkpoint = torch.load(atk_path, map_location=device)
    #     self.attacker.load_state_dict(atk_checkpoint['state_dict_act'])

    # def _atk_obs(self):
    #     observation = []

    #     observation.append(self.frame.ball.x)
    #     observation.append(self.frame.ball.y)
    #     observation.append(self.frame.ball.z)
    #     observation.append(self.frame.ball.v_x)
    #     observation.append(self.frame.ball.v_y)

    #     for i in range(self.n_robots_blue):
    #         observation.append(self.frame.robots_blue[i].x)
    #         observation.append(self.frame.robots_blue[i].y)
    #         observation.append(self.frame.robots_blue[i].v_x)
    #         observation.append(self.frame.robots_blue[i].v_y)

    #     for i in range(self.n_robots_yellow):
    #         observation.append(self.frame.robots_yellow[i].x)
    #         observation.append(self.frame.robots_yellow[i].y)
    #         observation.append(self.frame.robots_yellow[i].v_x)
    #         observation.append(self.frame.robots_yellow[i].v_y)

    #     return np.array(observation)

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
        v_wheel1 = actions[0]
        v_wheel2 = actions[1]
        self.energy_penalty = -(abs(v_wheel1 * 100) + abs(v_wheel2 * 100))
        commands.append(Robot(yellow=False, id=0, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))

        # Send random commands to the other robots
        commands.append(Robot(yellow=False, id=1, v_wheel1=random.uniform(-1, 1),
                              v_wheel2=random.uniform(-1, 1)))
        commands.append(Robot(yellow=False, id=2, v_wheel1=random.uniform(-1, 1),
                              v_wheel2=random.uniform(-1, 1)))
        commands.append(Robot(yellow=True, id=0, v_wheel1=random.uniform(-1, 1),
                              v_wheel2=random.uniform(-1, 1)))
        commands.append(Robot(yellow=True, id=1, v_wheel1=random.uniform(-1, 1),
                              v_wheel2=random.uniform(-1, 1)))
        commands.append(Robot(yellow=True, id=2, v_wheel1=random.uniform(-1, 1),
                              v_wheel2=random.uniform(-1, 1)))

        return commands

    def _calculate_future_point(self, pos, vel):
        if vel[0] > 0:
            dist = distance(
                (self.field_params['field_length'] / 2, 0), (pos[0], pos[1]))

            time_to_goal = dist/np.sqrt(vel[0]**2 + vel[1]**2)
            future_x = pos[0] + vel[0]*time_to_goal
            future_y = pos[1] + vel[1]*time_to_goal

            return future_x, future_y
        else:
            return None

    def _calculate_reward_and_done(self):
        done = False
        reward = 0
        goal_score = 0
        dist_future_rew = 0
        ball_potential = 0
        time_reward = 0
        dist_robot_own_goal_bar = 0

        w_future = 0.0001
        w_ball_pot = 0.0001

        # Revisar
        w_time = 0.0001
        w_distance = 0.0001

        # Check if a goal has ocurred
        if self.last_frame is not None:
            self.previous_ball_potential = None
            if self.frame.ball.x > (self.field_params['field_length'] / 2):
                goal_score = 1
            if self.frame.ball.x < -(self.field_params['field_length'] / 2):
                goal_score = -1

            # If goal scored reward = 1 favoured, and -1 if against
            if goal_score != 0:
                reward = goal_score

            else:
                # Move Reward : Reward the robot for moving in ball direction
                future_ball = self._calculate_future_point(
                    [self.last_frame.ball.x, self.last_frame.ball.y],
                    [self.last_frame.ball.v_x, self.last_frame.ball.v_y])
                if future_ball:
                    dist_future_rew = distance(
                        (self.frame.robots_blue[0].x,
                         self.frame.robots_blue[0].y),
                        future_ball)
                else:
                    dist_future_rew = 0

                # Ball Potential Reward :

                half_field_length = (
                    (self.field_params['field_length'] / 2)
                    + self.field_params['goal_depth'])

                dist_ball_enemy_goal_center = distance(
                    (self.frame.ball.x, self.frame.ball.y),
                    (half_field_length, 0)
                )

                dist_ball_own_goal_center = distance(
                    (self.frame.ball.x, self.frame.ball.y),
                    (-half_field_length, 0)
                )

                dist_robot_own_goal_bar = self.field_params['field_length'] / \
                    2 - self.frame.robots_blue[0].x

                time_ingame = self.frame.time * 1000

                time_reward = 50 if time_ingame > 30 else 0

                ball_potential = dist_ball_own_goal_center - dist_ball_enemy_goal_center

        """
        
        Goleiro:
            Manter-se na área de defesa (entre a bola e o gol) mesmo com a bola em ataque
            >Em caso de gol sofrido, se ele estiver fora da área defensiva receberá recompensa negativa (✅)

            Evitar um gol sofrido resulta em recompensa positiva
            > (30 segundos sem tomar gol, a partir de 30, a reward vai aumentando com o tempo) (✅)
            
            Gol = Reward (?)

        """

        reward = w_future * dist_future_rew + \
            w_ball_pot * ball_potential + \
            w_time * time_reward + \
            w_distance * dist_robot_own_goal_bar

        # + w_collision * collisions

        self.last_frame = self.frame

        done = self.frame.time >= 300000 or goal_score != 0

        return reward, done

    def _get_initial_positions_frame(self):
        """
        Goalie starts at the center of the goal, striker and ball randomly.
        Other robots also starts at random positions.
        """
        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2
        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)
        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)
        pos_frame: Frame = Frame()

        pos_frame.ball.x = random.uniform(-field_half_length + 0.1,
                                          field_half_length - 0.1)
        pos_frame.ball.y = random.uniform(-field_half_width + 0.1,
                                          field_half_width - 0.1)

        pos_frame.robots_blue[0] = Robot(x=-field_half_length + 0.05,
                                         y=0.0,
                                         theta=0)
        pos_frame.robots_blue[1] = Robot(x=x(), y=y(), theta=0)
        pos_frame.robots_blue[2] = Robot(x=x(), y=y(), theta=0)

        pos_frame.robots_yellow[0] = Robot(x=x(), y=y(), theta=math.pi)
        pos_frame.robots_yellow[1] = Robot(x=x(), y=y(), theta=math.pi)
        pos_frame.robots_yellow[2] = Robot(x=x(), y=y(), theta=math.pi)

        return pos_frame
