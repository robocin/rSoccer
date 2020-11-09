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
        0       Ball X
        1       Ball Y
        2       Ball Z
        3       Ball Vx
        4       Ball Vy
        5       id 0 Blue Robot X
        6       id 0 Blue Robot Y
        7       id 0 Blue Robot Dir
        8       id 0 Blue Robot Vx
        9       id 0 Blue Robot Vy
        10      id 0 Blue Robot VDir
        11      id 1 Blue Robot X
        12      id 1 Blue Robot Y
        13      id 1 Blue Robot Dir
        14      id 1 Blue Robot Vx
        15      id 1 Blue Robot Vy
        16      id 1 Blue Robot VDir
        17      id 2 Blue Robot X
        18      id 2 Blue Robot Y
        19      id 2 Blue Robot Dir
        20      id 2 Blue Robot Vx
        21      id 2 Blue Robot Vy
        22      id 2 Blue Robot VDir
        23      id 0 Yellow Robot X
        24      id 0 Yellow Robot Y
        25      id 0 Yellow Robot Dir
        26      id 0 Yellow Robot Vx
        27      id 0 Yellow Robot Vy
        28      id 0 Yellow Robot VDir
        29      id 1 Yellow Robot X
        30      id 1 Yellow Robot Y
        31      id 1 Yellow Robot Dir
        32      id 1 Yellow Robot Vx
        33      id 1 Yellow Robot Vy
        34      id 1 Yellow Robot VDir
        35      id 2 Yellow Robot X
        36      id 2 Yellow Robot Y
        37      id 2 Yellow Robot Dir
        38      id 2 Yellow Robot Vx
        39      id 2 Yellow Robot Vy
        40      id 2 Yellow Robot VDir
        41      Episode time
    Actions:
        Type: Box(2, )
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
        print('Environment initialized')

    def _frame_to_observations(self):
        observation = []

        observation.append(normX(self.frame.ball.x))
        observation.append(normY(self.frame.ball.y))
        observation.append(normVx(self.frame.ball.v_x))
        observation.append(normVx(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(normX(self.frame.robots_blue[i].x))
            observation.append(normY(self.frame.robots_blue[i].y))
            observation.append(normVx(self.frame.robots_blue[i].v_x))
            observation.append(normVx(self.frame.robots_blue[i].v_y))
            observation.append(normVt(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(normX(self.frame.robots_yellow[i].x))
            observation.append(normY(self.frame.robots_yellow[i].y))
            observation.append(normVx(self.frame.robots_yellow[i].v_x))
            observation.append(normVx(self.frame.robots_yellow[i].v_y))
            observation.append(normVt(self.frame.robots_yellow[i].v_theta))

        observation.append(self.frame.time/(5*60*1000))

        return np.array(observation)

    def _get_commands(self, actions):
        commands = []
        v_wheel1 = actions[0]
        v_wheel2 = actions[1]
        self.energy_penalty = -(abs(v_wheel1 * 100) + abs(v_wheel2 * 100))
        commands.append(Robot(yellow=False, id=0, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))
        
        # Send random commands to the other robots
        commands.append(Robot(yellow=False, id=1, v_wheel1=0,
                              v_wheel2=0))
        commands.append(Robot(yellow=False, id=2, v_wheel1=0,
                              v_wheel2=0))
        commands.append(Robot(yellow=True, id=0, v_wheel1=0,
                              v_wheel2=0))
        commands.append(Robot(yellow=True, id=1, v_wheel1=0,
                              v_wheel2=0))
        commands.append(Robot(yellow=True, id=2, v_wheel1=0,
                              v_wheel2=0))

        return commands

    def _calculate_reward_and_done(self):
        goal_score = 0
        done = False
        reward = 0
        
        w_move = 0.0001
        w_ball_pot = 0.0001
        w_ball_grad = 0.0001
        w_energy = 0.0001

        # Check if a goal has ocurred
        if self.last_frame is not None:
            self.previous_ball_potential = None
            if self.frame.goals_blue > self.last_frame.goals_blue:
                goal_score = 1
            if self.frame.goals_yellow > self.last_frame.goals_yellow:
                goal_score = -1
            # if self.frame.ball.x > (self.field_params['field_length'] / 2):
            #     goal_score = 1
            # if self.frame.ball.x < -(self.field_params['field_length'] / 2):
            #     goal_score = -1

            # If goal scored reward = 1 favoured, and -1 if against
            if goal_score != 0:
                reward = goal_score
            else:
                # Move Reward : Reward the robot for moving in ball direction
                prev_dist_robot_ball = distance(
                    (self.last_frame.ball.x, self.last_frame.ball.y),
                    (self.last_frame.robots_blue[0].x, self.last_frame.robots_blue[0].y)
                )
                dist_robot_ball = distance(
                    (self.frame.ball.x, self.frame.ball.y),
                    (self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
                )
                move_reward = prev_dist_robot_ball - dist_robot_ball
                
                # Ball Potential Reward : Reward the ball for moving in the opponent goal direction and away from team goal
                half_field_length = ((self.field_params['field_length'] / 2) + self.field_params['goal_depth'])
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
                ball_potential = dist_ball_own_goal_center - dist_ball_enemy_goal_center
                
                ball_grad = (dist_ball_own_goal_center - prev_dist_ball_own_goal_center) + \
                    (prev_dist_ball_enemy_goal_center - dist_ball_enemy_goal_center)
                
                # Energy Reward : Calculated at _get_commands() since it needs the action sent to robot
                
                # Colisions Reward : Penalty when too close to teammates TODO
                
                reward = w_move * move_reward + \
                    w_ball_pot * ball_potential + \
                    w_ball_grad * ball_grad + \
                    w_energy * self.energy_penalty
                    # + w_collision * collisions


        self.last_frame = self.frame
        done = self.frame.time >= 300000 or goal_score != 0

        return reward, done
    
    def _get_initial_positions_frame(self):
        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2
        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)
        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)
        pos_frame: Frame = Frame()

        pos_frame.ball.x = x()
        pos_frame.ball.y = y()

        pos_frame.robots_blue[0] = Robot(x=x(),
                                         y=y(),
                                         theta=0)
        pos_frame.robots_blue[1] = Robot(x=x(), y=y(), theta=0)
        pos_frame.robots_blue[2] = Robot(x=x(), y=y(), theta=0)

        pos_frame.robots_yellow[0] = Robot(x=x(), y=y(), theta=math.pi)
        pos_frame.robots_yellow[1] = Robot(x=x(), y=y(), theta=math.pi)
        pos_frame.robots_yellow[2] = Robot(x=x(), y=y(), theta=math.pi)
        
        return pos_frame
