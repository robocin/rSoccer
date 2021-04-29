import math
import os
import random
import time

import gym
import numpy as np
import torch
from rsoccer_gym.Entities import Frame, Robot
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.vss.env_gk.attacker.models import DDPGActor, GaussianPolicy


class rSimVSSGK(VSSBaseEnv):
    """
    Description:
        This environment controls a single robot football goalkeeper against an attacker in the VSS League 3v3 match
        robots_blue[0]      -> Goalkeeper
        robots_yellow[0]    -> Attacker
    Observation:
        Type: Box(40)
        Goalkeeper:
            Num                 Observation normalized
            0                   Ball X
            1                   Ball Y
            2                   Ball Vx
            3                   Ball Vy
            4 + (7 * i)         id i Blue Robot X
            5 + (7 * i)         id i Blue Robot Y
            6 + (7 * i)         id i Blue Robot sin(theta)
            7 + (7 * i)         id i Blue Robot cos(theta)
            8 + (7 * i)         id i Blue Robot Vx
            9 + (7 * i)         id i Blue Robot Vy
            10 + (7 * i)        id i Blue Robot v_theta
            25 + (5 * i)        id i Yellow Robot X
            26 + (5 * i)        id i Yellow Robot Y
            27 + (5 * i)        id i Yellow Robot Vx
            28 + (5 * i)        id i Yellow Robot Vy
            29 + (5 * i)        id i Yellow Robot v_theta
        Attacker:
            Num                 Observation normalized
            0                   Ball X
            1                   Ball Y
            2                   Ball Vx
            3                   Ball Vy
            4 + (7 * i)         id i Yellow Robot -X
            5 + (7 * i)         id i Yellow Robot Y
            6 + (7 * i)         id i Yellow Robot sin(theta)
            7 + (7 * i)         id i Yellow Robot -cos(theta)
            8 + (7 * i)         id i Yellow Robot -Vx
            9 + (7 * i)         id i Yellow Robot Vy
            10 + (7 * i)        id i Yellow Robot -v_theta
            25 + (5 * i)        id i Blue Robot -X
            26 + (5 * i)        id i Blue Robot Y
            27 + (5 * i)        id i Blue Robot -Vx
            28 + (5 * i)        id i Blue Robot Vy
            29 + (5 * i)        id i Blue Robot -v_theta
    Actions:
        Type: Box(2, )
        Num     Action
        0       id 0 Blue Robot Wheel 1 Speed (%)
        1       id 0 Blue Robot Wheel 2 Speed (%)
    Reward:
        Sum Of Rewards:
            Defense
            Ball leaves the goalkeeper's area
            Move to Ball_Y
            Distance From The Goalkeeper to Your Goal Bar
        Penalized By:
            Goalkeeper leaves the goalkeeper's area
    Starting State:
        Random Ball Position 
        Random Attacker Position
        Random Goalkeeper Position Inside the Goalkeeper's Area
    Episode Termination:
        Attacker Goal
        Goalkeeper leaves the goalkeeper's area
        Ball leaves the goalkeeper's area
    """

    atk_target_rho = 0
    atk_target_theta = 0
    atk_target_x = 0
    atk_target_y = 0

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                         time_step=0.025)

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2, ), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-1,
                                                high=1,
                                                shape=(40,),
                                                dtype=np.float32)

        self.last_frame = None
        self.energy_penalty = 0
        self.reward_shaping_total = None
        self.attacker = None
        self.previous_ball_direction = []
        self.isInside = False
        self.ballInsideArea = False
        self.load_atk()
        print('Environment initialized')
    
    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def load_atk(self):
        device = torch.device('cuda')
        atk_path = os.path.dirname(os.path.realpath(
            __file__)) + '/attacker/atk_model.pth'
        self.attacker = DDPGActor(40, 2)
        print(atk_path)
        atk_checkpoint = torch.load(atk_path, map_location=device)
        self.attacker.load_state_dict(atk_checkpoint['state_dict_act'])
        self.attacker.eval()

    def _atk_obs(self):
        observation = []
        observation.append(self.norm_pos(-self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(-self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))
        
        #  we reflect the side that the attacker is attacking,
        #  so that he will attack towards the goal where the goalkeeper is
        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(-self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(
                -np.cos(np.deg2rad(self.frame.robots_yellow[i].theta))
            )
            observation.append(self.norm_v(-self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(-self.frame.robots_yellow[i].v_theta))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(-self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(self.norm_v(-self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(-self.frame.robots_blue[i].v_theta))

        return np.array(observation)

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
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        return np.array(observation)

    def _get_commands(self, actions):
        commands = []
        self.energy_penalty = -(abs(actions[0] * 100) + abs(actions[1] * 100))
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))

        # Send random commands to the other robots
        for i in range(1, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        atk_action = self.attacker.get_action(self._atk_obs())
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(atk_action)
        # we invert the speed on the wheels because of the attacker's reflection on the Y axis.
        commands.append(Robot(yellow=True, id=0, v_wheel0=v_wheel1,
                              v_wheel1=v_wheel0))
        for i in range(1, self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue+i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed , right_wheel_speed

    def _calculate_future_point(self, pos, vel):
        if vel[0] > 0:
            goal_center = np.array([self.field_params['field_length'] / 2, 0])
            pos = np.array(pos)
            dist = np.linalg.norm(goal_center - pos)
            time_to_goal = dist/np.sqrt(vel[0]**2 + vel[1]**2)
            future_x = pos[0] + vel[0]*time_to_goal
            future_y = pos[1] + vel[1]*time_to_goal

            return future_x, future_y
        else:
            return None

    def __move_reward(self):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''
        
        if self.frame.ball.x < self.field_params['field_length'] / 4  - 5:
            ball = np.array([self.frame.ball.x, self.frame.ball.y])
            robot = np.array([self.frame.robots_blue[0].x,
                            self.frame.robots_blue[0].y])
            robot_vel = np.array([self.frame.robots_blue[0].v_x,
                                self.frame.robots_blue[0].v_y])
            robot_ball = ball - robot
            robot_ball = robot_ball/np.linalg.norm(robot_ball)

            move_reward = np.dot(robot_ball, robot_vel)

            move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        else:
            move_reward = 0
        return move_reward

    def __move_reward_y(self):
        '''Calculate Move to ball_Y reward

        Cosine between the robot vel_Y vector and the vector robot_Y -> ball_Y.
        This indicates rather the robot is moving towards the ball_Y or not.
        '''
        ball = np.array([np.clip(self.frame.ball.y, -0.35, 0.35)])
        robot = np.array([self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_y])
        robot_ball = ball - robot
        robot_ball = robot_ball/np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def __defended_ball(self):
        '''Calculate Defended Ball Reward 
        
        Create a zone between the goalkeeper and if the ball enters this zone
        keep the ball speed vector norm to know the direction it entered, 
        and if the ball leaves the area in a different direction it means 
        that the goalkeeper defended the ball.
        '''
        pos = np.array([self.frame.robots_blue[0].x,
                        self.frame.robots_blue[0].y])
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        distance_gk_ball = np.linalg.norm(pos - ball) * 100 
        field_half_length = self.field_params['field_length'] / 2

        defense_reward = 0
        if distance_gk_ball < 8 and not self.isInside:
            self.previous_ball_direction.append((self.frame.ball.v_x + 0.000001) / \
                                                (abs(self.frame.ball.v_x)+ 0.000001))
            self.previous_ball_direction.append((self.frame.ball.v_y + 0.000001) / \
                                                (abs(self.frame.ball.v_y) + 0.000001))
            self.isInside = True
        elif self.isInside:
            direction_ball_vx = (self.frame.ball.v_x + 0.000001) / \
                                (abs(self.frame.ball.v_x) + 0.000001)
            direction_ball_vy = (self.frame.ball.v_y + 0.000001) / \
                                (abs(self.frame.ball.v_x) + 0.000001)

            if (self.previous_ball_direction[0] != direction_ball_vx or \
                self.previous_ball_direction[1] != direction_ball_vy) and \
                self.frame.ball.x > -field_half_length+0.1:
                self.isInside = False
                self.previous_ball_direction.clear()
                defense_reward = 1
        
        return defense_reward

    def __ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field_params['field_length'] * 100
        half_lenght = (self.field_params['field_length'] / 2.0)\
            + self.field_params['goal_depth']

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def _calculate_reward_and_done(self):
        done = False
        reward = 0
        goal_score = 0
        move_reward = 0
        ball_potential = 0
        move_y_reward = 0
        dist_robot_own_goal_bar = 0
        ball_defense_reward = 0
        ball_leave_area_reward = 0

        w_defense = 1.8
        w_move = 0.2
        w_ball_pot = 0.1
        w_move_y  = 0.3
        w_distance = 0.1
        w_blva = 2.0

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'goal_score': 0, 'move': 0,
                                         'ball_grad': 0, 'energy': 0,
                                         'goals_blue': 0, 'goals_yellow': 0,
                                         'defense': 0,'ball_leave_area': 0,
                                         'move_y': 0, 'distance_own_goal_bar': 0 }

        # This case the Goalkeeper leaves the gk area
        if self.frame.robots_blue[0].x > -0.63 or self.frame.robots_blue[0].y > 0.4 \
            or self.frame.robots_blue[0].y < -0.4: 
            reward = -5
            done = True
            self.isInside = False
            self.ballInsideArea = False

        elif self.last_frame is not None:
            self.previous_ball_potential = None
            
            # If the ball entered in the gk area
            if (not self.ballInsideArea) and self.frame.ball.x < -0.6 and (self.frame.ball.y < 0.35 \
               and self.frame.ball.y > -0.35):
                self.ballInsideArea = True

            # If the ball entered in the gk area and leaves it
            if self.ballInsideArea and (self.frame.ball.x > -0.6 or self.frame.ball.y > 0.35 \
               or self.frame.ball.y < -0.35):
                ball_leave_area_reward = 1 
                self.ballInsideArea = False
                done = True

            # If the enemy scored a goal
            if self.frame.ball.x < -(self.field_params['field_length'] / 2):
                self.reward_shaping_total['goals_yellow'] += 1
                self.reward_shaping_total['goal_score'] -= 1
                goal_score = -2 
                self.ballInsideArea = False

            if goal_score != 0:
                reward = goal_score

            else:
                move_reward = self.__move_reward()
                move_y_reward = self.__move_reward_y()
                ball_defense_reward = self.__defended_ball() 
                dist_robot_own_goal_bar = -self.field_params['field_length'] / \
                    2 + 0.15 - self.frame.robots_blue[0].x

                reward = w_move_y * move_y_reward + \
                         w_distance * dist_robot_own_goal_bar + \
                         w_defense * ball_defense_reward + \
                         w_blva * ball_leave_area_reward

                self.reward_shaping_total['move'] += w_move * move_reward
                self.reward_shaping_total['move_y'] += w_move_y * move_y_reward
                self.reward_shaping_total['ball_grad'] += w_ball_pot * ball_potential
                self.reward_shaping_total['distance_own_goal_bar'] += w_distance * dist_robot_own_goal_bar
                self.reward_shaping_total['defense'] += ball_defense_reward * w_defense
                self.reward_shaping_total['ball_leave_area'] += w_blva * ball_leave_area_reward
            self.last_frame = self.frame
        done = goal_score != 0 or done

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
