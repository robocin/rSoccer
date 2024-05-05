import pickle
import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree

from rsoccer_gym.vss.env_vss.shared_tcc import observations, w_ball_grad_tcc, w_energy_tcc, w_move_tcc, goal_reward

distancia = lambda o1, o2: np.sqrt((o1.x - o2.x)**2 + (o1.y - o2.y)**2)


def menor_angulo(v1, v2):
    angle = math.acos(np.dot(v1,v2))

    if np.cross(v1,v2) > 0:
        return -angle

    return angle

def transform(v1, ang):
    mod = np.sqrt(v1[0]**2 + v1[1]**2)
    v1 = (v1[0]/mod, v1[1]/mod)

    mn = menor_angulo(v1, (math.cos(ang), math.sin(ang)))

    return mn, (math.cos(mn)*mod, math.sin(mn)*mod)

class vss_tcc_medio(VSSBaseEnv):
    """This environment controls a singl
    e robot in a VSS soccer League 3v3 match 


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot X
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Move to Ball
                Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=1, n_robots_yellow=3,
                         time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(15, ), dtype=np.float32)

        self.goleiro_teve_bola = False

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05


        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None
        for ou in self.ou_actions:
            ou.reset()
        self.goleiro_teve_bola = False
        return super().reset()

    def step(self, action):
        
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total
    
    def _frame_to_observations(self):
        return self.observations_atacante()

    def observations_atacante(self):
        return observations(self)
    
    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions[:2]
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))
        
        for i in range(2, self.n_robots_yellow):
            continue
            actions = self.ou_actions[self.n_robots_blue+i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=True, id=i-1, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))

        return commands
    
    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = .1
        w_ball_grad = 2
        w_energy = 3e-4
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'goal_score': 0, 'move': 0,
                                         'ball_grad': 0, 'energy': 0,
                                         'goals_blue': 0, 'goals_yellow': 0}

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_shaping_total['goal_score'] += 1
            self.reward_shaping_total['goals_blue'] += 1
            reward = 100
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total['goal_score'] -= 1
            self.reward_shaping_total['goals_yellow'] += 1
            reward = -100
            goal = True
        else:

            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()
                # Calculate Move ball
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()

                reward = w_move * move_reward + \
                    w_ball_grad * grad_ball_potential + \
                    w_energy * energy_penalty
                
        return reward, goal
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()


        pos_frame.ball = Ball(x=x(), y=y())
        # pos_frame.ball = Ball(x=0, y=0)

        # while pos_frame.ball.x > -.25:
        #     pos_frame.ball = Ball(x=x(), y=y())


        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        # pos_frame.robots_blue[0] = Robot(x=-.5, y=0, theta=theta())

        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist or pos[0] > -.1:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        # posicao inicial do goleiro
        pos = (.6, y())

        while places.get_nearest(pos)[1] < min_dist or pos[1] > .7 or pos[1] < -.7:
            pos = (.6, y())

        places.insert(pos)
        pos_frame.robots_yellow[0] = Robot(x=pos[0], y=pos[1], theta=180)



        # pos_frame.robots_yellow[1] = Robot(x=0, y=-.4, theta=theta())
        # pos_frame.robots_yellow[2] = Robot(x=0, y=.4, theta=theta())

        for i in range(1, self.n_robots_yellow):
            pos = (-.4-x()/3, y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (-.4-x()/3, y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())


        return pos_frame

    def _actions_to_v_wheels(self, actions):
        # recebe os valores da rede e converte (velocidade linear, velocidade angular)
        # para velocidades da roda entre -1 e 1

        # espaçamento entre rodas do carrinho, 1 para que o valor maximo seja 1 tbm
        L = 1

        vleft = (actions[0] - (actions[1]*L)/2) * 1

        vright = (actions[0] + (actions[1]*L)/2) * 1
        # print(actions, (vleft,vright))
        left_wheel_speed = vleft * self.max_v
        right_wheel_speed = vright * self.max_v

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

    def __ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0)\
            + self.field.goal_depth

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

            k = (ball_potential+1)*2
            if k > 1 and diff != 0:
                k = k*k*k
                grad_ball_potential = grad_ball_potential*k


        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def __move_reward(self):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_x,
                              self.frame.robots_blue[0].v_y])
        robot_ball = ball - robot
        robot_ball = robot_ball/np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty
    
