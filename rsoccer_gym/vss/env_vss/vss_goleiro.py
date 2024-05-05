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
from rsoccer_gym.vss.env_vss.shared_tcc import observations, observations_da, w_ball_grad_tcc, w_energy_tcc, w_move_tcc, goal_reward

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

class vss_goleiro(VSSBaseEnv):
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
        for _ in range(self.n_robots_blue + self.n_robots_yellow):
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
        return self.observations_atacante(), self.observations_goleiro()

    def observations_atacante(self):
        return observations_da(self)
    
    def observations_goleiro(self):
        observation = []

        max_comprimento = self.field.length + self.field.penalty_length
        max_altura = self.field.width

        vetor_x_gol_oponente = (max_comprimento/2 - self.frame.robots_yellow[0].x) / max_comprimento
        vetor_y_gol_oponente = (0 - self.frame.robots_yellow[0].y) / (max_altura/2)

        # de cima pra baixo verde e azul -> 4,71 rad -> 270 graus
        # significa que azul e verde e 90 graus
        # azul e a frente do robo, quando alinhada no eixo x ela e 0 graus
        ang = np.deg2rad(self.frame.robots_yellow[0].theta)

        angle, (v1_x, v1_y) = transform((vetor_x_gol_oponente, vetor_y_gol_oponente), ang)
        distance_rb = np.sqrt(v1_x * v1_x + v1_y * v1_y)

        observation.append(distance_rb) # vetor robo -> gol oponente
        observation.append(angle/math.pi) # vetor robo -> gol oponente


        # observacao para bola
        vetor_x_bola = (self.frame.ball.x - self.frame.robots_yellow[0].x) / max_comprimento
        vetor_y_bola = (self.frame.ball.y - self.frame.robots_yellow[0].y) / max_altura

        angle, (v1_x, v1_y) = transform((vetor_x_bola, vetor_y_bola), ang)
        distance_rb = np.sqrt(v1_x * v1_x + v1_y * v1_y)
        
        observation.append(distance_rb)
        observation.append(angle/math.pi) # vetor robo -> bola


        # observação do inimigo para o goleiro
        amigo_x = (self.frame.robots_yellow[0].x - self.frame.robots_blue[0].x) / max_comprimento
        amigo_y = (self.frame.robots_yellow[0].y - self.frame.robots_blue[0].y) / max_altura

        angle, (v1_x, v1_y) = transform((amigo_x, amigo_y), ang)
        distance_rb = np.sqrt(v1_x * v1_x + v1_y * v1_y)
        
        observation.append(distance_rb)
        observation.append(angle/math.pi)


        observation.append(np.sin(ang))
        observation.append(np.cos(ang))

        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        observation.append(self.norm_v(self.frame.robots_yellow[0].v_x))
        observation.append(self.norm_v(self.frame.robots_yellow[0].v_y))
        observation.append(self.norm_w(self.frame.robots_yellow[0].v_theta))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        # [acao_atacante, acao_goleiro]

        self.actions[0] = actions[0]
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0])
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))
        
        self.actions[1] = actions[1]
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[1])
        commands.append(Robot(yellow=True, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))
        
        return commands
    
    def _calculate_reward_and_done(self):
        reward = 0
        goal = False

        if self.reward_shaping_total is None:
            self.reward_shaping_total = {}

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            reward = -goal_reward
            goal = True
            self.reward_shaping_total['goal_bad'] = True
        elif self.frame.ball.x < -(self.field.length / 2):
            reward = 10
            self.reward_shaping_total['goal_good'] = True
            goal = True
        else:

            if self.last_frame is not None:

                reward += 5e-4 * self.__energy_penalty()

                goal_area_x = .7
                goal_area_y = 0

                dist_ball_from_area = np.sqrt((goal_area_x - self.frame.ball.x)**2 + (goal_area_y - self.frame.ball.y)**2)

                if dist_ball_from_area < .25:
                    reward -= 1
                else:
                    reward += .25

                dist = np.sqrt((goal_area_x - self.frame.robots_yellow[0].x)**2 + (goal_area_y - self.frame.robots_yellow[0].y)**2)
                
                if dist > .4:
                    reward -= 5
                else:
                    reward += .25

        return reward, goal

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.2,
                                       field_half_length - 0.2)

        def y(): return random.uniform(-field_half_width + 0.2,
                                       field_half_width - 0.2)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()


        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.2

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        

        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist or pos[0] > -.1:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        pos = (.6, y())

        while places.get_nearest(pos)[1] < min_dist or pos[1] > .7 or pos[1] < -.7:
            pos = (.6, y())

        places.insert(pos)
        pos_frame.robots_yellow[0] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(1,self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

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

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[1].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[1].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty
    


