import math
import random
from typing import Dict

import gym
import numpy as np
import rc_gym.vss.env_coach.deterministic_vss.univectorPosture as univectorPosture
import rc_gym.vss.env_coach.deterministic_vss.utils as utils
from gym.spaces import Box, Discrete
from rc_gym.Entities import Frame, Robot
from rc_gym.Utils import normVt, normVx, normX
from rc_gym.vss.env_coach import DeepAtk, DeepDef, DeepGK, DetDef, DetGK
from rc_gym.vss.env_coach.deterministic_vss.controleDefender import PID
from rc_gym.vss.env_coach.deterministic_vss.defender import \
    DefenderDeterministic
from rc_gym.vss.env_coach.deterministic_vss.goleiro import \
    GoalKeeperDeterministic
from rc_gym.vss.vss_gym_base import VSSBaseEnv


class VSSCoachEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


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
            Type: Discrete(27)
            Num     Action

        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Penalty Infractions
                Faults Infractions
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                         time_step=0.032)

        self.versus = 0
        self.num_atk_faults = 0
        self.num_penalties = 0
        self.stop_counter = 0
        low_obs_bound = [-1.2, -1.2, -1.25, -1.25]
        low_obs_bound += [-1.2, -1.2, -1, -1,
                          -1.25, -1.25, -1.2]*self.n_robots_blue
        low_obs_bound += [-1.2, -1.2, -1.25, -1.25, -1.2]*self.n_robots_yellow
        high_obs_bound = [1.2, 1.2, 1.25, 1.25]
        high_obs_bound += [1.2, 1.2, 1, 1, 1.25, 1.25, 1.2]*self.n_robots_blue
        high_obs_bound += [1.2, 1.2, 1.25, 1.25, 1.2]*self.n_robots_yellow
        low_obs_bound = np.array(low_obs_bound, dtype=np.float32)
        high_obs_bound = np.array(high_obs_bound, dtype=np.float32)
        self.formations = ["000", "001", "002", "010",
                           "011", "012", "020", "021",
                           "022", "100", "101", "102",
                           "110", "111", "112", "120",
                           "121", "122", "200", "201",
                           "202", "210", "211", "212",
                           "220", "221", "222"]

        self.action_space = Discrete(27)
        self.observation_space = Box(low=low_obs_bound,
                                     high=high_obs_bound)

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05
        atk_params = dict(n_robots_blue=self.n_robots_blue,
                          n_robots_yellow=self.n_robots_yellow,
                          linear_speed_range=self.rsim.linear_speed_range,
                          v_wheel_deadzone=self.v_wheel_deadzone)

        self.deep_atks = [DeepAtk(robot_idx=i, **atk_params)
                          for i in range(self.n_robots_blue)]
        self.deep_defs = [DeepDef(robot_idx=i, **atk_params)
                          for i in range(self.n_robots_blue)]
        self.deep_gks = [DeepGK(robot_idx=i, **atk_params)
                         for i in range(self.n_robots_blue)]
        # self.det_gks = [DetGK(i) for i in range(self.n_robots_blue)]
        # self.det_defs = [DetDef(i) for i in range(self.n_robots_blue)]


        self.blue_gks = [GoalKeeperDeterministic() for _ in range(self.n_robots_blue)]
        self.blue_pid_gks = [PID() for _ in range(self.n_robots_blue)]
        self.blue_defenders = [GoalKeeperDeterministic() for _ in range(self.n_robots_blue)]
        self.blue_pid_defs = [PID() for _ in range(self.n_robots_blue)]

        self.yellow_gks = [GoalKeeperDeterministic() for _ in range(self.n_robots_yellow)]
        self.yellow_pid_gks = [PID() for _ in range(self.n_robots_yellow)]
        self.yellow_defenders = [GoalKeeperDeterministic() for _ in range(self.n_robots_yellow)]
        self.yellow_pid_defs = [PID() for _ in range(self.n_robots_yellow)]

        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.stop_counter = 0
        self.versus = int(random.choice([0, 18, 21]))
        return super().reset()

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):

        observation = []

        observation.append(normX(self.frame.ball.x))
        observation.append(normX(self.frame.ball.y))
        observation.append(normVx(self.frame.ball.v_x))
        observation.append(normVx(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(normX(self.frame.robots_blue[i].x))
            observation.append(normX(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(normVx(self.frame.robots_blue[i].v_x))
            observation.append(normVx(self.frame.robots_blue[i].v_y))
            observation.append(normVt(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(normX(self.frame.robots_yellow[i].x))
            observation.append(normX(self.frame.robots_yellow[i].y))
            observation.append(normVx(self.frame.robots_yellow[i].v_x))
            observation.append(normVx(self.frame.robots_yellow[i].v_y))
            observation.append(normVt(self.frame.robots_yellow[i].v_theta))

        return np.array(observation)
    
    def run_deterministic_behavior(self, index, yellow, beh, p, goalie):
        width = 1.3/2.0
        lenght = (1.5/2.0) + 0.1

        ball = self.frame.ball
        robot = self.frame.robots_yellow[index] if yellow else self.frame.robots_blue[index]

        if yellow:
            angle_rob = np.deg2rad(robot.theta)
            robot_pos = ((lenght + robot.x)*100, (width + robot.y) * 100)
            ball_pos = ((lenght + ball.x) * 100, (width + ball.y) * 100)
            ball_speed = (ball.v_x * 100, ball.v_y * 100)

            allies = []
            for i in range(len(self.frame.robots_yellow)):
                robot = self.frame.robots_yellow[i]
                allies.append(((lenght + robot.x) * 100,
                               (width + robot.y) * 100))
            enemies = []
            for i in range(len(self.frame.robots_blue)):
                robot = self.frame.robots_blue[i]
                enemies.append(
                    ((lenght + robot.x) * 100, (width + robot.y) * 100))
        else:
            angle_rob = np.deg2rad(robot.theta + 180)
            if angle_rob > math.pi:
                angle_rob -= 2*math.pi
            elif angle_rob < -math.pi:
                angle_rob += 2*math.pi
            robot_pos = ((lenght - robot.x) * 100, (width - robot.y) * 100)
            ball_pos = ((lenght - ball.x) * 100, (width - ball.y) * 100)
            ball_speed = (-ball.v_x * 100, -ball.v_y * 100)

            allies = []
            for i in self.frame.robots_blue:
                robot = self.frame.robots_blue[i]
                allies.append(((lenght - robot.x) * 100,
                               (width - robot.y) * 100))
            enemies = []
            for i in self.frame.robots_yellow:
                robot = self.frame.robots_yellow[i]
                enemies.append(
                    ((lenght - robot.x) * 100, (width - robot.y) * 100))

        # print(angle_rob)
        obj_pos = None

        obj_pos = beh.decideAction(ball_pos, ball_speed, robot_pos)

        # print(obj_pos)
        ret = None
        if(obj_pos == None):
            speeds = utils.spin(robot_pos, ball_pos, ball_speed)
            ret = (speeds[0], speeds[1])

        else:
            if(goalie):
                next_step = univectorPosture.update(
                    ball_pos, robot_pos, obj_pos, allies, enemies, index)
                speeds = p.run(angle_rob, next_step, robot_pos)
            else:
                next_step = univectorPosture.update(
                    ball_pos, robot_pos, obj_pos, allies, enemies, index)
                speeds = p.run(angle_rob, next_step, robot_pos)
            ret = (speeds[0]*0.02, speeds[1]*0.02)

        return ret

    def _get_commands(self, action):
        commands = []
        formation = self.formations[action]

        # Send random commands to the other robots
        for i in range(self.n_robots_blue):
            role = int(formation[i])
            if role == 0:
                v_wheel1, v_wheel2 = self.deep_atks[i](self.frame)
            elif role == 1:
                v_wheel1, v_wheel2 = self.run_deterministic_behavior(
                    index=i,
                    yellow=False,
                    beh=self.blue_defenders[i],
                    p=self.blue_pid_defs[i],
                    goalie=False
                )
            else:
                v_wheel1, v_wheel2 = self.run_deterministic_behavior(
                    index=i,
                    yellow=False,
                    beh=self.blue_gks[i],
                    p=self.blue_pid_gks[i],
                    goalie=True
                )

            commands.append(Robot(yellow=False, id=i, v_wheel1=v_wheel1,
                                  v_wheel2=v_wheel2))

        formation = self.formations[self.versus]
        for i in range(self.n_robots_yellow):
            role = int(formation[i])
            yellow_frame = self.frame.get_yellow_frame()
            if role == 0:
                v_wheel2, v_wheel1 = self.deep_atks[i](yellow_frame)
            elif role == 1:
                v_wheel1, v_wheel2 = self.run_deterministic_behavior(
                    index=i,
                    yellow=True,
                    beh=self.yellow_defenders[i],
                    p=self.yellow_pid_defs[i],
                    goalie=False
                )
            else:
                v_wheel1, v_wheel2 = self.run_deterministic_behavior(
                    index=i,
                    yellow=True,
                    beh=self.yellow_gks[i],
                    p=self.yellow_pid_gks[i],
                    goalie=True
                )
            commands.append(Robot(yellow=True, id=i, v_wheel1=v_wheel1,
                                  v_wheel2=v_wheel2))

        return commands

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
                                          -1.0, 1.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def is_atk_fault(self):
        atk_fault = False
        bx, by = self.frame.ball.x, self.frame.ball.y
        if bx > 0.6 and abs(by) < 0.35:
            one_in_fault_area = False
            for i in range(self.n_robots_blue):
                rx = self.frame.robots_blue[i].x
                ry = self.frame.robots_blue[i].y
                if rx > 0.6 and abs(ry) < 0.35:
                    if (one_in_fault_area):
                        atk_fault = True
                    else:
                        one_in_fault_area = True
        return atk_fault

    def is_penalty(self):
        penalty = False
        bx, by = self.frame.ball.x, self.frame.ball.y
        if bx < -0.6 and abs(by) < 0.35:
            one_in_pen_area = False
            for i in range(self.n_robots_blue):
                rx = self.frame.robots_blue[i].x
                ry = self.frame.robots_blue[i].y
                if rx < -0.6 and abs(ry) < 0.35:
                    if (one_in_pen_area):
                        penalty = True
                    else:
                        one_in_pen_area = True
        return penalty

    def is_ball_stopped(self):
        same_x = self.frame.ball.x - self.last_frame.ball.x
        same_x = same_x < 1.3
        same_y = self.frame.ball.y - self.last_frame.ball.y
        same_y = same_y < 1.3
        return same_x and same_y

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        penalty = False
        fault = False
        w_ball_grad = 0.8
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {'goal_score': 0, 'ball_grad': 0,
                                         'penalties': 0, 'faults': 0,
                                         'goals_blue': 0, 'goals_yellow': 0}
        # Check if goal ocurred
        if self.frame.ball.x > (self.field_params['field_length'] / 2):
            self.reward_shaping_total['goal_score'] += 1
            self.reward_shaping_total['goals_blue'] += 1
            reward = 10
            goal = True
        elif self.frame.ball.x < -(self.field_params['field_length'] / 2):
            self.reward_shaping_total['goal_score'] -= 1
            self.reward_shaping_total['goals_yellow'] += 1
            reward = -10
            goal = True
        else:
            if self.is_penalty():
                self.num_penalties += 1
                reward = -3.5
                penalty = True
            if self.is_atk_fault():
                self.num_atk_faults += 1
                reward -= 1
                fault = True

            if self.is_ball_stopped():
                self.stop_counter += 1
                if self.stop_counter*self.time_step >= 10:
                    fault = True
                    self.stop_counter = 0
            else:
                self.stop_counter = 0

            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()

                reward += w_ball_grad * grad_ball_potential

                self.reward_shaping_total['penalties'] = self.num_penalties
                self.reward_shaping_total['ball_grad'] += w_ball_grad \
                    * grad_ball_potential
                self.reward_shaping_total['faults'] = self.num_atk_faults

        if goal or fault or penalty:
            initial_pos_frame: Frame = self._get_initial_positions_frame()
            self.rsim.reset(initial_pos_frame)
            self.frame = self.rsim.get_frame()
            self.last_frame = None

        done = self.steps * self.time_step >= 300

        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the inicial frame'''
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

        agents = []
        for i in range(self.n_robots_blue):
            pos_frame.robots_blue[i] = Robot(x=x(), y=y(), theta=theta())
            agents.append(pos_frame.robots_blue[i])

        for i in range(self.n_robots_yellow):
            pos_frame.robots_yellow[i] = Robot(x=x(), y=y(), theta=theta())
            agents.append(pos_frame.robots_blue[i])

        def same_position_ref(x, y, x_ref, y_ref, radius):
            if x >= x_ref - radius and x <= x_ref + radius and \
                    y >= y_ref - radius and y <= y_ref + radius:
                return True
            return False

        radius_ball = 0.2
        radius_robot = 0.2
        same_pos = True

        while same_pos:
            for i in range(len(agents)):
                same_pos = False
                while same_position_ref(agents[i].x, agents[i].y,
                                        pos_frame.ball.x, pos_frame.ball.y,
                                        radius_ball):
                    agents[i] = Robot(x=x(), y=y(), theta=theta())
                    same_pos = True
                for j in range(i + 1, len(agents)):
                    while same_position_ref(agents[i].x, agents[i].y,
                                            agents[j].x, agents[j].y,
                                            radius_robot):
                        agents[i] = Robot(x=x(), y=y(), theta=theta())
                        same_pos = True

        for i in range(self.n_robots_blue):
            pos_frame.robots_blue[i] = agents[i]

        for i in range(self.n_robots_blue,
                       self.n_robots_yellow + self.n_robots_blue):
            pos_frame.robots_blue[i] = agents[i]

        return pos_frame
