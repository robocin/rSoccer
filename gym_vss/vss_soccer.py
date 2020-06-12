from .utils import *
from .vss_agent import VSSAgent

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
from collections import deque

import random
from gym_vss.speed_estimator import SpeedEstimator


class Pose:
    def __init__(self, x=0, y=0, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw


class EntityState:

    def __init__(self, entity=None):

        if entity is None:
            self.pose = Pose()
            self.v_pose = Pose()
        else:
            self.pose = Pose(entity.pose.x, entity.pose.y, entity.pose.yaw)
            self.v_pose = Pose(entity.v_pose.x, entity.v_pose.y, entity.v_pose.yaw)

class VSSSoccerEnv(gym.Env):

    env_context = None

    def __init__(self):
        # -- Env parameters
        self.parameters = None
        self.is_rendering = False
        self.steps = 0
        self.frame_step_size = (1000 / 60.0)
        self.frame_skip = 5
        self.cmd_wait = self.frame_skip * self.frame_step_size  # frames x (1s in ms)/fps
        self.command_rate = round(self.cmd_wait)
        self.time_limit = -1
        self.team_name = 'Yellow'
        self.last_state = None
        self.prev_time = 0
        self.observation_space = None
        self.action_space = None
        self.state_format = None
        self.action_format = None

        # -- Game info
        self.team = [VSSAgent(i) for i in range(3)]
        self.ball_x = None
        self.ball_y = None
        self.my_goals = 0
        self.opponent_goals = 0
        self.goals_diff = 0
        self.goal_score = 0
        self.is_team_yellow = True
        self.robot_l = 4  # robot wheel-center distance
        self.robot_r = 2  # robot wheel radius

        # -- Reward info
        self.prev_ball_potential = None
        self.rw_goal = 0
        self.rw_ball_grad = 0

        # -- reward parameters
        self.w_rescaling = 10
        self.w_goal_p = 1.0 * self.w_rescaling  # goal pro
        self.w_goal_n = 1.0 * self.w_rescaling  # goal against
        self.w_move = 0.02 * self.w_rescaling  # (-1,1)
        self.w_ball_grad = 0.08 * self.w_rescaling  # (-1,1)
        self.w_collision = 0.0000 * self.w_rescaling  # {-1,0}
        self.w_ball_pot = 0.0 * self.w_rescaling  # (-1,0)
        self.w_energy = 0.0 * self.w_rescaling  # (-1,0)
        self.min_dist_move_rw = 0.0
        self.collision_distance = 12

        # -- reward shaping decays
        self.w_dec_move = 1.0
        self.w_dec_ball_grad = 1.0
        self.w_dec_collision = 1.0
        self.w_dec_ball_pot = 1.0
        self.w_dec_energy = 1.0

        # -- env control
        self.is_paused = False
        self.running_time = 0  # environment time

        # -- connection
        self.ip = '127.0.0.1'
        self.port = 5555

        self.action_dict = {0: (0, 0),
                            1: (-math.pi / 12, 0),
                            2: (math.pi / 12, 0),
                            3: (0, 12),
                            4: (0, -12),
                            }

        self.synchronize_env = None
        self.status = None

        self.rndball = None

        self.reset_rewards()

        self.ball_estim = SpeedEstimator(0.25, id=-1)
        self.my_team_estim = [SpeedEstimator(0.5, id=i) for i in range(3)]
        self.opponent_estim = [SpeedEstimator(0.25, id=3+i) for i in range(3)]

    def set_parameters(self, parameters):

        self.parameters = parameters
        self.ip = parameters['ip']
        self.port = parameters['port']
        self.is_team_yellow = parameters['is_team_yellow']

        if self.is_team_yellow:
            self.team_name = parameters['run_name'] + '-Yellow'
        else:
            self.team_name = parameters['run_name'] + '-Blue'

        print(self.team_name)

        self.frame_step_size = parameters['frame_step_size']
        self.frame_skip = parameters['frame_skip']
        self.cmd_wait = self.frame_skip * self.frame_step_size  # frames x (1s in ms)/fps
        self.command_rate = round(self.cmd_wait)
        self.time_limit = parameters['time_limit_ms']

        if 'actions' in parameters:
            self.action_dict = parameters['actions']
        self.robot_l = parameters['robot_center_wheel_dist']
        self.robot_r = parameters['robot_wheel_radius']
        for i in range(0, len(self.team)):
            self.team[i].robot_l = self.robot_l
            self.team[i].robot_r = self.robot_r
            self.team[i].alpha_base = parameters['alpha_base'][i]
            self.team[i].track_rw = parameters['track_rewards'][i]

        self.w_rescaling = parameters['w_rescaling']
        self.w_goal_p = parameters['w_goal'][0] * self.w_rescaling
        self.w_goal_n = parameters['w_goal'][1] * self.w_rescaling

        self.w_move, self.w_dec_move = parameters['w_move']
        self.w_move = self.w_move * self.w_rescaling

        self.w_ball_grad, self.w_dec_ball_grad = parameters['w_ball_grad']
        self.w_ball_grad = self.w_ball_grad * self.w_rescaling

        self.w_collision, self.w_dec_collision = parameters['w_collision']
        self.w_collision = self.w_collision * self.w_rescaling

        self.w_ball_pot, self.w_dec_ball_pot = parameters['w_ball_pot']
        self.w_ball_pot = self.w_ball_pot * self.w_rescaling

        self.w_energy, self.w_dec_energy = parameters['w_energy']
        self.w_energy = self.w_energy * self.w_rescaling

        self.min_dist_move_rw = parameters['min_dist_move_rw']
        self.collision_distance = parameters['collision_distance']


    def reset_game_vars(self):
        self.prev_ball_potential = None
        for i in range(0, len(self.team)):
            self.team[i].target_x = None
            self.team[i].target_y = None
            self.team[i].target_theta = None
            self.team[i].ball_dist = None
            self.team[i].prev_ball_dist = None
            self.ball_estim.reset()
            self.my_team_estim[i].reset()
            self.opponent_estim[i].reset()

        self.steps = 0
        self.prev_time = 0
        self.goals_diff = 0

    def set_synchronize_resets(self):
        raise NotImplementedError()

    # def process_input(self):
    #     print("process parent")
    #     pass

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, commands):

        self._set_action(commands)

        while True:

            state = self._receive_state()
            if state is None:
                print(self.team_name, "received a 'None' state")
                return [None, None, None], [0, 0, 0], [True, True, True], {}

            dt = state.time - self.prev_time

#            print("test", dt, self.is_paused)

            if dt > 0:  # a posterior reset (or crash occurred) if not
                if not self.is_paused:
                    break
                else:
                    self._update_state(state)
            # else:
            #     print("dt is zero!", "dt:%d, time: %d, prev:%d, state:%d" % (dt, self.running_time, self.prev_time, state.time))

            if state.time > 0:
                self.prev_time = state.time

        dt = state.time - self.prev_time

        self.running_time += dt
        self.prev_time = state.time
        #print("dt:%d, time: %d, prev:%d, state:%d" % (dt, self.running_time, self.prev_time, state.time))
        state.time = self.running_time

        self.last_state, reward, done = self._parse_state(state, dt)
        self.steps += 1

        # self.send_debug()
        return self.last_state, reward, done, {}

    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def reset(self):
        self.reset_game_vars()

        state = self._receive_state()
        self.prev_time = state.time
        self.running_time = 0

        print(self.team_name, "start time:%d <##### RESET #####" % self.prev_time)
        self.last_state, reward, done = self._parse_state(state, self.cmd_wait)

        return self.last_state

    def pause(self):
        self.is_paused = True
        print("Environment paused")

    def resume(self):
        self.is_paused = False
        print("Environment resumed")

    def setTeamYellow(self):
        self.is_team_yellow = True
        print("Team is Yellow")

    def setTeamBlue(self):
        self.is_team_yellow = False
        print("Team is Blue")

    def init_space_action(self):
        self.last_state, reward, done = self._parse_state(self._receive_state(), self.cmd_wait)
        shape = len(self.last_state[0])
        self.state_format = "%df" % shape

        self.observation_space = spaces.Box(low=-200, high=200, dtype=np.float32, shape=(shape,))
        if self.action_dict is None:
            self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(2,))
            self.action_format = '2f'
        else:
            self.action_space = spaces.Discrete(len(self.action_dict))
            self.action_format = 'B'

        return self.observation_space, self.action_space

    def render(self, mode='human', close=False):
        pass

    # Extension methods
    # ----------------------------

    def _receive_state(self):
        raise NotImplementedError()

    #def _parse_state(self, state, dt):
    #    raise NotImplementedError()

    def _set_action(self, commands):
        raise NotImplementedError()

    def stop_all(self):
        raise NotImplementedError()

    # Reward methods
    # ----------------------------

    def _teammate_dist(self, team):

        team_mate_dist = np.full(len(self.team), 9999.00)

        for i, t1_robot1 in enumerate(team):
            t1x = t1_robot1.pose.x
            t1y = t1_robot1.pose.y

            for j, t1_robot2 in enumerate(team):
                if i != j:
                    t2x = t1_robot2.pose.x
                    t2y = t1_robot2.pose.y
                    dist = np.linalg.norm([t1x - t2x, t1y - t2y])
                    if team_mate_dist[i] > dist:
                        team_mate_dist[i] = dist

        return team_mate_dist

    def _check_collision(self, team):

        same_team_col = [False] * len(self.team)

        for i, t1_robot1 in enumerate(team):
            t1x = t1_robot1.pose.x
            t1y = t1_robot1.pose.y

            for j, t1_robot2 in enumerate(team):
                t2x = t1_robot2.pose.x
                t2y = t1_robot2.pose.y
                if i != j and np.linalg.norm([t1x - t2x, t1y - t2y]) < self.collision_distance:
                    same_team_col[i] = True

        return same_team_col

    def _ball_potential_penalty(self):
        """
            Calculate ball potential according to this formula:
            pot = ((-sqrt((170-x)^2 + 2*(65-y)^2) + sqrt((0-x)^2 + 2*(65-y)^2))/170 - 1)/2

            the potential is zero (maximum) at the center of attack goal (170,65) and -1 at the defense goal (0,65)
            it changes twice as fast on y coordinate than on the x coordinate
        """

        dx_d = 0 - self.ball_x  # distance to defence
        dx_a = 170.0 - self.ball_x  # distance to attack
        dy = 65.0 - self.ball_y
        potential = ((-math.sqrt(dx_a ** 2 + 2 * dy ** 2)
                      + math.sqrt(dx_d ** 2 + 2 * dy ** 2)) / 170 - 1) / 2

        return potential

    def _calculate_rewards(self, goal_score, my_team, dt):
        ball_potential = self._ball_potential_penalty()  # (-1,0)

        if self.prev_ball_potential is not None:
            grad_ball_potential = clip((ball_potential - self.prev_ball_potential) * 4000 / dt, -1.0,
                                       1.0)  # (-1,1)
        else:
            grad_ball_potential = 0

        if goal_score != 0:  # A goal has happened
            if goal_score > 0:
                rewards = np.full(len(self.team), goal_score * self.w_goal_p)
            else:
                rewards = np.full(len(self.team), goal_score * self.w_goal_n)

            self.prev_ball_potential = None
            for i in range(0, len(self.team)):
                self.team[i].prev_ball_dist = None

            # record rewards and score
            self.rw_goal += rewards[0]
            self.goal_score += goal_score
            if goal_score > 0:
                self.my_goals += 1
            else:
                self.opponent_goals += 1

            for i in range(0, len(self.team)):
                self.team[i].rw_total += rewards[0]

            print("******************GOAL****************")
            print("GOAL in %d steps Reward:%.4f. %d x %d" % (
                self.steps, rewards[0], self.my_goals, self.opponent_goals))

        else:
            move_reward = np.full(len(self.team), 0.0)
            energy_penalty = np.full(len(self.team), 0.0)
            if len(my_team) < 2:
                collisions = np.full(len(self.team), 0.0)
            else:
                collisions = -8.0 / (0.001 + self._teammate_dist(my_team))

            for i in range(0, len(self.team)):
                energy_penalty[i] = -self.team[i].energy
                if self.team[i].prev_ball_dist is not None and self.team[i].ball_dist > self.min_dist_move_rw:

                    dist = (1-self.team[i].alpha_base)*(self.team[i].prev_ball_dist - self.team[i].ball_dist) + \
                           self.team[i].alpha_base*(self.team[i].prev_base_dist - self.team[i].base_dist)

                    move_reward[i] = clip(dist * 50 / dt, -1.0, 1.0)  # (-1,1)

            rewards = self.w_move * move_reward + \
                      self.w_ball_grad * grad_ball_potential + \
                      self.w_collision * collisions + \
                      self.w_ball_pot * ball_potential + \
                      self.w_energy * energy_penalty

            # record ball_grad reward
            self.rw_ball_grad += self.w_ball_grad * grad_ball_potential

            # record agent specific rewards
            for i in range(0, len(self.team)):
                self.team[i].rw_move += self.w_move * move_reward[i]
                self.team[i].rw_collision += self.w_collision * collisions[i]
                self.team[i].rw_energy += self.w_energy * energy_penalty[i]
                self.team[i].rw_total += rewards[i]

        self.prev_ball_potential = ball_potential
        return rewards

    def reset_rewards(self):
        self.my_goals = 0
        self.opponent_goals = 0
        self.goal_score = 0
        self.rw_goal = 0
        self.rw_ball_grad = 0
        for i in range(0, len(self.team)):
            self.team[i].reset_rewards()

    def _decay_reward_shaping(self):
        self.w_move = self.w_move * self.w_dec_move
        self.w_ball_grad = self.w_ball_grad * self.w_dec_ball_grad
        self.w_collision = self.w_collision * self.w_dec_collision
        self.w_ball_pot = self.w_ball_pot * self.w_dec_ball_pot
        self.w_energy = self.w_energy * self.w_dec_energy

    # Debug methods
    # ----------------------------

    def print_summary(self):
        steps = 1 if self.steps == 0 else self.steps

        print("finished in %d steps: %d x %d "
              % (self.steps, self.my_goals, self.opponent_goals))

        print("Rewards: Goal: %.4f (%.4f), Ball Gradient: %.4f (%.4f)"
              % (self.rw_goal,
                 self.rw_goal / steps,
                 self.rw_ball_grad,
                 self.rw_ball_grad / steps))

        for agent in self.team:
            print("Rewards[%d]: Move: %.4f (%.4f), Collision: %.4f (%.4f), Total: %.4f"
                  % (agent.id,
                     agent.rw_move,
                     agent.rw_move / steps,
                     agent.rw_collision,
                     agent.rw_collision / steps,
                     agent.rw_total))


    def is_in_defense_area(self, pose):
        safe = 15
        if pose.x < 25+safe and (40-safe < pose.y < 90+safe):
            return True

        return False

    def is_in_atack_area(self, pose):
        safe = 15
        if pose.x > 135-safe and (40-safe < pose.y < 90+safe):
            return True

        return False

    def state_heuristics(self, ball_state, my_team_state, opponent_team_state):

        striker1 = 0
        striker2 = 1
        keeper = 2

        ball_state_array = [EntityState(ball_state), EntityState(ball_state), EntityState(ball_state)]

        # # hold keeper behind:
        # if ball_state.pose.x > 80:  # ball x for goalkeeper >20
        #     ball_state_array[keeper].pose.x = 20
        #     ball_state_array[keeper].pose.y = min(max(40, ball_state.pose.y), 90)
        #     # print("ball x:", ball_state_array[2].pose.x)

        # prevent penalty:
        if self.is_in_defense_area(ball_state.pose):
            if self.is_in_defense_area(my_team_state[keeper].pose):
                ball_state_array[striker1].pose.x = 35
                ball_state_array[striker2].pose.x = 35

        #     if self.is_in_defense_area(my_team_state[striker1].pose):
        #         ball_state_array[keeper].pose.x = 35
        #         ball_state_array[striker2].pose.x = 35

        #     if self.is_in_defense_area(my_team_state[striker2].pose):
        #         ball_state_array[striker1].pose.x = 35
        #         ball_state_array[keeper].pose.x = 35

        #     #print("Prevent penalty")

        # # prevent attack fault:
        # if self.is_in_atack_area(ball_state.pose):
        #     if self.is_in_atack_area(my_team_state[striker1].pose):
        #         ball_state_array[striker2].pose.x = 140
        #         #print("Prevent fault")
        #     if self.is_in_atack_area(my_team_state[striker2].pose):
        #         ball_state_array[striker1].pose.x = 140
        #         #print("Prevent fault")

        #print(".")
        return ball_state_array

# # #================= Heuristics

    def _update_state(self, state):

        # if not self.is_team_yellow:
        #     state = transpose_state(state)

        if self.is_team_yellow:
            my_team = state.robots_yellow
            opponent_team = state.robots_blue
        else:
            my_team = state.robots_blue
            opponent_team = state.robots_yellow

        self.ball_x = state.balls[0].pose.x
        self.ball_y = state.balls[0].pose.y

        ball_np = np.array((self.ball_x - 4.0, self.ball_y))  # where the ball is now (a little behind it)

        balls = self.state_heuristics(state.balls[0], my_team, opponent_team)

        # Normalize ball sates:
        ball_state = [None, None, None]
        for i in range(0, len(ball_state)):
            # balls[i].v_pose.x, balls[i].v_pose.y = self.ball_estim.estimate_speed(state.balls[0].pose, state.time, False)

            ball_state[i] = (normX(balls[i].pose.x), normX(balls[i].pose.y),
                          normVx(balls[i].v_pose.x), normVx(balls[i].v_pose.y))

        my_team_state = deque()

        for idx, t1_robot in enumerate(my_team):

            # if (idx == 0):
            #     print("A", t1_robot.v_pose.y, t1_robot.v_pose.x, t1_robot.v_pose.yaw, t1_robot.pose.y, state.time)
            #     print("A", t1_robot.v_pose.yaw, t1_robot.pose.yaw, state.time)

            # t1_robot.v_pose.x, t1_robot.v_pose.y, t1_robot.v_pose.yaw = self.my_team_estim[idx].estimate_speed(t1_robot.pose, state.time)

            # if (idx == 0):
            #     print("B", t1_robot.v_pose.y, t1_robot.v_pose.x, t1_robot.v_pose.yaw, t1_robot.pose.y, state.time)
            #     #print("B", t1_robot.v_pose.yaw, t1_robot.pose.yaw, state.time)

            self.team[idx].fill(t1_robot, ball_np)

            # build the state. Encode yaw as sin(yaw) and cos(yaw)
            rbt_state = (normX(self.team[idx].target_x), normX(self.team[idx].target_y), normX(t1_robot.pose.x),
                         normX(t1_robot.pose.y), math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
                         normVx(t1_robot.v_pose.x), normVx(t1_robot.v_pose.y),
                         normVt(t1_robot.v_pose.yaw))

            my_team_state.append(rbt_state)

            if ctrl_mode & CTRL_RANDBALL:
                if idx == 0:
                    if self.rndball is None:
                        dist = 0
                    else:
                        dist = np.linalg.norm(self.rndball - np.array([self.team[idx].x, self.team[idx].y]))  # distance to current ball position

                    if dist < 10:
                        self.ball_x = ball.pose.x = random.uniform(20, 150)
                        self.ball_y = ball.pose.y = random.uniform(10, 120)
                        self.rndball = np.array((self.ball_x, self.ball_y))
                        print("Dist: %d, new rnd ball: (%d, %d)" % (dist, self.ball_x, self.ball_y))

                ball_state[idx] = (normX(self.rndball[0]), normX(self.rndball[1]), 0, 0)

            #print(idx, "lin:%.2f ang:%.2f vr:%.2f x: %.2f" % (math.sqrt(t1_robot.v_pose.x**2+t1_robot.v_pose.y**2), t1_robot.v_pose.yaw, t1_robot.v_pose.yaw*self.robot_l, t1_robot.pose.x))
            # if(idx == 0):
            #     print("lin:%.2f ang:%.2f vr:%.2f x: %.2f y:%.2f a:%.2f" % (math.sqrt(t1_robot.v_pose.x ** 2 + t1_robot.v_pose.y ** 2),
            #                                               t1_robot.v_pose.yaw,
            #                                               t1_robot.v_pose.yaw * self.robot_l, t1_robot.pose.x, t1_robot.pose.y, t1_robot.pose.yaw), state.time)

                #print(t1_robot.pose.x,t1_robot.pose.y)

        opponent_team_state = ()
        for idx, t2_robot in enumerate(opponent_team):

            # t2_robot.v_pose.x, t2_robot.v_pose.y, t2_robot.v_pose.yaw = self.opponent_estim[idx].estimate_speed(t2_robot.pose, state.time)

            opponent_team_state += (normX(t2_robot.pose.x), normX(t2_robot.pose.y), math.sin(t2_robot.pose.yaw),
                         math.cos(t2_robot.pose.yaw),
                         normVx(t2_robot.v_pose.x), normVx(t2_robot.v_pose.y),
                         normVt(t2_robot.v_pose.yaw))  # estimated values

        return ball_state, my_team_state, opponent_team_state, my_team


    def _parse_state(self, state, dt):

        ball_state, my_team_state, opponent_team_state, my_team = self._update_state(state)

        if self.is_team_yellow:
            goals_diff = state.goals_yellow - state.goals_blue
        else:
            goals_diff = state.goals_blue - state.goals_yellow

        if goals_diff > self.goals_diff:
            goal_score = 1
        elif goals_diff < self.goals_diff:
            goal_score = -1
        else:
            goal_score = 0

        self.goals_diff = goals_diff

        rewards = self._calculate_rewards(goal_score, my_team, dt)

        if self.time_limit < 0:
            done = False
            time_norm = 0.0
        else:
            done = state.time >= self.time_limit
            time_norm = max(1.0 - (state.time/self.time_limit), 0.0)

        # pack next state:
        env_states = []
        # agent_ids = [(1,0,0), (0,1,0), (0,0,1)]

        for a in range(0, len(self.team)):
            t1_state_tuple = tuple([i for sub in my_team_state for i in sub])
            #env_state = ball_state + t1_state_tuple + t2_state
            # env_state = (time_norm,) + agent_ids[a] + ball_state + t1_state_tuple + t2_state
            env_state = (time_norm,) + ball_state[a] + t1_state_tuple + opponent_team_state
            np_state = np.array(env_state, dtype=np.float32)
            env_states.append(np_state)
            my_team_state.rotate(-1)

        if done:
            self.print_summary()
            self._decay_reward_shaping()

        return env_states, rewards, [done, done, done]

# # #================ Normal

    # def _update_state(self, state):

    #     # if not self.is_team_yellow:
    #     #     state = transpose_state(state)

    #     for ball in state.balls:
    #         ball.v_pose.x, ball.v_pose.y = self.ball_estim.estimate_speed(ball.pose, state.time, False)

    #         ball_state = (normX(ball.pose.x), normX(ball.pose.y),
    #                       normVx(ball.v_pose.x), normVx(ball.v_pose.y))

    #         self.ball_x = ball.pose.x
    #         self.ball_y = ball.pose.y

    #     ball_np = np.array((self.ball_x - 4.0, self.ball_y))  # where the ball is now (a little behind it)

    #     if self.is_team_yellow:
    #         my_team = state.robots_yellow
    #         opponent_team = state.robots_blue
    #     else:
    #         my_team = state.robots_blue
    #         opponent_team = state.robots_yellow

    #     my_team_state = deque()

    #     for idx, t1_robot in enumerate(my_team):

    #         # if (idx == 0):
    #         #     print("A", t1_robot.v_pose.y, t1_robot.v_pose.x, t1_robot.v_pose.yaw, t1_robot.pose.y, state.time)
    #         #     print("A", t1_robot.v_pose.yaw, t1_robot.pose.yaw, state.time)

    #         t1_robot.v_pose.x, t1_robot.v_pose.y, t1_robot.v_pose.yaw = self.my_team_estim[idx].estimate_speed(t1_robot.pose, state.time)

    #         # if (idx == 0):
    #         #     print("B", t1_robot.v_pose.y, t1_robot.v_pose.x, t1_robot.v_pose.yaw, t1_robot.pose.y, state.time)
    #         #     #print("B", t1_robot.v_pose.yaw, t1_robot.pose.yaw, state.time)

    #         self.team[idx].fill(t1_robot, ball_np)

    #         # build the state. Encode yaw as sin(yaw) and cos(yaw)
    #         rbt_state = (normX(self.team[idx].target_x), normX(self.team[idx].target_y), normX(t1_robot.pose.x),
    #                      normX(t1_robot.pose.y), math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
    #                      normVx(t1_robot.v_pose.x), normVx(t1_robot.v_pose.y),
    #                      normVt(t1_robot.v_pose.yaw))

    #         my_team_state.append(rbt_state)

    #         if ctrl_mode & CTRL_RANDBALL:
    #             if idx == 0:
    #                 if self.rndball is None:
    #                     dist = 0
    #                 else:
    #                     dist = np.linalg.norm(self.rndball - np.array([self.team[idx].x, self.team[idx].y]))  # distance to current ball position

    #                 if dist < 10:
    #                     self.ball_x = ball.pose.x = random.uniform(20, 150)
    #                     self.ball_y = ball.pose.y = random.uniform(10, 120)
    #                     self.rndball = np.array((self.ball_x, self.ball_y))
    #                     print("Dist: %d, new rnd ball: (%d, %d)" % (dist, self.ball_x, self.ball_y))

    #             ball_state = (normX(self.rndball[0]), normX(self.rndball[1]), 0, 0)

    #         #print(idx, "lin:%.2f ang:%.2f vr:%.2f x: %.2f" % (math.sqrt(t1_robot.v_pose.x**2+t1_robot.v_pose.y**2), t1_robot.v_pose.yaw, t1_robot.v_pose.yaw*self.robot_l, t1_robot.pose.x))
    #         # if(idx == 2):
    #         #     print("lin:%.2f ang:%.2f vr:%.2f x: %.2f y:%.2f a:%.2f" % (math.sqrt(t1_robot.v_pose.x ** 2 + t1_robot.v_pose.y ** 2),
    #         #                                               t1_robot.v_pose.yaw,
    #         #                                               t1_robot.v_pose.yaw * self.robot_l, t1_robot.pose.x, t1_robot.pose.y, t1_robot.pose.yaw), state.time)

    #             #print(t1_robot.pose.x,t1_robot.pose.y)

    #     opponent_team_state = ()
    #     for idx, t2_robot in enumerate(opponent_team):

    #         t2_robot.v_pose.x, t2_robot.v_pose.y, t2_robot.v_pose.yaw = self.opponent_estim[idx].estimate_speed(t2_robot.pose, state.time)

    #         opponent_team_state += (normX(t2_robot.pose.x), normX(t2_robot.pose.y), math.sin(t2_robot.pose.yaw),
    #                      math.cos(t2_robot.pose.yaw),
    #                      normVx(t2_robot.v_pose.x), normVx(t2_robot.v_pose.y),
    #                      normVt(t2_robot.v_pose.yaw))  # estimated values

    #     return ball_state, my_team_state, opponent_team_state, my_team


    # def _parse_state(self, state, dt):

    #     ball_state, my_team_state, opponent_team_state, my_team = self._update_state(state)

    #     if self.is_team_yellow:
    #         goals_diff = state.goals_yellow - state.goals_blue
    #     else:
    #         goals_diff = state.goals_blue - state.goals_yellow

    #     if goals_diff > self.goals_diff:
    #         goal_score = 1
    #     elif goals_diff < self.goals_diff:
    #         goal_score = -1
    #     else:
    #         goal_score = 0

    #     self.goals_diff = goals_diff

    #     rewards = self._calculate_rewards(goal_score, my_team, dt)

    #     if self.time_limit < 0:
    #         done = False
    #         time_norm = 0.0
    #     else:
    #         done = state.time >= self.time_limit
    #         time_norm = max(1.0 - (state.time/self.time_limit), 0.0)

    #     # pack next state:
    #     env_states = []
    #     # agent_ids = [(1,0,0), (0,1,0), (0,0,1)]

    #     for a in range(0, len(self.team)):
    #         t1_state_tuple = tuple([i for sub in my_team_state for i in sub])
    #         #env_state = ball_state + t1_state_tuple + t2_state
    #         # env_state = (time_norm,) + agent_ids[a] + ball_state + t1_state_tuple + t2_state
    #         env_state = (time_norm,) + ball_state + t1_state_tuple + opponent_team_state
    #         np_state = np.array(env_state, dtype=np.float32)
    #         env_states.append(np_state)
    #         my_team_state.rotate(-1)

    #     if done:
    #         self.print_summary()
    #         self._decay_reward_shaping()

    #     return env_states, rewards, [done, done, done]

