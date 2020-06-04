from .sim_parser import *
from gym import spaces
from envs.gym_soccer.vss_soccer_sim_continuous import *
import random


class SimSoccerContinuousMultiAgentEnv(SimSoccerContinuousEnv):

    def __init__(self):
        super(SimSoccerContinuousMultiAgentEnv, self).__init__()
        self.agent_id = 0
        self.soccer_env = self
        self.action_format = '6f'
        self.prev_min_ball_dist = None

    def step(self, commands):
        self._set_action(commands)

        while True:
            state = self._receive_state()
            if state is None:
                print(self.team_name, "received a 'None' state")
                return None, 0, True, {}

            if state.time < self.prev_time + self.env_start_time:  # a posterior reset (or crash occurred)
                self.env_start_time = state.time
                self.prev_time = 0
                print(self.team_name, "a posterior reset (or crash) occurred")
                return None, 0, True, {}

            state.time = state.time - self.env_start_time

            if state.time > self.prev_time:
                break

        #print(self.team_name, state.time, "/", self.time_limit)

        dt = state.time - self.prev_time
        #print("dt:%d, time: %d, prev:%d, start:%d" % (dt, state.time, self.prev_time, self.env_start_time))
        self.prev_time = state.time
        self.last_state, reward, done = self._parse_state(state, dt)

        # self.send_debug()
        return self.last_state, reward, done, {}

    def init_space_action(self):

        self.last_state, reward, done = self._parse_state(self._receive_state(), self.cmd_wait)
        shape = len(self.last_state)
        self.state_format = "%df" % shape
        self.action_format = '6f'

        self.observation_space = spaces.Box(low=-200, high=200, dtype=np.float32, shape=(shape,))
        self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(6,))

        return self.observation_space, self.action_space

    def _set_action(self, commands):

        rbt_i = 0
        for i in range(0, len(commands)-1, 2):

            rbt = self.team[rbt_i]

            rbt.write_log([self.ball_x, self.ball_y])

            rbt.update_targets()

            if rbt_i < self.parameters['ctrl_agents']:  # robots being controlled
                # continuous control
                rbt.angular_speed_desired = commands[i] * self.range_angular
                rbt.linear_speed_desired = commands[i + 1] * self.range_linear
            else:
                # random walk
                rbt.angular_speed_desired = random.uniform(-1.0, 1.0) * self.range_angular
                rbt.linear_speed_desired = random.uniform(-1.0, 1.0) * self.range_linear

                # calculate wheels' linear speeds:
            rbt.left_wheel_speed = clip(rbt.linear_speed_desired - self.robot_l * rbt.angular_speed_desired, -100, 100)
            rbt.right_wheel_speed = clip(rbt.linear_speed_desired + self.robot_l * rbt.angular_speed_desired, -100, 100)

            # update desired target x and y (used in parse state):
            rbt.target_rho = rbt.linear_speed_desired / 1.5
            rbt.target_theta = to_pi_range(rbt.theta + rbt.angular_speed_desired / -7.5)
            rbt.target_x = rbt.x + rbt.target_rho * math.cos(rbt.target_theta)
            rbt.target_y = rbt.y + rbt.target_rho * math.sin(rbt.target_theta)

            # Assumes energy consumption is proportional to wheel speeds:
            rbt.energy = pow(rbt.left_wheel_speed, 2) + pow(rbt.right_wheel_speed, 2)
            rbt_i += 1

        self.sim.send_speeds(self.team)

    def reset_game_vars(self):
        self.prev_min_ball_dist = None
        super(SimSoccerContinuousMultiAgentEnv, self).reset_game_vars()

    # def _calculate_rewards(self, goal_score, my_team, dt):
    #     ball_potential = self._ball_potential_penalty()  # (-1,0)
    #
    #     if self.prev_ball_potential is not None:
    #         grad_ball_potential = clip((ball_potential - self.prev_ball_potential) * 4000 / dt, -1.0,
    #                                    1.0)  # (-1,1)
    #     else:
    #         grad_ball_potential = 0
    #
    #     if goal_score != 0:  # A goal has happened
    #         reward = goal_score * self.w_goal
    #
    #         self.prev_ball_potential = None
    #         self.prev_min_ball_dist = None
    #         for i in range(0, len(self.team)):
    #             self.team[i].prev_ball_dist = None
    #
    #         # record rewards and score
    #         self.rw_goal += reward
    #         self.goal_score += goal_score
    #         if goal_score > 0:
    #             self.my_goals += 1
    #         else:
    #             self.opponent_goals += 1
    #
    #         for i in range(0, len(self.team)):
    #             self.team[i].rw_total += reward
    #
    #         print("******************GOAL****************")
    #         print("GOAL in %d steps Reward:%.4f. %d x %d" % (
    #             self.steps, reward, self.my_goals, self.opponent_goals))
    #
    #     else:
    #         move_reward = 0
    #         energy_penalty = 0
    #         collisions = sum(-8.0 / self._teammate_dist(my_team))
    #
    #         avg_ball_dist = 0
    #         for i in range(0, len(self.team)):
    #             avg_ball_dist += self.team[i].ball_dist
    #             energy_penalty -= self.team[i].energy
    #
    #         avg_ball_dist /= 3.0
    #         energy_penalty /= 3.0
    #
    #         if self.prev_min_ball_dist is not None:
    #             move_reward = clip((self.prev_min_ball_dist - avg_ball_dist) * 50 / dt, -1.0, 1.0)  # (-1,1)
    #
    #         reward = self.w_move * move_reward + \
    #                  self.w_ball_grad * grad_ball_potential + \
    #                  self.w_collision * collisions + \
    #                  self.w_ball_pot * ball_potential + \
    #                  self.w_energy * energy_penalty
    #
    #         # record ball_grad reward
    #         self.rw_ball_grad += self.w_ball_grad * grad_ball_potential
    #
    #         # record agent specific rewards
    #         for i in range(0, len(self.team)):
    #             self.team[i].rw_move += self.w_move * move_reward
    #             self.team[i].rw_collision += self.w_collision * collisions
    #             self.team[i].rw_energy += self.w_energy * energy_penalty
    #             self.team[i].rw_total += reward
    #
    #         self.prev_min_ball_dist = avg_ball_dist
    #
    #     self.prev_ball_potential = ball_potential
    #     return reward

    def _calculate_rewards(self, goal_score, my_team, dt):
        ball_potential = self._ball_potential_penalty()  # (-1,0)

        if self.prev_ball_potential is not None:
            grad_ball_potential = clip((ball_potential - self.prev_ball_potential) * 4000 / dt, -1.0,
                                       1.0)  # (-1,1)
        else:
            grad_ball_potential = 0

        if goal_score != 0:  # A goal has happened
            rewards = np.full(len(self.team), goal_score * self.w_goal)

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
            base_striker = 160.0
            base_defender = 85.0
            base_keeper = 10.0

            resp_striker = (160.0-abs(self.ball_x-base_striker))/160.0
            resp_defender = (160.0-abs(self.ball_x-base_defender))/160.0
            resp_keeper = (160.0-abs(self.ball_x-base_keeper))/160.0
            responsibility = [resp_striker, resp_defender, resp_keeper]

            move_reward = np.full(len(self.team), 0.0)
            energy_penalty = np.full(len(self.team), 0.0)
            collisions = -8.0 / self._teammate_dist(my_team)

            for i in range(0, len(self.team)):
                energy_penalty[i] = -self.team[i].energy
                if self.team[i].prev_ball_dist is not None and self.team[i].ball_dist > self.min_dist_move_rw:
                    move_reward[i] = clip((self.team[i].prev_ball_dist - self.team[i].ball_dist) * 50 / dt, -1.0,
                                          1.0)  # (-1,1)

            rewards = np.multiply(responsibility,
                                  self.w_move * move_reward +
                                  self.w_ball_grad * grad_ball_potential) + \
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

            #if self.is_team_yellow:
                #print("ball_x:%.2f, r_striker: %.2f, r_defender: %.2f, r_keep:: %.2f" % (self.ball_x, resp_striker, resp_defender, resp_keeper))

        self.prev_ball_potential = ball_potential

        return sum(rewards)

    def _parse_state(self, state, dt):

        if not self.is_team_yellow:
            state = transpose_state(state)

        for ball in state.balls:
            # real values
            ball_state = (normX(ball.pose.x), normX(ball.pose.y),
                          normVx(ball.v_pose.x), normVx(ball.v_pose.y))

            self.ball_x = ball.pose.x
            self.ball_y = ball.pose.y

        ball_np = np.array((self.ball_x - 4.0, self.ball_y))  # where the ball is now (a little behind it)

        if self.is_team_yellow:
            my_team = state.robots_yellow
            opponent_team = state.robots_blue
        else:
            my_team = state.robots_blue
            opponent_team = state.robots_yellow

        t1_state = ()
        for idx, t1_robot in enumerate(my_team):
            self.team[idx].fill(t1_robot, ball_np)

            # build the state. Encode yaw as sin(yaw) and cos(yaw)
            t1_state += (normX(self.team[idx].target_x), normX(self.team[idx].target_y), normX(t1_robot.pose.x),
                         normX(t1_robot.pose.y), math.sin(t1_robot.pose.yaw), math.cos(t1_robot.pose.yaw),
                         normVx(t1_robot.v_pose.x), normVx(t1_robot.v_pose.y),
                         normVt(t1_robot.v_pose.yaw))

        t2_state = ()
        for t2_robot in opponent_team:
            t2_state += (normX(t2_robot.pose.x), normX(t2_robot.pose.y), math.sin(t2_robot.pose.yaw),
                         math.cos(t2_robot.pose.yaw),
                         normVx(t2_robot.v_pose.x), normVx(t2_robot.v_pose.y),
                         normVt(t2_robot.v_pose.yaw))  # estimated values

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
            time_norm = max(1.0 - (state.time / self.time_limit), 0.0)

        # pack next state:
        env_states = (time_norm,) + ball_state + t1_state + t2_state

        self.steps += 1

        if done:
            self.print_summary()
            self._decay_reward_shaping()

        return env_states, rewards, done
