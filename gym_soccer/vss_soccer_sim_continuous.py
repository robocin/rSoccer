from .sim_parser import *
from envs.gym_soccer import SimSoccerEnv
from envs.utils import *
import random


class SimSoccerContinuousEnv(SimSoccerEnv):
    barrier = None

    def __init__(self):
        super(SimSoccerContinuousEnv, self).__init__()

        # -- Simulator parameters
        self.range_linear = 90.
        self.range_angular = 30.

    def set_parameters(self, parameters):
        super(SimSoccerContinuousEnv, self).set_parameters(parameters)

        self.range_linear = parameters['range_linear']
        self.range_angular = parameters['range_angular']

    def process_input(self):
        pass
        
    # Env methods
    # ----------------------------

    # receive state with fake gaussian noise for training networks
    def _receive_state(self):
        state = super(SimSoccerContinuousEnv, self)._receive_state()
        sigma_pos = 0.5
        sigma_ang = math.atan2(sigma_pos, 2*self.robot_l/3)  # angle sift given blob detection error

        # # fake noise
        # for i in range(0, 3):
        #     state.robots_yellow[i].pose.x += random.gauss(0, sigma_pos)
        #     state.robots_yellow[i].pose.y += random.gauss(0, sigma_pos)
        #     state.robots_yellow[i].pose.yaw += random.gauss(0, sigma_ang)

        #     state.robots_blue[i].pose.x += random.gauss(0, sigma_pos)
        #     state.robots_blue[i].pose.y += random.gauss(0, sigma_pos)
        #     state.robots_blue[i].pose.yaw += random.gauss(0, sigma_ang)

        # state.balls[0].pose.x += random.gauss(0, sigma_pos)
        # state.balls[0].pose.y += random.gauss(0, sigma_pos)

        return state

    def _set_action(self, commands):

        for i in range(0, len(commands)):

            rbt = self.team[i]

            rbt.write_log([self.ball_x, self.ball_y])

            rbt.update_targets()

            rbt.angular_speed_desired = commands[i][0].item() * self.range_angular
            rbt.linear_speed_desired = commands[i][1].item() * self.range_linear

            # command_angular_speed_desired = commands[i][0].item() * self.range_angular
            # command_linear_speed_desired = commands[i][1].item() * self.range_linear
            #
            # # continuous control
            # # moving average
            # if rbt.linear_speed_desired is None:
            #     rbt.angular_speed_desired = command_angular_speed_desired
            #     rbt.linear_speed_desired = command_linear_speed_desired
            # else:
            #     percent = 0.5
            #     rbt.angular_speed_desired = percent * command_angular_speed_desired + (1 - percent) * rbt.angular_speed_desired
            #     rbt.linear_speed_desired = percent * command_linear_speed_desired + (1 - percent) * rbt.linear_speed_desired


            # # Overhide speeds:
            # rbt.angular_speed_desired = 0#-15.0#-5.0
            # rbt.linear_speed_desired = 20#-30.0#-60.0s

            # update desired target x and y (used in parse state):
            rbt.target_rho = rbt.linear_speed_desired / 1.5
            rbt.target_theta = to_pi_range(rbt.theta + rbt.angular_speed_desired / -7.5)
            rbt.target_x = rbt.x + rbt.target_rho * math.cos(rbt.target_theta)
            rbt.target_y = rbt.y + rbt.target_rho * math.sin(rbt.target_theta)

            # calculate wheels' linear speeds:
            rbt.left_wheel_speed = (rbt.linear_speed_desired - self.robot_l * rbt.angular_speed_desired)
            rbt.right_wheel_speed = (rbt.linear_speed_desired + self.robot_l * rbt.angular_speed_desired)

            # # Overhide speeds:
            # rbt.left_wheel_speed = 50
            # rbt.right_wheel_speed = -50

            # For run ctrl model
            if i == 0 and (ctrl_mode & CTRL_CORRECT):
                rbt.update_wheel_sdps_model()
            elif (ctrl_mode & CTRL_RANDOM_WALK):
                rbt.update_wheel_sdps_random()
            elif (ctrl_mode & CTRL_SPLINE):
                rbt.update_wheel_sdps_spline()
            #else:
                #rbt.update_wheel_sdps_analytic()

            # if i == 0:
            #     calc_lin = (rbt.right_wheel_speed + rbt.left_wheel_speed)/2.0
            #     calc_ang = (rbt.right_wheel_speed - rbt.left_wheel_speed) / (2.0*self.robot_l)
            #     print("\nDesired Lin: %.4f Desired Ang: %.4f" % (rbt.linear_speed_desired, rbt.angular_speed_desired))
            #     print("Right wheel: %.4f Left wheel : %.4f" % (rbt.right_wheel_speed, rbt.left_wheel_speed))
            #     print("Recalcu Lin: %.4f Recalcu Ang: %.4f" % (calc_lin, calc_ang))

            # Assumes energy consumption is proportional to wheel speeds:
            rbt.energy = abs(rbt.left_wheel_speed) + abs(rbt.right_wheel_speed)

        self._send_wheel_speeds(self.team)

    def _send_wheel_speeds(self, team):

        pulse_speed_ratio = self.robot_r

        for rbt in team:

            # cut off very low speeds
            if -15 < rbt.left_wheel_speed < 15:
                rbt.left_wheel_speed = 0

            if -15 < rbt.right_wheel_speed < 15:
                rbt.right_wheel_speed = 0

            clip_range = 100 #384.615384615385  # this value matches the effects of clip100 in real robots
            # print("vdes:", pulse_speed_ratio *rbt.left_wheel_speed, pulse_speed_ratio *rbt.right_wheel_speed)
            rbt.left_wheel_speed = clip(int(round(pulse_speed_ratio * rbt.left_wheel_speed)), -clip_range, clip_range)
            rbt.right_wheel_speed = clip(int(round(pulse_speed_ratio * rbt.right_wheel_speed)), -clip_range, clip_range)
            # print("clip:", rbt.left_wheel_speed, rbt.right_wheel_speed)


        self.sim.send_speeds(self.team)

