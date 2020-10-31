from .utils import *
import numpy as np

import torch
import os
import datetime
import random
import joblib

from .utils import *
from gym_vss.spline import SplineRandomizer
from gym_vss.ctrl.ctrl_model import ModelFF, ModelLSTM



class VSSPlayer:
    def __init__(self, id):
        base_striker = 140.0  # 160.0
        base_defender = 70  # 85.0
        base_keeper = 14.0
        self.base = [base_striker, base_defender, base_keeper]
        self.alpha_base = 0.5

        self.id = id
        self.x = 0
        self.y = 0
        self.theta = 0
        self.target_x = None
        self.target_y = None
        self.target_theta = None
        self.target_rho = 0
        self.robot_l = 4  # robot wheel-center distance
        self.robot_r = 2  # robot wheel radius

        self.linear_speed_obs = 0.0
        self.angular_speed_obs = 0.0
        self.linear_speed_desired = 0.0
        self.angular_speed_desired = 0.0

        self.prev_ball_dist = None
        self.ball_dist = None
        self.base_dist = None
        self.prev_base_dist = None

        self.np_last_spds_measured = np.array([0, 0])
        self.np_spds_desired = np.array([0, 0])
        self.np_spds_measured = np.array([0, 0])
        self.np_wheel_applied = np.array([0, 0])
        self.np_pos = np.array([0, 0, 0])
        self.np_target = np.array([0, 0])

        self.ctrl_model = None
        self.ctrl_log = None

        # -- Randomizer
        self.randomizer_left = SplineRandomizer(-100, 100, 30)
        self.randomizer_right = SplineRandomizer(-100, 100, 30)

        self.is_first_iteration = True

        if (ctrl_mode & CTRL_CORRECT) and self.id == 0:
            #self.load_ctrl_model('ctrl/models/ff_log_full_4.cpth')
            self.load_ctrl_model(dict_ctrl_model_id[self.id])

        if (ctrl_mode & CTRL_COLLECT_LOG) and self.id == 0:
            self.ctrl_log = open("ctrl/ctrl_log" + str(self.id)
                                 + datetime.datetime.now().strftime('_%Y-%m-%d_%H:%M:%S') + ".csv", "a")

        # self.speed_log = open("speed.csv", "a")
        # reward logs
        self.track_rw = True
        self.rw_move = 0
        self.rw_collision = 0
        self.rw_energy = 0
        self.rw_total = 0

        self.left_wheel_speed = 0
        self.right_wheel_speed = 0
        self.energy = 0

        self.custom_velos_lin = []
        self.custom_velos_ang = []
        self.custom_velo_index = 0

        for step in range(4, 32, 4):
            self.create_custom_veloc(steps=step)


    def load_ctrl_model(self, path):
        if os.path.isfile(path):
            print("Loading control model:" + path + " for agent: ", self.id)
            #self.ctrl_model = ModelFF(2, 4, 2)
            self.ctrl_model = ModelFF(ctrl_model_input_size, ctrl_model_input_size*2, ctrl_model_output_size)
            print(self.ctrl_model)
            device = torch.device('cpu')
            self.ctrl_model.load_state_dict(torch.load(path, map_location=device))

        else:
            print("ERROR: control file not found: " + path)

    def load_ctrl_model_lstm(self, path):
        if os.path.isfile(path):
            print("Loading control model:" + path)
            self.ctrl_model = ModelLSTM(2, 4, 2)
            device = torch.device('cpu')
            self.ctrl_model.load_state_dict(torch.load(path, map_location=device))
        else:
            print("ERROR: control file not found: " + path)

    def load_ctrl_model_sklearn(self, path):
        if os.path.isfile(path):
            print("Loading control model:" + path)
            self.ctrl_model = joblib.load(path)
        else:
            print("ERROR: control file not found: " + path)

    def reset_rewards(self):
        self.rw_move = 0
        self.rw_collision = 0
        self.rw_energy = 0
        self.rw_total = 0

    def fill(self, robot_state, ball):
        # update agent variables:
        self.x = robot_state.pose.x
        self.y = robot_state.pose.y
        self.theta = robot_state.pose.yaw

        # initialize agent target variables if needed
        if self.target_x is None:
            self.target_x = self.x
            self.target_y = self.y
            self.target_rho = 0
            self.target_theta = self.theta

        self.linear_speed_obs = mod(robot_state.v_pose.x, robot_state.v_pose.y)

        if abs(angle(robot_state.v_pose.x, robot_state.v_pose.y) - self.theta) > math.pi / 2.:
            self.linear_speed_obs = - self.linear_speed_obs

        self.angular_speed_obs = robot_state.v_pose.yaw

        self.prev_ball_dist = self.ball_dist

        # target_y = 0.7*ball[1]+0.3*65  # 1/3 y distance between ball_y and the center of the goal
        # self.ball_dist = abs(target_y - self.y)  # distance to base position in x
        self.ball_dist = np.linalg.norm(ball - np.array([self.x, self.y]))  # distance to current ball position

        self.prev_base_dist = self.base_dist

        self.base_dist = abs(self.base[self.id] - self.x)  # distance to base position in x
        # self.base_dist = np.linalg.norm(np.array([self.base[self.id], 65]) - np.array([self.x, self.y]))

    def update_targets(self):
        # self.speed_log.write("%f, %f\n" % (self.linear_speed_obs, self.angular_speed_obs))
        # self.speed_log.flush()
        # calculate next (rho,theta) target theta for all agents based on their current (x,y) target:
        self.target_rho = mod(self.target_x - self.x, self.target_y - self.y)

        if abs(self.target_rho) > 0.01:
            self.target_theta = math.atan2((self.target_y - self.y), (self.target_x - self.x))
        else:
            self.target_theta = self.theta

        if abs(smallest_angle_diff(self.target_theta, self.theta)) > math.pi / 2:
            self.target_rho = -self.target_rho
            self.target_theta = to_pi_range(self.target_theta + math.pi)

    def update_wheel_sdps_random(self):
        self.left_wheel_speed = random.uniform(-100, 100)
        self.right_wheel_speed = random.uniform(-100, 100)

        self.angular_speed_desired = (self.right_wheel_speed - self.left_wheel_speed) / (self.robot_l * 2.0)
        self.linear_speed_desired = (self.right_wheel_speed + self.left_wheel_speed) / 2.0

    def update_wheel_sdps_spline(self):
        self.left_wheel_speed = self.randomizer_left.get_next()
        self.right_wheel_speed = self.randomizer_right.get_next()

        self.angular_speed_desired = (self.right_wheel_speed - self.left_wheel_speed) / (self.robot_l * 2.0)
        self.linear_speed_desired = (self.right_wheel_speed + self.left_wheel_speed) / 2.0

    def update_wheel_sdps_analytic(self):
        self.left_wheel_speed = clip(self.linear_speed_desired - self.robot_l * self.angular_speed_desired, -100, 100)
        self.right_wheel_speed = clip(self.linear_speed_desired + self.robot_l * self.angular_speed_desired, -100, 100)
        print("lin:%.2f ang:%.2f vl:%.2f vr:%.2f " % (self.linear_speed_desired, self.angular_speed_desired, self.left_wheel_speed, self.right_wheel_speed))

    def update_wheel_sdps_model(self):

        #in_ctrl = torch.from_numpy(np.array([self.linear_speed_desired / 90.0, self.angular_speed_desired / 30.0],
        #                                    dtype=np.float32))
        in_ctrl = torch.from_numpy(np.array([self.linear_speed_obs / 90.0, self.angular_speed_obs / 30.0, self.linear_speed_desired / 90.0, self.angular_speed_desired / 30.0],
                                            dtype=np.float32))
        out_ctrl = self.ctrl_model(in_ctrl)
        self.left_wheel_speed = 100 * out_ctrl[0].item()
        self.right_wheel_speed = 100 * out_ctrl[1].item()

    def update_wheel_spds_sklearn(self):
        out_ctrl = self.ctrl_model.predict([[self.linear_speed_desired / 90.0, self.angular_speed_desired / 30.0]])
        self.left_wheel_speed = 100 * out_ctrl[0][0].item()
        self.right_wheel_speed = 100 * out_ctrl[0][1].item()

    def create_custom_veloc(self, steps = 20):
        ang_range = 15.0
        lin_range = 95.0
        repeat = 4

        for ang in np.linspace(0, 2 * math.pi, steps):
            for i in range(0, repeat):
                self.custom_velos_ang.append(ang_range * math.sin(ang*2)/8)#(ang_range * math.sin(ang)/10) / (self.robot_l*2.0))
                self.custom_velos_lin.append(2*lin_range * math.sin(ang) / (2.0))

        for ang in np.linspace(0, 2 * math.pi, steps):
            for i in range(0, repeat):
                self.custom_velos_ang.append(ang_range * math.sin(ang))
                self.custom_velos_lin.append(0.0)

        ang_range = 100.0
        for ang in np.linspace(0, 2 * math.pi, steps):
            for i in range(0, repeat):
                self.custom_velos_ang.append((ang_range * math.sin(ang) - 0.0) / (self.robot_l*2.0))
                self.custom_velos_lin.append((ang_range * math.sin(ang) + 0.0) / (2.0))

        for ang in np.linspace(0, 2 * math.pi, steps):
            for i in range(0, repeat):
                self.custom_velos_ang.append((0.0 - ang_range * math.sin(ang)) / (self.robot_l*2.0))
                self.custom_velos_lin.append((0.0 + ang_range * math.sin(ang)) / (2.0))


    def update_wheel_sdps_custom(self):
        self.angular_speed_desired  = self.custom_velos_ang[self.custom_velo_index]
        self.linear_speed_desired  = self.custom_velos_lin[self.custom_velo_index]

        self.custom_velo_index = self.custom_velo_index + 1
        if(self.custom_velo_index >= len(self.custom_velos_lin)):
            self.custom_velo_index = 0

        self.left_wheel_speed = self.linear_speed_desired - self.robot_l * self.angular_speed_desired
        self.right_wheel_speed = self.linear_speed_desired + self.robot_l * self.angular_speed_desired

    def valid_point(self):
        return (self.x > 20 and self.x < 150 and self.y > 8 and self.y < 122) # check border distance

    def write_log(self, ball_pos):

        if (ctrl_mode & CTRL_COLLECT_LOG) and self.id == 0 and not self.is_first_iteration:

            self.np_last_spds_measured = self.np_spds_measured
            self.np_spds_measured = np.array([self.linear_speed_obs / 90.0, self.angular_speed_obs / 30.0])
            self.np_spds_desired = np.array([self.linear_speed_desired / 90.0, self.angular_speed_desired / 30.0])
            self.np_wheel_applied = np.array([self.left_wheel_speed / 100.0, self.right_wheel_speed / 100.0])
            self.np_pos = np.array([self.x, self.y, self.theta])
            self.np_target = np.array([self.target_x, self.target_y])
            np_ball_pos = np.array(ball_pos)
            #if self.valid_point():
            self.ctrl_log.write(
                "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (
                    self.np_last_spds_measured[0],self.np_last_spds_measured[1],
                    self.np_spds_measured[0], self.np_spds_measured[1],
                    self.np_wheel_applied[0], self.np_wheel_applied[1],
                    self.np_spds_desired[0], self.np_spds_desired[1],
                    self.np_pos[0], self.np_pos[1], self.np_pos[2],
                    self.np_target[0], self.np_target[1],
                    np_ball_pos[0], np_ball_pos[1]))
            #     print('(%d, %d): valid!' % (self.x, self.y))
            # else:
            #     print('(%d, %d): invalid!'% (self.x, self.y))

            # print("Desired Lin: %.4f Desired Ang: %.4f" % (self.np_spds_desired[0]*90, self.np_spds_desired[1]*30))
            # print("Right wheel: %.4f Left wheel : %.4f" % (self.np_wheel_applied[1]*100, self.np_wheel_applied[0]*100))

                #self.linear_speed_desired
                #self.angular_speed_desired

                #vl = self.linear_speed_desired - self.robot_l * self.angular_speed_desired
                #vr = self.linear_speed_desired + self.robot_l * self.angular_speed_desired
                #print("LG %f,%f,%f,%f\n" % (self.linear_speed_desired, self.angular_speed_desired,self.left_wheel_speed,self.right_wheel_speed))

        self.is_first_iteration = False

    # def __del__(self):
    #     self.speed_log.close()
    #     if self.ctrl_log:
    #         self.ctrl_log.close()
    #         print("log file closed.")
