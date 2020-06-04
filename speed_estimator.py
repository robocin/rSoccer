from envs.utils import *
import math


class SpeedEstimator:
    def __init__(self, beta, acc_th=100, id=0):
        self.acc_th = acc_th*1000
        self.id = id
        self.prev_pos = [85, 65, 0]
        self.prev_time = None
        self.beta = beta
        self.moving_avg_x = 0
        self.corr_moving_avg_x = 0
        self.moving_avg_y = 0
        self.corr_moving_avg_y = 0
        self.moving_avg_yaw = 0
        self.corr_moving_avg_yaw = 0
        self.count_avg = 0
        self.prev_lin = None
        self.prev_vel_x = 0
        self.prev_vel_y = 0
        self.prev_vel_yaw = 0

    def reset(self):
        self.prev_pos = [85, 65, 0]
        self.prev_time = None
        self.prev_vel_x = 0
        self.prev_vel_y = 0
        self.prev_vel_yaw = 0
        self.prev_lin = None
        self.moving_avg_x = 0
        self.corr_moving_avg_x = 0
        self.moving_avg_y = 0
        self.corr_moving_avg_y = 0
        self.moving_avg_yaw = 0
        self.corr_moving_avg_yaw = 0
        self.count_avg = 0

    def estimate_speed(self, pose, time, have_angle=True):

        time = time / 1000.
        scale = 1.
        if self.prev_time is None or (time - self.prev_time) == 0:
            self.prev_time = time
            self.prev_pos[0] = pose.x
            self.prev_pos[1] = pose.y
            if have_angle:
                self.prev_pos[2] = pose.yaw
                return 0.0, 0.0, 0.0
            else:
                return 0.0, 0.0

        else:
            dt = (time - self.prev_time)
            #print(1000*dt)
            self.count_avg += 1
            vel_x = (pose.x - self.prev_pos[0]) / dt
            vel_y = (pose.y - self.prev_pos[1]) / dt
            if have_angle:
                vel_yaw = -smallest_angle_diff(pose.yaw, self.prev_pos[2]) / dt

            lin = math.sqrt(vel_x ** 2 + vel_y ** 2)
            if self.prev_lin is not None:
                if abs(lin-self.prev_lin)/dt > self.acc_th:
                    #print(self.id, ": abnormal acceleration! %.2f: %.2f->%.2f" % (abs(lin-self.prev_lin)/dt, self.prev_lin, lin))
                    lin = None
                    vel_x = self.prev_vel_x
                    vel_y = self.prev_vel_y
                    vel_yaw = self.prev_vel_yaw

            # if self.id == 0:
            #     print(vel_x, vel_y, vel_yaw, lin, self.prev_lin)

            self.moving_avg_x = self.beta * self.moving_avg_x + (1 - self.beta) * vel_x
            self.corr_moving_avg_x = self.moving_avg_x / (1 - self.beta ** self.count_avg)

            self.moving_avg_y = self.beta * self.moving_avg_y + (1 - self.beta) * vel_y
            self.corr_moving_avg_y = self.moving_avg_y / (1 - self.beta ** self.count_avg)

            if have_angle:
                self.moving_avg_yaw = self.beta * self.moving_avg_yaw + (1 - self.beta) * vel_yaw
                self.corr_moving_avg_yaw = self.moving_avg_yaw / (1 - self.beta ** self.count_avg)
                self.prev_pos = [pose.x, pose.y, pose.yaw]
                self.prev_vel_y = vel_yaw
            else:
                self.prev_pos = [pose.x, pose.y, 0]

            self.prev_time = time
            self.prev_lin = lin
            self.prev_vel_x = vel_x
            self.prev_vel_y = vel_y

            # if self.id == 0:
            #     print("v_dif:", abs(vel_x-self.corr_moving_avg_x), vel_x, self.corr_moving_avg_x)

            if have_angle:
                #return 0,0,0#1,1,1#0.0, 0.0, 0.0
                return scale*self.corr_moving_avg_x, scale*self.corr_moving_avg_y, scale*self.corr_moving_avg_yaw
            else:
                #return 0,0#1,1#0.0, 0.0
                return scale*self.corr_moving_avg_x, scale*self.corr_moving_avg_y
