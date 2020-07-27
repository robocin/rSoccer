import numpy as np
from .proto_state_models import Pose


class KalmanFilter2D:
    def __init__(self, accel_noise_mag=0.1, u=0.005, pose=None, dt=None):
        self.u = u
        self.Q_estimate = []
        self.accel_noise_mag = accel_noise_mag
        self.tkn_x = None
        self.tkn_y = None
        self.Ez = []
        self.Ex = []
        self.A = []
        self.B = []
        self.P = []
        self.C = []

        if pose is not None:
            self.set_kalman(pose)

        if dt is not None:
            self.update_dt_matrices(dt)

    def set_kalman(self, pose):
        self.Q_estimate = np.array([[pose.x],
                                    [pose.y],
                                    [0],
                                    [0]])

        self.tkn_x = 1
        self.tkn_y = 1

        self.Ez = np.array([[self.tkn_x, 0],
                            [0, self.tkn_y]])

        self.Ex = []
        self.A = []
        self.B = []
        self.P = None

        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

    def update_dt_matrices(self, dt):

        self.Ex = np.array([[(dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                            [0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                            [(dt ** 3) / 2, 0, (dt ** 2), 0],
                            [0, (dt ** 3) / 2, 0, (dt ** 2)]])
        self.Ex *= (self.accel_noise_mag ** 2)

        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.B = np.array([[(dt ** 2) / 2],
                           [(dt ** 2) / 2],
                           [dt],
                           [dt]])

        if self.P is None:
            self.P = self.Ex

    def predict(self, dt=None):
        if dt is not None:
            self.update_dt_matrices(dt)

        # Predict
        self.Q_estimate = np.dot(self.A, self.Q_estimate) + self.B * self.u
        self.P = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.Ex

    def predict_update(self, pose, dt=None):
        if dt is not None:
            self.update_dt_matrices(dt)

        pose_kf = np.array([[pose.x],
                            [pose.y]])
        # Predict
        self.Q_estimate = np.dot(self.A, self.Q_estimate) + self.B * self.u

        self.P = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.Ex
        # print(self.P)

        # Update
        K = np.dot(np.dot(self.P, self.C.transpose()),
                   np.linalg.inv(np.dot(np.dot(self.C, self.P), self.C.transpose()) + self.Ez))
        # print(K)
        self.Q_estimate = self.Q_estimate + np.dot(K, (pose_kf - np.dot(self.C, self.Q_estimate)))
        # print(self.Q_estimate)
        self.P = np.dot((np.eye(4, 4) - np.dot(K, self.C)), self.P)
        # print(self.P)

        final_estimate = self.Q_estimate.reshape(1, -1).squeeze()  # x, y, vx, vy

        # return pose, v_pose
        return Pose(x=final_estimate[0], y=final_estimate[1]), Pose(final_estimate[2], final_estimate[3])
