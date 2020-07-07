from gym_vss.gym_real_soccer.kalman_filter_2d import KalmanFilter2D
import threading
import time
from .proto_state_models import *

class KalmanParser:
    def __init__(self, accel_noise_mag=0.1, u=0.005, vision_parser=None):
        self.vision_parser = vision_parser
        self.filter_yellow_robots = [KalmanFilter2D(accel_noise_mag, u) for _ in range(3)]
        self.filter_blue_robots = [KalmanFilter2D(accel_noise_mag, u) for _ in range(3)]
        self.filter_ball = KalmanFilter2D(accel_noise_mag, u)
        self.time = None
        self.update_time = None
        self.predict_time = None
        self.state = self.prev_state = None

    def predict_update_state(self, state):
        now = time.time()

        if self.state is None:
            self.state = ProtoState(state)

        if self.time is not None:
            dt = state.time - self.time
            if self.predict_time is not None:
                predict_dt = now - self.predict_time
                if predict_dt < dt:
                    dt = predict_dt

                self.predict_time = now

            if dt > 0:
                for i in range(0, 3):
                    self.state.robots_yellow[i].pose, self.state.robots_yellow[i].v_pose = self.filter_yellow_robots[i].predict_update(state.robots_yellow[i].pose, dt)

                for i in range(0, 3):
                    self.state.robots_blue[i].pose, self.state.robots_blue[i].v_pose = self.filter_blue_robots[i].predict_update(state.robots_blue[i].pose, dt)

                self.state.balls[0].pose, self.state.balls[0].v_pose = self.filter_ball.predict_update(state.balls[0].pose, dt)

        else:  # first iteration, just set kalman pose
            for i in range(0, 3):
                self.filter_yellow_robots[i].set_kalman(state.robots_yellow[i].pose)

            for i in range(0, 3):
                self.filter_blue_robots[i].set_kalman(state.robots_blue[i].pose)

            self.filter_ball.set_kalman(state.balls[0].pose)

        self.time = state.time
        self.update_time = now
        return self.state

    def predict_state(self, state):
        now = time.time()

        if self.predict_time is not None:
            dt = now - self.predict_time

            for i in range(0, 3):
                self.state.robots_yellow[i].pose, self.state.robots_yellow[i].v_pose = self.filter_yellow_robots[i].predict(dt)

            for i in range(0, 3):
                self.state.robots_blue[i].pose, self.state.robots_blue[i].v_pose = self.filter_blue_robots[i].predict(dt)

            self.state.balls[0].pose, self.state.balls[0].v_pose = self.filter_ball.predict(dt)

        self.predict_time = now
        return state

    def reset(self):
        self.vision_parser.reset()

    def run_receive(self):
        data = self.conn.receive()
        state = self.vision_parser.vss_to_state_pkt(data, debug=False)
        state = self.predict_update_state(state)
        self.vision_parser.state = state
        return state

    def receive(self):
        if self.time is None:  # at the first time must wait for an observation
            state = self.run_receive()
            self.thread = threading.Thread(target=self.run_receive)
            self.thread.start()
            self.prev_state = state
        else:
            state = self.predict_state(self.prev_state)
            self.prev_state = state

        return state

    def __del__(self):
        self.thread.join()
