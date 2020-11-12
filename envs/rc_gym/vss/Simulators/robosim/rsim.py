import robosim
import numpy as np
from typing import Dict, List
from rc_gym.Entities import Frame


class SimulatorVSS:
    def __init__(self, field_type: int,
                 n_robots_blue: int, n_robots_yellow: int):
        self.simulator = robosim.SimulatorVSS(field_type=field_type,
                                              n_robots_blue=n_robots_blue,
                                              n_robots_yellow=n_robots_yellow)
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.linear_speed_range = 1.5               #m/s
        self.angular_speed_range = np.deg2rad(360)  #rad/s
        self.robot_dist_center_to_wheel = 0.0425    #center to wheel + wheel thickness

    def reset(self):
        self.simulator.reset()

    def stop(self):
        del(self.simulator)

    def send_commands(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            sim_commands[rbt_id][0] = cmd.v_wheel1 * 100 / 2 # convert from m/s to cm/s and split by wheels
            sim_commands[rbt_id][1] = cmd.v_wheel2 * 100 / 2 # convert from m/s to cm/s and split by wheels

        self.simulator.step(sim_commands)

    def replace_from_frame(self, frame: Frame):
        replacement_pos: Dict[str, np.ndarray] = {}

        ball_pos: List[float] = [frame.ball.x,
                                 frame.ball.y, frame.ball.v_x, frame.ball.v_y]
        replacement_pos['ball'] = np.array(ball_pos)

        blue_pos: List[List[float]] = []
        for robot in frame.robots_blue.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            blue_pos.append(robot_pos)
        replacement_pos['blue_pos'] = np.array(blue_pos)

        yellow_pos: List[List[float]] = []
        for robot in frame.robots_yellow.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            yellow_pos.append(robot_pos)
        replacement_pos['yellow_pos'] = np.array(yellow_pos)

        self.simulator.replace_with_vel(**replacement_pos)

    def get_frame(self) -> Frame:
        state = self.simulator.get_state()
        status = self.simulator.get_status()
        # Update frame with new status and state
        frame = Frame()
        frame.parse(state, status, self.n_robots_blue,
                    self.n_robots_yellow)

        return frame

    def get_field_params(self):
        return self.simulator.get_field_params()
