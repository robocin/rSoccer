import robosim
import numpy as np
from typing import Dict, List
from rc_gym.Entities import Frame


class SimulatorVSS:
    def __init__(self, field_type: int, n_robots_blue: int,
                 n_robots_yellow: int):
        # Positions needed just to initialize the simulator
        ball_pos = [0, 0, 0, 0]
        blue_robots_pos = [[-0.2 * i, 0, 0]
                           for i in range(1, n_robots_blue + 1)]
        yellow_robots_pos = [[0.2 * i, 0, 0]
                             for i in range(1, n_robots_yellow + 1)]

        self.simulator = robosim.SimulatorVSS(field_type=field_type,
                                              n_robots_blue=n_robots_blue,
                                              n_robots_yellow=n_robots_yellow,
                                              ball_pos=ball_pos,
                                              blue_robots_pos=blue_robots_pos,
                                              yellow_robots_pos=yellow_robots_pos
                                              )
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.linear_speed_range = 1.5  # m/s
        self.angular_speed_range = np.deg2rad(360)  # rad/s
        # center to wheel + wheel thickness
        self.robot_dist_center_to_wheel = 0.0425

    def reset(self, frame: Frame):
        placement_pos = self._placement_dict_from_frame(frame)
        self.simulator.reset(**placement_pos)

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
            # convert from m/s to cm/s and split by wheels
            sim_commands[rbt_id][0] = cmd.v_wheel1 / 0.026
            # convert from m/s to cm/s and split by wheels
            sim_commands[rbt_id][1] = cmd.v_wheel2 / 0.026

        self.simulator.step(sim_commands)

    def _placement_dict_from_frame(self, frame: Frame):
        replacement_pos: Dict[str, np.ndarray] = {}

        ball_pos: List[float] = [frame.ball.x, frame.ball.y,
                                 frame.ball.v_x, frame.ball.v_y]
        replacement_pos['ball_pos'] = np.array(ball_pos)

        blue_pos: List[List[float]] = []
        for robot in frame.robots_blue.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            blue_pos.append(robot_pos)
        replacement_pos['blue_robots_pos'] = np.array(blue_pos)

        yellow_pos: List[List[float]] = []
        for robot in frame.robots_yellow.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            yellow_pos.append(robot_pos)
        replacement_pos['yellow_robots_pos'] = np.array(yellow_pos)

        return replacement_pos

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
