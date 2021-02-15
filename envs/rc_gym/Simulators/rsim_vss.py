from typing import Dict, List

import numpy as np
import robosim
from rc_gym.Entities import FrameVSS
from rc_gym.Simulators.rsim_base import RSim


class RSimVSS(RSim):
    def __init__(
        self,
        field_type: int,
        n_robots_blue: int,
        n_robots_yellow: int,
        time_step_ms: int,
    ):
        super().__init__(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=time_step_ms,
        )

        self.linear_speed_range = 1.15  # m/s
        self.robot_wheel_radius = 0.026

    def send_commands(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64
        )

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            # convert from linear speed to angular speed
            sim_commands[rbt_id][0] = cmd.v_wheel1 / self.robot_wheel_radius
            sim_commands[rbt_id][1] = cmd.v_wheel2 / self.robot_wheel_radius
        self.simulator.step(sim_commands)

    def get_frame(self) -> FrameVSS:
        state = self.simulator.get_state()
        # Update frame with new state
        frame = FrameVSS()
        frame.parse(state, self.n_robots_blue, self.n_robots_yellow)

        return frame

    def _init_simulator(
        self,
        field_type,
        n_robots_blue,
        n_robots_yellow,
        ball_pos,
        blue_robots_pos,
        yellow_robots_pos,
        time_step_ms,
    ):

        return robosim.SimulatorVSS(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            ball_pos=ball_pos,
            blue_robots_pos=blue_robots_pos,
            yellow_robots_pos=yellow_robots_pos,
            time_step_ms=time_step_ms,
        )
