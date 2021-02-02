import robosim
import numpy as np
from typing import Dict, List
from rc_gym.Entities import FrameSSL
from rc_gym.Simulators.rsim_base import RSim


class RSimSSL(RSim):
    def __init__(self, field_type: int, n_robots_blue: int,
                 n_robots_yellow: int, time_step_ms: int):
        super().__init__(field_type=field_type, n_robots_blue=n_robots_blue,
                         n_robots_yellow=n_robots_yellow, time_step_ms=time_step_ms)

        self.linear_speed_range = 1.15  # m/s

    def send_commands(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 6), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            # convert from linear speed to angular speed
            sim_commands[rbt_id][0] = cmd.v_x
            sim_commands[rbt_id][1] = cmd.v_y
            sim_commands[rbt_id][2] = cmd.v_theta
            sim_commands[rbt_id][3] = cmd.kick_v_x
            sim_commands[rbt_id][4] = cmd.kick_v_z
            sim_commands[rbt_id][5] = cmd.dribbler
            
        self.simulator.step(sim_commands)

    def get_frame(self) -> FrameSSL:
        state = self.simulator.get_state()
        # Update frame with new state
        frame = FrameSSL()
        frame.parse(state, self.n_robots_blue, self.n_robots_yellow)

        return frame

    def _init_simulator(self, field_type, n_robots_blue, n_robots_yellow,
                        ball_pos, blue_robots_pos, yellow_robots_pos,
                        time_step_ms):

        return robosim.SimulatorSSL(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            ball_pos=ball_pos,
            blue_robots_pos=blue_robots_pos,
            yellow_robots_pos=yellow_robots_pos,
            time_step_ms=time_step_ms
            )
