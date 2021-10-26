import numpy as np
import robosim

from typing import Dict, List
from rsoccer_gym.Entities import Frame, FrameVSS, FrameSSL, Field


class RSim:
    def __init__(
        self,
        field_type: int,
        n_robots_blue: int,
        n_robots_yellow: int,
        time_step_ms: int,
    ):
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow

        # Positions needed just to initialize the simulator
        ball_pos = [0, 0, 0, 0]
        blue_robots_pos = [[-0.2 * i, 0, 0]
                           for i in range(1, n_robots_blue + 1)]
        yellow_robots_pos = [[0.2 * i, 0, 0]
                             for i in range(1, n_robots_yellow + 1)]
        self.simulator = self._init_simulator(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            ball_pos=ball_pos,
            blue_robots_pos=blue_robots_pos,
            yellow_robots_pos=yellow_robots_pos,
            time_step_ms=time_step_ms
        )
        self.field = self.get_field_params()

    def reset(self, frame: Frame):
        placement_pos = self._placement_dict_from_frame(frame)
        self.simulator.reset(placement_pos["ball_pos"], placement_pos["blue_robots_pos"], placement_pos["yellow_robots_pos"])

    def stop(self):
        del self.simulator

    def send_commands(self, commands):
        raise NotImplementedError

    def get_frame(self) -> Frame:
        raise NotImplementedError

    def get_field_params(self):
        return Field(**self.simulator.get_field_params())
    
    def _placement_dict_from_frame(self, frame: Frame):
        replacement_pos: Dict[str, np.ndarray] = {}

        ball_pos: List[float] = [
            frame.ball.x,
            frame.ball.y,
            frame.ball.v_x,
            frame.ball.v_y,
        ]
        replacement_pos["ball_pos"] = np.array(ball_pos)

        blue_pos: List[List[float]] = []
        for robot in frame.robots_blue.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            blue_pos.append(robot_pos)
        replacement_pos["blue_robots_pos"] = np.array(blue_pos)

        yellow_pos: List[List[float]] = []
        for robot in frame.robots_yellow.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            yellow_pos.append(robot_pos)
        replacement_pos["yellow_robots_pos"] = np.array(yellow_pos)

        return replacement_pos

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
        raise NotImplementedError


class RSimVSS(RSim):
    def send_commands(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            sim_commands[rbt_id][0] = cmd.v_wheel0
            sim_commands[rbt_id][1] = cmd.v_wheel1
        self.simulator.step(sim_commands)

    def get_frame(self) -> FrameVSS:
        state = self.simulator.get_state()
        # Update frame with new state
        frame = FrameVSS()
        frame.parse(state, self.n_robots_blue, self.n_robots_yellow)

        return frame

    def _init_simulator(self, field_type, n_robots_blue, n_robots_yellow,
                        ball_pos, blue_robots_pos, yellow_robots_pos,
                        time_step_ms):

        return robosim.VSS(
            field_type,
            n_robots_blue,
            n_robots_yellow,
            time_step_ms,
            ball_pos,
            blue_robots_pos,
            yellow_robots_pos,
        )


class RSimSSL(RSim):
    def send_commands(self, commands):
        sim_cmds = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 8), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            if cmd.wheel_speed:
                sim_cmds[rbt_id][0] = cmd.wheel_speed
                sim_cmds[rbt_id][1] = cmd.v_wheel0
                sim_cmds[rbt_id][2] = cmd.v_wheel1
                sim_cmds[rbt_id][3] = cmd.v_wheel2
                sim_cmds[rbt_id][4] = cmd.v_wheel3
                sim_cmds[rbt_id][5] = cmd.kick_v_x
                sim_cmds[rbt_id][6] = cmd.kick_v_z
                sim_cmds[rbt_id][7] = cmd.dribbler
            else:
                sim_cmds[rbt_id][0] = cmd.wheel_speed
                sim_cmds[rbt_id][1] = cmd.v_x
                sim_cmds[rbt_id][2] = cmd.v_y
                sim_cmds[rbt_id][3] = cmd.v_theta
                sim_cmds[rbt_id][5] = cmd.kick_v_x
                sim_cmds[rbt_id][6] = cmd.kick_v_z
                sim_cmds[rbt_id][7] = cmd.dribbler
            
        self.simulator.step(sim_cmds)

    def get_frame(self) -> FrameSSL:
        state = self.simulator.get_state()
        # Update frame with new state
        frame = FrameSSL()
        frame.parse(state, self.n_robots_blue, self.n_robots_yellow)

        return frame
    
    def _init_simulator(self, field_type, n_robots_blue, n_robots_yellow,
                        ball_pos, blue_robots_pos, yellow_robots_pos,
                        time_step_ms):

        return robosim.SSL(
            field_type,
            n_robots_blue,
            n_robots_yellow,
            time_step_ms,
            ball_pos,
            blue_robots_pos,
            yellow_robots_pos,
        )
