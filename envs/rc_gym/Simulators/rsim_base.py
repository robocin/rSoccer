import robosim
import numpy as np
from typing import Dict, List
from rc_gym.Entities import Frame


class RSim:
    def __init__(self, field_type: int, n_robots_blue: int,
                 n_robots_yellow: int, time_step_ms: int):
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow

        # Positions needed just to initialize the simulator
        ball_pos = [0, 0, 0, 0]
        blue_robots_pos = [[-0.2 * i, 0, 0]
                           for i in range(1, n_robots_blue + 1)]
        yellow_robots_pos = [[0.2 * i, 0, 0]
                             for i in range(1, n_robots_yellow + 1)]

        self.simulator = self._init_simulator(field_type=field_type,
                                              n_robots_blue=n_robots_blue,
                                              n_robots_yellow=n_robots_yellow,
                                              ball_pos=ball_pos,
                                              blue_robots_pos=blue_robots_pos,
                                              yellow_robots_pos=yellow_robots_pos,
                                              time_step_ms=time_step_ms)

    def reset(self, frame: Frame):
        placement_pos = self._placement_dict_from_frame(frame)
        self.simulator.reset(**placement_pos)

    def stop(self):
        del(self.simulator)

    def send_commands(self, commands):
        raise NotImplementedError

    def get_frame(self) -> Frame:
        raise NotImplementedError

    def get_field_params(self):
        return self.simulator.get_field_params()
    
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

    def _init_simulator(self, field_type, n_robots_blue, n_robots_yellow, 
                        ball_pos, blue_robots_pos, yellow_robots_pos,
                        time_step_ms):
        raise NotImplementedError
