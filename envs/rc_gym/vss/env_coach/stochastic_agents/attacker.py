import os

import numpy as np
import torch
from rc_gym.Entities import Frame
from rc_gym.Utils import normVt, normVx, normX
from rc_gym.vss.env_coach.stochastic_agents.ddpg import DDPGActor


class Attacker:

    model = DDPGActor(40, 2)

    def __init__(self, robot_idx: int,
                 n_robots_blue: int,
                 n_robots_yellow: int,
                 linear_speed_range: float,
                 v_wheel_deadzone: int) -> None:
        self.robot_idx = robot_idx
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.device = torch.device('cpu')
        self.linear_speed_range = linear_speed_range
        self.v_wheel_deadzone = v_wheel_deadzone
        self.load_model()

    def load_model(self):
        atk_path = os.path.dirname(os.path.realpath(__file__))\
            + '/models/atk.pth'
        atk_checkpoint = torch.load(atk_path, map_location=self.device)
        self.model.load_state_dict(atk_checkpoint['state_dict_act'])
        self.model.eval()

    def __call__(self, frame: Frame) -> np.ndarray:
        observation = self._frame_to_observations(frame)
        actions = self.model.get_action(observation)
        speeds = self._actions_to_v_wheels(actions=actions)

        return speeds

    def _actions_to_v_wheels(self, actions: np.ndarray) -> tuple:
        left_wheel_speed = actions[0] * self.linear_speed_range
        right_wheel_speed = actions[1] * self.linear_speed_range

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -2.6, 2.6)

        return left_wheel_speed, right_wheel_speed

    def get_rotated_obs(self, frame: Frame) -> list:
        robots_dict = dict()
        for i in range(self.n_robots_blue):
            robots_dict[i] = list()
            robots_dict[i].append(normX(frame.robots_blue[i].x))
            robots_dict[i].append(normX(frame.robots_blue[i].y))
            robots_dict[i].append(
                np.sin(np.deg2rad(frame.robots_blue[i].theta))
            )
            robots_dict[i].append(
                np.cos(np.deg2rad(frame.robots_blue[i].theta))
            )
            robots_dict[i].append(normVx(frame.robots_blue[i].v_x))
            robots_dict[i].append(normVx(frame.robots_blue[i].v_y))
            robots_dict[i].append(normVt(frame.robots_blue[i].v_theta))

        aux_dict = {}
        aux_dict.update(robots_dict)
        rotated = list()
        rotated = rotated + aux_dict.pop(self.robot_idx)
        teammates = list(aux_dict.values())
        for teammate in teammates:
            rotated = rotated + teammate

        return rotated

    def _frame_to_observations(self, frame: Frame) -> np.ndarray:

        observation = list()
        teammates = self.get_rotated_obs(frame)

        observation.append(normX(frame.ball.x))
        observation.append(normX(frame.ball.y))
        observation.append(normVx(frame.ball.v_x))
        observation.append(normVx(frame.ball.v_y))

        observation += teammates

        for i in range(self.n_robots_yellow):
            observation.append(normX(frame.robots_yellow[i].x))
            observation.append(normX(frame.robots_yellow[i].y))
            observation.append(normVx(frame.robots_yellow[i].v_x))
            observation.append(normVx(frame.robots_yellow[i].v_y))
            observation.append(normVt(frame.robots_yellow[i].v_theta))

        return np.array(observation)
