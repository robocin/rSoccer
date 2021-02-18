import os

import torch
from rc_gym.vss.env_coach.stochastic_agents.stochastic_agent import \
    StochasticAgent


class Attacker(StochasticAgent):

    def __init__(self, robot_idx: int,
                 n_robots_blue: int,
                 n_robots_yellow: int,
                 linear_speed_range: float,
                 v_wheel_deadzone: int) -> None:
        super().__init__(robot_idx, n_robots_blue, n_robots_yellow,
                         linear_speed_range, v_wheel_deadzone)

    def load_model(self):
        atk_path = os.path.dirname(os.path.realpath(__file__))\
            + '/models/atk.pth'
        atk_checkpoint = torch.load(atk_path, map_location=self.device)
        self.model.load_state_dict(atk_checkpoint['state_dict_act'])
        self.model.eval()


class Defender(StochasticAgent):

    def __init__(self, robot_idx: int,
                 n_robots_blue: int,
                 n_robots_yellow: int,
                 linear_speed_range: float,
                 v_wheel_deadzone: int) -> None:
        super().__init__(robot_idx, n_robots_blue, n_robots_yellow,
                         linear_speed_range, v_wheel_deadzone)

    def load_model(self):
        atk_path = os.path.dirname(os.path.realpath(__file__))\
            + '/models/def.pth'
        atk_checkpoint = torch.load(atk_path, map_location=self.device)
        self.model.load_state_dict(atk_checkpoint['state_dict_act'])
        self.model.eval()


class Goalie(StochasticAgent):

    def __init__(self, robot_idx: int,
                 n_robots_blue: int,
                 n_robots_yellow: int,
                 linear_speed_range: float,
                 v_wheel_deadzone: int) -> None:
        super().__init__(robot_idx, n_robots_blue, n_robots_yellow,
                         linear_speed_range, v_wheel_deadzone)

    def load_model(self):
        atk_path = os.path.dirname(os.path.realpath(__file__))\
            + '/models/gk.pth'
        atk_checkpoint = torch.load(atk_path, map_location=self.device)
        self.model.load_state_dict(atk_checkpoint['state_dict_act'])
        self.model.eval()
