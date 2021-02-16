import math
from enum import Enum

import numpy as np
from rc_gym.Entities.Field import VSSField
from rc_gym.Utils.Utils import clip, distancePointSegment
from rc_gym.vss.env_coach.deterministic_agents.pid import PID


class Actions(Enum):
    SPIN = 0
    MOVE = 1


class DeterministicAgent:

    field = VSSField()
    pid = PID()

    def __init__(self):
        self.spinDistance = 8.0
        self.action = Actions.MOVE

    def get_action(self, ball_pos: np.ndarray,
                   ball_speed: np.ndarray,
                   robot_pos: np.ndarray,
                   robot_angle: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_objective(self, ball_pos: np.ndarray,
                      ball_speed: np.ndarray,
                      robot_pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, ball_pos: np.ndarray,
                 ball_speed: np.ndarray,
                 robot_pos: np.ndarray,
                 robot_angle: np.ndarray) -> np.ndarray:
        return self.get_action(ball_pos=ball_pos, ball_speed=ball_speed,
                               robot_pos=robot_pos, robot_angle=robot_angle)

    def spin(self, ball_pos: np.ndarray,
             ball_speed: np.ndarray,
             robot_pos: np.ndarray) -> np.ndarray:
        spinDirection = False
        if (robot_pos[1] > ball_pos[1]):
            spinDirection = False
        else:
            spinDirection = True
        if(ball_pos[0] > self.field.middle[0] - 10):
            if(ball_pos[1] > self.field.middle[1]):
                if(ball_pos[1] < robot_pos[1] and ball_pos[0] > robot_pos[0]):
                    spinDirection = not spinDirection
            else:
                if(ball_pos[1] > robot_pos[1] and ball_pos[0] > robot_pos[0]):
                    spinDirection = not spinDirection

        if (ball_pos[0] < 20):
            if (ball_pos[0] < robot_pos[0]):
                if (ball_pos[1] < self.field.middle[1]):
                    spinDirection = False
                else:
                    spinDirection = True

        if(robot_pos[0] > self.field.m_max[0] - 3.75):
            if(ball_pos[0] < robot_pos[0]):
                p1 = ball_pos
                p2 = (ball_pos[0] + ball_speed[0]*5,
                      ball_pos[1] + ball_speed[1]*5)
                angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
                if(math.sin(angle) > 0):
                    spinDirection = True
                elif(math.sin(angle) < 0):
                    spinDirection = False

        if(spinDirection):
            return np.array([-0.7, 0.7])
        else:
            return np.array([0.7, -0.7])
