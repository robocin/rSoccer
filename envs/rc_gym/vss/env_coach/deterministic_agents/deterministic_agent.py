import math
from enum import Enum

import numpy as np
from rc_gym.Entities import Frame
from rc_gym.Entities.Field import VSSField
from rc_gym.Utils.Utils import clip, distancePointSegment
from rc_gym.vss.env_coach.deterministic_agents.pid import PID


class Actions(Enum):
    SPIN = 0
    MOVE = 1


class DeterministicAgent:

    field = VSSField()
    pid = PID()

    def __init__(self, robot_idx: int) -> None:
        self.spin_distance = 8.0
        self.action = Actions.MOVE
        self.robot_idx = robot_idx

    def get_action(self, ball_pos: np.ndarray,
                   ball_speed: np.ndarray,
                   robot_pos: np.ndarray,
                   robot_angle: float) -> np.ndarray:
        raise NotImplementedError

    def get_objective(self, ball_pos: np.ndarray,
                      ball_speed: np.ndarray,
                      robot_pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, frame: Frame) -> np.ndarray:
        ball_pos = np.array([frame.ball.x, frame.ball.y])
        ball_speed = np.array([frame.ball.v_x, frame.ball.v_y])
        robot_pos = np.array([frame.robots_blue[self.robot_idx].x,
                              frame.robots_blue[self.robot_idx].y])
        robot_angle = frame.robots_blue[self.robot_idx].theta
        if robot_angle > 180:
            robot_angle = -(360 - robot_angle) 

        return self.get_action(ball_pos=ball_pos, ball_speed=ball_speed,
                               robot_pos=robot_pos, robot_angle=robot_angle)

    def spin(self, ball_pos: np.ndarray,
             ball_speed: np.ndarray,
             robot_pos: np.ndarray) -> np.ndarray:
        spin_direction = False
        if (robot_pos[1] > ball_pos[1]):
            spin_direction = False
        else:
            spin_direction = True
        if(ball_pos[0] > self.field.middle[0] - 10):
            if(ball_pos[1] > self.field.middle[1]):
                if(ball_pos[1] < robot_pos[1] and ball_pos[0] > robot_pos[0]):
                    spin_direction = not spin_direction
            else:
                if(ball_pos[1] > robot_pos[1] and ball_pos[0] > robot_pos[0]):
                    spin_direction = not spin_direction

        if (ball_pos[0] < 20):
            if (ball_pos[0] < robot_pos[0]):
                if (ball_pos[1] < self.field.middle[1]):
                    spin_direction = False
                else:
                    spin_direction = True

        if(robot_pos[0] > self.field.m_max[0] - 3.75):
            if(ball_pos[0] < robot_pos[0]):
                p1 = ball_pos
                p2 = (ball_pos[0] + ball_speed[0]*5,
                      ball_pos[1] + ball_speed[1]*5)
                angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
                if(math.sin(angle) > 0):
                    spin_direction = True
                elif(math.sin(angle) < 0):
                    spin_direction = False

        if(spin_direction):
            return np.array([-0.7, 0.7])
        else:
            return np.array([0.7, -0.7])
