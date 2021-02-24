import math
from enum import Enum

import numpy as np
from rc_gym.Entities import Frame
from rc_gym.Entities.Field import VSSField
from rc_gym.Utils.Utils import clip, distancePointSegment
from rc_gym.vss.env_coach.deterministic_agents.pid import PID
from rc_gym.vss.env_coach.deterministic_agents import univector_posture


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
                   robot_pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_objective(self, ball_pos: np.ndarray,
                      ball_speed: np.ndarray,
                      robot_pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, frame: Frame) -> np.ndarray:
        width = 1.3/2.0
        length = (1.5/2.0) + 0.1
        ball_pos = np.array([frame.ball.x, frame.ball.y])
        ball_speed = np.array([frame.ball.v_x, frame.ball.v_y])
        robot_pos = np.array([frame.robots_blue[self.robot_idx].x,
                              frame.robots_blue[self.robot_idx].y])
        robot_angle = frame.robots_blue[self.robot_idx].theta
        robot_angle = np.deg2rad(robot_angle + 180)
        if robot_angle > math.pi:
            robot_angle -= 2*math.pi
        elif robot_angle < -math.pi:
            robot_angle += 2*math.pi

        robot_angle = np.deg2rad(robot_angle)
        base_pos = np.array([length, width])
        robot_pos = base_pos - robot_pos
        robot_pos *= 100
        ball_pos = base_pos - ball_pos
        ball_pos *= 100
        ball_speed *= 100

        objective = self.get_action(ball_pos=ball_pos,
                                    ball_speed=ball_speed,
                                    robot_pos=robot_pos)

        if self.action == Actions.SPIN:
            speeds = self.spin(robot_pos, ball_pos, ball_speed)
        else:
            allies = []
            for ally in frame.robots_blue.values():
                ally_pos = np.array([ally.x, ally.y])
                ally_pos = base_pos - ally_pos
                ally_pos *= 100
                allies.append(ally_pos)
            enemies = []
            for enemy in frame.robots_yellow.values():
                enemy_pos = np.array([enemy.x, enemy.y])
                enemy_pos = base_pos - enemy_pos
                enemy_pos *= 100
                enemies.append(enemy_pos)
            next_step = univector_posture.update(robot_pos, objective, allies,
                                                 enemies, self.robot_idx)
            speeds = self.pid.run(robot_angle, next_step, robot_pos)
        return speeds

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
