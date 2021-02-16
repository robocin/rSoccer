import math

import numpy as np
from rc_gym.Utils.Utils import clip, distancePointSegment
from rc_gym.vss.env_coach.deterministic_agents import (Actions,
                                                       DeterministicAgent)


class Goalie(DeterministicAgent):

    def __init__(self, robot_idx: int):
        super().__init__(robot_idx)
        self.lineOffset = -3
        self.boundGK = 4.3

    def get_action(self, ball_pos: np.ndarray,
                   ball_speed: np.ndarray,
                   robot_pos: np.ndarray,
                   robot_angle: float) -> np.ndarray:
        spinDirection = False
        pred_ball = ball_pos + ball_speed*5

        angle = math.atan2(*np.flip(ball_pos - pred_ball))
        if math.sin(angle) > 0:
            spinDirection = True
        elif math.sin(angle) < 0:
            spinDirection = False

        dist_ball = np.linalg.norm(robot_pos - ball_pos)
        should_spin = dist_ball < self.spinDistance
        should_spin &= (robot_pos[0] + 5) > ball_pos[0]
        ask1 = not spinDirection and robot_pos[1] >= ball_pos[1]
        ask2 = spinDirection and robot_pos[1] <= ball_pos[1]
        ask3 = ask1 or ask2
        should_spin &= ask3

        if should_spin:
            self.action = Actions.SPIN
            return self.spin(robot_pos, ball_pos, ball_speed)
        else:
            self.action = Actions.MOVE
            self.objective = self.get_objective(ball_pos,
                                                robot_pos,
                                                ball_speed)
            speeds = self.pid.run(robot_angle,
                                  self.objective,
                                  robot_pos)
            return speeds

    def get_objective(self, ball_pos: np.ndarray,
                      ball_speed: np.ndarray,
                      robot_pos: np.ndarray) -> np.ndarray:

        destination = np.array([0, 0])
        destination[0] = self.field.goalMin[0] + self.lineOffset
        pred_ball = ball_pos + ball_speed*5
        check_axis = ball_pos - pred_ball
        check_axis = np.any(check_axis == 0)
        inbound = ball_speed[0] < self.field.middle[0]

        if check_axis and ball_speed[0] > 10.0 and inbound:
            var_x = pred_ball[0] - ball_pos[0]
            var_y = pred_ball[1] - ball_pos[1]
            var2_x = destination[0] - pred_ball[0]
            var2_y = (var2_x*var_y)/var_x
            destination[1] = pred_ball[1] + var2_y
            bound_low = self.field.m_min[1] - self.field.height
            bound_high = self.field.m_min[1] + self.field.height
            inbound = destination[1] < self.field.m_min[1]
            if inbound:
                if (destination[1] > bound_low):
                    destination[1] = self.field.m_min[1] \
                        + self.field.m_min[1] \
                        - destination[1]
                else:
                    destination[1] = ball_pos[1]

            else:
                if (destination[1] < bound_high):
                    destination[1] = self.field.m_max[1] \
                        - destination[1] \
                        - self.field.m_max[1]
                else:
                    destination[1] = ball_pos[1]

            pred_far = ball_pos + ball_speed*500
            distance = distancePointSegment(ball_pos, pred_far, robot_pos)

            if(math.fabs(distance) < 2.75):
                destination = robot_pos
        else:
            destination[1] = ball_pos[1]

        destination[0] = self.field.goalMin[0] + self.lineOffset
        destination[1] = clip(destination[1],
                              self.field.goalMin[1]+self.boundGK,
                              self.field.goalMax[1]-self.boundGK)

        range1 = self.field.goalMin[1] - self.field.goalAreaWidth * (2.0/3.0)
        range2 = self.field.goalMax[1] + self.field.goalAreaWidth * (2.0/3.0)
        dunno = ball_pos[1] >= self.field.goalMin[1] - range1
        dunno &= ball_pos[1] <= self.field.goalMax[1] + range2
        if dunno:
            destination[1] = clip(destination[1],
                                  self.field.goalMin[1] + self.boundGK*1.53,
                                  self.field.goalMax[1] - self.boundGK*1.53)
        destination = np.array(destination)
        return destination
