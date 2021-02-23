import math

import numpy as np
from rc_gym.Utils.Utils import (distance, distancePointSegment, insideOurArea,
                                projectPointToSegment)
from rc_gym.vss.env_coach.deterministic_agents import (Actions,
                                                       DeterministicAgent)
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


HALFAXIS = 3.75


class Defender(DeterministicAgent):

    def __init__(self, robot_idx: int):
        super().__init__(robot_idx)
        self.offensive_safe_radius = 25

    def get_action(self, ball_pos: np.ndarray,
                   ball_speed: np.ndarray,
                   robot_pos: np.ndarray,
                   robot_angle: np.ndarray) -> np.ndarray:
        can_spin = distance(robot_pos, ball_pos) < self.spin_distance
        should_spin = robot_pos[0] + 3 > ball_pos[0]
        should_spin &= not insideOurArea(ball_pos, 0, 0)
        if can_spin and should_spin:
            self.action = Actions.SPIN
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

        ball_pred = ball_pos + ball_speed*5
        angle_ball_goal = math.atan2(self.field.middle[1] - ball_pos[1],
                                     self.field.m_max[0]-ball_pos[0])
        ball_dir = math.atan2(ball_pred[1]-ball_pos[1],
                              ball_pred[0]-ball_pos[0])
        dest = [ball_pos[0] + math.cos(ball_dir)*self.offensive_safe_radius,
                ball_pos[1] + math.sin(ball_dir)*self.offensive_safe_radius]

        ball_point = Point(ball_pos)
        field_point = Point((self.field.goal_area_max[0],
                             self.field.goal_area_min[1]))
        goal_point = Point(self.field.goal_area_max)
        triangle = Polygon((ball_point, field_point, goal_point))
        if (triangle.contains(Point(ball_pred))):
            distance = distance(ball_pos, robot_pos)
            dest = [ball_pos[0] + math.cos(ball_dir)*distance,
                    ball_pos[1] + math.sin(ball_dir)*distance]
        else:
            pred_x = math.cos(angle_ball_goal)*self.offensive_safe_radius
            pred_y = math.sin(angle_ball_goal)*self.offensive_safe_radius
            dest = [ball_pos[0] + pred_x,
                    ball_pos[1] + pred_y]

        dest = np.array(dest)
        angle_to_dest = np.flip(dest - ball_pos)
        angle_to_dest = math.atan2(*angle_to_dest)
        pred_far = np.array([math.cos(angle_to_dest) * 500,
                             math.sin(angle_to_dest) * 500])
        auxPos = pred_far + ball_pos
        auxPos2 = ball_pos - pred_far

        robot_dist_to_goal = distancePointSegment(self.field.goal_max,
                                                  self.field.goal_min,
                                                  robot_pos)
        ball_dist_to_goal = distancePointSegment(self.field.goal_max,
                                                 self.field.goal_min,
                                                 ball_pos)
        robot_dist_to_pred_far = distancePointSegment(auxPos,
                                                      auxPos2,
                                                      robot_pos)

        robot_too_far = robot_dist_to_goal > ball_dist_to_goal
        robot_far_from_pred = robot_dist_to_pred_far < HALFAXIS*4
        if robot_too_far and robot_far_from_pred:
            if math.sin(angle_to_dest) > 0:
                angle_to_dest += math.pi/2
            else:
                angle_to_dest += -math.pi/2
            x_pred = math.cos(angle_to_dest) * self.offensive_safe_radius
            y_pred = math.sin(angle_to_dest) * self.offensive_safe_radius
            dest = [x_pred + ball_pos[0], y_pred + ball_pos[1]]
        else:
            if(robot_dist_to_pred_far > HALFAXIS*6):
                dest = projectPointToSegment(auxPos, auxPos2, robot_pos)

        if(insideOurArea(dest, 0, 0)):
            # print(dest)
            dest[0] -= self.field.goal_area_width

        return np.array(dest)
