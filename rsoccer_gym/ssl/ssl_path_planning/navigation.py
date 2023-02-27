"""Utility functions for path planning"""

from dataclasses import dataclass
from collections import namedtuple
from typing import Final, Optional
import math

PROP_VELOCITY_MIN_FACTOR: Final[float] = 0.1
MAX_VELOCITY: Final[float] = 2.2
ANGLE_EPSILON: Final[float] = 0.1
ANGLE_KP: Final[float] = 5
ROTATE_IN_POINT_MIN_VEL_FACTOR: Final[float] = 0.18
ROTATE_IN_POINT_APPROACH_KP: Final[float] = 2
ROTATE_IN_POINT_MAX_VELOCITY: Final[float] = 1.8
ROTATE_IN_POINT_ANGLE_KP: Final[float] = 5
MIN_DIST_TO_PROP_VELOCITY: Final[float] = 720

ADJUST_ANGLE_MIN_DIST: Final[float] = 50

M_TO_MM: Final[float] = 1000.0


Point2D = namedtuple("Point2D", ["x", "y"])
RobotMove = namedtuple("RobotMove", ["velocity", "angular_velocity"])


def dist_to(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the distance between two points"""
    return ((p_1.x - p_2.x) ** 2 + (p_1.y - p_2.y) ** 2) ** 0.5


def length(point: Point2D) -> float:
    """Returns the length of a vector"""
    return (point.x ** 2 + point.y ** 2) ** 0.5


def pt_angle(point: Point2D) -> float:
    """Returns the angle of a vector"""
    return math.atan2(point.y, point.x)


def math_map(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Maps a value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def math_bound(value: float, min_val: float, max_val: float) -> float:
    """Bounds a value between a min and max value"""
    return min(max_val, max(min_val, value))


def math_modularize(value: float, mod: float) -> float:
    """Make a value modular between 0 and mod"""
    if not -mod <= value <= mod:
        value = math.fmod(value, mod)

    if value < 0:
        value += mod

    return value


def smallest_angle_diff(angle_a: float, angle_b: float) -> float:
    """Returns the smallest angle difference between two angles"""
    angle: float = math_modularize(angle_b - angle_a, 2 * math.pi)
    if angle >= math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


def abs_smallest_angle_diff(angle_a: float, angle_b: float) -> float:
    """Returns the absolute smallest angle difference between two angles"""
    return abs(smallest_angle_diff(angle_a, angle_b))


def from_polar(radius: float, theta: float) -> Point2D:
    """Returns a point from polar coordinates"""
    return Point2D(radius * math.cos(theta), radius * math.sin(theta))


def cross(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the cross product of two points"""
    return p_1.x * p_2.y - p_1.y * p_2.x


def dot(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the dot product of two points"""
    return p_1.x * p_2.x + p_1.y * p_2.y


def angle_between(p_1: Point2D, p_2: Point2D) -> float:
    """Returns the angle between two points"""
    return math.atan2(cross(p_1, p_2), dot(p_1, p_2))


@dataclass()
class GoToPointEntry:
    """Go to point entry"""
    target: Point2D = Point2D(0.0, 0.0)
    target_angle: float = 0.0

    max_velocity: Optional[float] = None
    k_p: Optional[float] = None
    prop_velocity_factor: Optional[float] = None
    prop_min_distance: Optional[float] = None
    using_prop_velocity: bool = False


def go_to_point(agent_position: Point2D, agent_angle: float, entry: GoToPointEntry) -> RobotMove:
    """Returns the robot move"""
    # If Player send max speed, this max speed has to be respected
    # Ohterwise, use the max speed received in the parameter

    max_velocity: float = entry.max_velocity if entry.max_velocity else MAX_VELOCITY
    distance_to_goal: float = dist_to(agent_position, entry.target)
    k_p: float = entry.k_p if entry.k_p else ANGLE_KP

    # If it is to use proportional speed and the distance to the target position is less the
    # threshold
    if entry.using_prop_velocity:
        prop_velocity_factor: float = PROP_VELOCITY_MIN_FACTOR
        min_prop_distance: float = entry.prop_min_distance if entry.prop_min_distance else MIN_DIST_TO_PROP_VELOCITY

        # If Player send the min prop speed factor, it has to be respected
        # Otherwise, use defined in the parameter
        if entry.prop_velocity_factor and 0.0 <= entry.prop_velocity_factor <= 1.0:
            prop_velocity_factor = entry.prop_velocity_factor

        if distance_to_goal <= min_prop_distance:
            max_velocity = max_velocity * math_map(distance_to_goal, 0.0, min_prop_distance, prop_velocity_factor, 1.0)

        if distance_to_goal > ADJUST_ANGLE_MIN_DIST:
            # Uses an angle PID (only proportional for now), and first try to get in the right angle,
            # using only angular speed and then use linear speed to get into the point

            theta: float = pt_angle(Point2D(entry.target.x - agent_position.x, entry.target.y - agent_position.y))
            d_theta: float = smallest_angle_diff(agent_angle, entry.target_angle)

            # Proportional to prioritize the angle correction
            v_prop: float = abs(smallest_angle_diff(math.pi - ANGLE_EPSILON, d_theta)) * (max_velocity / (math.pi - ANGLE_EPSILON))

            return RobotMove(from_polar(v_prop, theta), k_p * d_theta)
        
        d_theta: float = smallest_angle_diff(agent_angle, entry.target_angle)
        return RobotMove(Point2D(0.0, 0.0), k_p * d_theta)