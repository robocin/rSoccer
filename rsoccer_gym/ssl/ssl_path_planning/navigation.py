"""Utility functions for path planning"""

from dataclasses import dataclass
from collections import namedtuple
from typing import Final, Optional
import math

ANGLE_K_P: Final[float] = 4.0
ALLY_MAX_SPEED: Final[float] = 2200.0
ALLY_MIN_SPEED: Final[float] = 0.30
MIN_DIST_TO_PROP_VELOCITY: Final[float] = 800.00

SMALL_TOLERANCE_TO_DESIRED_POSITION: Final[float] = 10.0  # Not a parameter.
DEFAULT_TOLERANCE_TO_DESIRED_POSITION: Final[float] = 35.0  # Not a parameter.

ROBOT_VEL_BREAK_DECAY_FACTOR: Final[float] = 2.11
ROBOT_VEL_FAVORABLE_DECAY_FACTOR: Final[float] = 0.09

ROBOT_MAX_LINEAR_ACCELERATION: Final[float] = 2.4

ROBOT_MAX_ANGULAR_ACCELERATION: Final[float] = 10.0

CYCLE_STEP: Final[float] = 0.16

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
    custom_acceleration: Optional[float] = None
    min_velocity: Optional[float] = None
    prop_min_distance: Optional[float] = None
    using_prop_velocity: Optional[bool] = None
    required_high_precision_to_target: Optional[bool] = None


def go_to_point(agent_position: Point2D, agent_velocity: Point2D, agent_angle: float, entry: GoToPointEntry) -> RobotMove:
    """Returns the robot move"""
    s0: Point2D = agent_position
    s: Point2D = entry.target

    delta_s: Point2D = Point2D(s.x - s0.x, s.y - s0.y)
    k_p: float = entry.k_p if entry.k_p is not None else ANGLE_K_P

    def compute_max_velocity():
        max_velocity: float = ALLY_MAX_SPEED / \
            M_TO_MM if entry.max_velocity is None else entry.max_velocity

        if entry.using_prop_velocity:
            min_prop_distance: float = entry.prop_min_distance if entry.prop_min_distance is not None else MIN_DIST_TO_PROP_VELOCITY
            min_velocity: float = entry.min_velocity if entry.min_velocity is not None else ALLY_MIN_SPEED

            if length(delta_s) <= min_prop_distance:
                return max(math_map(length(delta_s), 0.0, min_prop_distance, min_velocity, max_velocity), min_velocity)

        return max_velocity

    max_velocity: float = compute_max_velocity()

    tolerance_to_target: float = SMALL_TOLERANCE_TO_DESIRED_POSITION if entry.required_high_precision_to_target else DEFAULT_TOLERANCE_TO_DESIRED_POSITION

    if length(delta_s) > tolerance_to_target:
        theta: float = pt_angle(delta_s)
        d_theta: float = smallest_angle_diff(
            agent_angle, entry.target_angle)

        delta_s = Point2D(delta_s.x / M_TO_MM, delta_s.y / M_TO_MM)

        v0: Point2D = Point2D(agent_velocity.x /
                              M_TO_MM, agent_velocity.y / M_TO_MM)
        v: Point2D = from_polar(max_velocity, theta)

        v0_decay: float = ROBOT_VEL_BREAK_DECAY_FACTOR if abs(angle_between(
            v, v0)) > math.pi / 2.0 else ROBOT_VEL_FAVORABLE_DECAY_FACTOR

        v0 = Point2D(v0.x - (v0.x * v0_decay) * CYCLE_STEP,
                     v0.y - (v0.y * v0_decay) * CYCLE_STEP)

        acceleration_required: Point2D = Point2D((v.x - v0.x) / CYCLE_STEP,
                                                 (v.y - v0.y) / CYCLE_STEP)

        acc_prop: float = entry.custom_acceleration if entry.custom_acceleration else ROBOT_MAX_LINEAR_ACCELERATION

        alpha: float = math_map(abs(d_theta), 0.0, math.pi, 0.0, 1.0)

        prop_factor: float = (alpha - 1.0) * (alpha - 1.0)

        acc_prop *= prop_factor

        if length(acceleration_required) > acc_prop:
            # acceleration_required *= acc_prop / length(acceleration_required)
            acceleration_required = Point2D(acceleration_required.x * acc_prop / length(acceleration_required),
                                            acceleration_required.y * acc_prop / length(acceleration_required))

        # // v = v0 + a*t
        new_velocity: Point2D = Point2D(v0.x + acceleration_required.x * CYCLE_STEP,
                                        v0.y + acceleration_required.y * CYCLE_STEP)

        # w0: float = 0
        # w: float = k_p * d_theta

        # // w0 = w0 - (w0 * v0Decay) * Env::Navigation::CYCLE_STEP
        # rotate_acceleration_required: float = (w - w0) / CYCLE_STEP

        # acc_rotate: float = math_bound(
        #     rotate_acceleration_required,
        #     -ROBOT_MAX_ANGULAR_ACCELERATION,
        #     ROBOT_MAX_ANGULAR_ACCELERATION
        # )

        # // v = v0 + a*t
        # new_ang_velocity: float = w0 + acc_rotate * CYCLE_STEP

        return RobotMove(velocity=new_velocity, angular_velocity=k_p * d_theta)

    def angle_pid(target_angle: float, k_p: float) -> RobotMove:
        d_theta: float = smallest_angle_diff(
            agent_angle, target_angle)

        return RobotMove(velocity=Point2D(0.0, 0.0), angular_velocity=k_p * d_theta)

    return angle_pid(entry.target_angle, k_p)
