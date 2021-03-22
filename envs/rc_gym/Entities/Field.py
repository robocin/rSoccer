from dataclasses import dataclass

@dataclass()
class Field:
    length: float
    width: float
    penalty_length: float
    penalty_width: float
    goal_width: float
    goal_depth: float
    ball_radius: float
    rbt_distance_center_kicker: float
    rbt_kicker_thickness: float
    rbt_kicker_width: float
    rbt_wheel0_angle: float
    rbt_wheel1_angle: float
    rbt_wheel2_angle: float
    rbt_wheel3_angle: float
    rbt_radius: float
    rbt_wheel_radius: float
    rbt_motor_max_rpm: float