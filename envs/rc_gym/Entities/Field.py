from dataclasses import dataclass

@dataclass()
class VSSField:
    height: float = 130.0
    length: float = 170.0
    goal_area_width: float = 15.0
    goal_area_height: float = 70.0
    m_min: tuple = (10,0)
    m_max: tuple = (160,130)
    middle: tuple = (85,65)
    goal_center: tuple =  (170,65)
    enemy_goal_center: tuple = (0,65)
    goal_min: tuple = (160, 45)
    goal_max: tuple = (170, 85)
    goal_area_min: tuple = (135,30)
    goal_area_max: tuple = (160,100)
