from dataclasses import dataclass

@dataclass
class Robot:
    yellow: bool
    id: int 
    kickVx: float
    kickVz: float
    vx: float
    vy: float
    vw: float
    dribbler: bool
    wheelSpeed: bool
    vWheel1: float
    vWheel2: float
    vWheel3: float
    vWheel4: float