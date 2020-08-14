from dataclasses import dataclass

@dataclass
class Robot:
    yellow: bool
    id: int 
    vx: float
    vy: float
    vw: float
    kickVx: float = 0
    kickVz: float = 0
    dribbler: bool = False
    wheelSpeed: bool = False
    vWheel1: float = 0
    vWheel2: float = 0
    vWheel3: float = 0
    vWheel4: float = 0