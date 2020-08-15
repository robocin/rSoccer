from dataclasses import dataclass

@dataclass
class Robot:
    yellow: bool = None
    id: int = None
    vx: float = 0
    vy: float = 0
    vw: float = 0
    x: float = None
    y: float = None
    w: float = None
    kickVx: float = 0
    kickVz: float = 0
    dribbler: bool = False
    wheelSpeed: bool = False
    vWheel1: float = 0
    vWheel2: float = 0
    vWheel3: float = 0
    vWheel4: float = 0