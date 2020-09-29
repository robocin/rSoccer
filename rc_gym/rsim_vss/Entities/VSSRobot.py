from dataclasses import dataclass

@dataclass
class VSSRobot:
    yellow: bool = None
    id: int = None
    x: float = None
    y: float = None
    vx: float = None
    vy: float = None
    vWheel1: float = 0
    vWheel2: float = 0