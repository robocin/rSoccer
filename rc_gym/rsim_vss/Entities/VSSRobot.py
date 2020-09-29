from dataclasses import dataclass

@dataclass
class VSSRobot:
    yellow: bool = None
    id: int = None
    x: float = None
    y: float = None
    vx: float = None
    vy: float = None
    vwheel1: float = 0
    vwheel2: float = 0