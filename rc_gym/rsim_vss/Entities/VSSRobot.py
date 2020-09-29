from dataclasses import dataclass

@dataclass
class VSSRobot:
    yellow: bool = None
    id: int = None
    x: float = None
    y: float = None
    vx: float = None
    vy: float = None
    v_wheel1: float = 0
    v_wheel2: float = 0