from dataclasses import dataclass

@dataclass
class Ball:
    x: float = None
    y: float = None
    z: float = None
    vx: float = 0.0
    vy: float = 0.0