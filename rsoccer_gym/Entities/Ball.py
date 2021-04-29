from dataclasses import dataclass

@dataclass()
class Ball:
    x: float = None
    y: float = None
    z: float = None
    v_x: float = 0.0
    v_y: float = 0.0
    v_z: float = 0.0