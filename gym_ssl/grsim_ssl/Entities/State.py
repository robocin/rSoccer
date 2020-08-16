from dataclasses import dataclass

@dataclass
class State:
    keeperY: float = None
    keeperVy: float = None
    pEP: float = None
    ballVx:float = None
    ballVy:float = None
    attackerX:float = None
    attackerY:float = None
    attackerW:float = None
