from dataclasses import dataclass

@dataclass()
class VSSField:
    height: float = 130.0
    length: float = 170.0
    offensiveCrossLine: float = 47.5
    deffensiveCrossLine: float = 122.5
    firstCross: float = 105.0
    secondCross: float = 65.0
    thirdCross: float = 25.0
    offsetX: float = 10.0
    offsetY: float = 0.0
    goalAreaWidth: float = 15.0
    goalAreaHeight: float = 70.0
    m_min: tuple = (10,0)
    m_max: tuple = (160,130)
    middle: tuple = (85,65)
    goalCenter: tuple =  (170,65)
    enemyGoalCenter: tuple = (0,65)
    goalMin: tuple = (160, 45)
    goalMax: tuple = (170, 85)
    goalAreaMin: tuple = (135,30)
    goalAreaMax: tuple = (160,100)
