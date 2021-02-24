import math
from enum import Enum

from rc_gym.vss.env_coach.deterministic_vss import Field, utils


class Actions(Enum):
    SPIN = 0
    MOVE = 1


class DefenderDeterministic:
    def __init__(self):
        self.spinDistance = 8
        self.action = Actions.MOVE
        self.offensiveSafeRadius = 25

    def decideAction(self, ballPos, ballSpeed, robotPos):
        if (utils.euclideanDistance(robotPos, ballPos) < self.spinDistance and robotPos[0] + 3 > ballPos[0] and not utils.insideOurArea(ballPos, 0, 0)):
            self.action = Actions.SPIN
        else:
            self.action = Actions.MOVE
            return self.decideObjective(ballPos, ballSpeed, robotPos)

    def decideObjective(self, ballPosition, ballSpeed, robotPos):
        dest = [0, 0]
        p2 = (ballPosition[0] + ballSpeed[0]*5,
              ballPosition[1] + ballSpeed[1]*5)
        p1 = ballPosition

        angleBallGoal = math.atan2(
            Field.middle[1] - ballPosition[1], -ballPosition[0])
        angleBallToOurGoal = math.atan2(
            Field.middle[1] - ballPosition[1], Field.m_max[0]-ballPosition[0])

        ballDirection = math.atan2(p2[1]-p1[1], p2[0]-p1[0])

        dest = [ballPosition[0] + math.cos(ballDirection)*self.offensiveSafeRadius,
                ballPosition[1] + math.sin(ballDirection)*self.offensiveSafeRadius]

        triangle = [
            p1, (Field.goalAreaMax[0], Field.goalAreaMin[1]), Field.goalAreaMax]

        if (utils.PointInPolygon(triangle, p2)):
            distance = utils.euclideanDistance(ballPosition, robotPos)
            dest = [ballPosition[0] + math.cos(ballDirection)*distance,
                    ballPosition[1] + math.sin(ballDirection)*distance]
        else:
            dest = [ballPosition[0] + math.cos(angleBallToOurGoal)*self.offensiveSafeRadius,
                    ballPosition[1] + math.sin(angleBallToOurGoal)*self.offensiveSafeRadius]

        angle = math.atan2(dest[1]-ballPosition[1], dest[0]-ballPosition[0])
        auxPos = ((math.cos(angle) * 500) +
                  ballPosition[0], (math.sin(angle) * 500) + ballPosition[1])
        auxPos2 = ((math.cos(angle+math.pi) * 500) +
                   ballPosition[0], (math.sin(angle+math.pi) * 500) + ballPosition[1])

        if(utils.distancePointSegment(Field.goalMax, Field.goalMin, robotPos) > utils.distancePointSegment(Field.goalMax, Field.goalMin, ballPosition) and utils.distancePointSegment(auxPos, auxPos2, robotPos) < utils.halfAxis*4):
            angle += math.pi/2 if math.sin(angle) > 0 else -math.pi/2
            dest = [(math.cos(angle) * self.offensiveSafeRadius) + ballPosition[0],
                    (math.sin(angle) * self.offensiveSafeRadius) + ballPosition[1]]
        else:
            if(utils.distancePointSegment(auxPos, auxPos2, robotPos) > utils.halfAxis*6):
                dest = utils.projectPointToSegment(auxPos, auxPos2, robotPos)

        if(utils.insideOurArea(dest, 0, 0)):
            # print(dest)
            dest[0] -= Field.goalAreaWidth

        return dest
