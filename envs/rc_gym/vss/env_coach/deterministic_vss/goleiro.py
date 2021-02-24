import math
from enum import Enum

from rc_gym.vss.env_coach.deterministic_vss import Field, utils


class Actions(Enum):
    SPIN = 0
    MOVE = 1


class GoalKeeperDeterministic:
    def __init__(self):
        self.lineOffset = -5
        self.boundGK = 4.3
        self.spinDistance = 8.0
        self.action = Actions.MOVE

    def decideAction(self, ballPos, ballSpeed, robotPos):
        spinDirection = False
        p1 = ballPos
        p2 = (ballPos[0] + ballSpeed[0]*5, ballPos[1] + ballSpeed[1]*5)

        angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        if(math.sin(angle) > 0):
            spinDirection = True
        elif(math.sin(angle) < 0):
            spinDirection = False

        if(utils.euclideanDistance(robotPos, ballPos) < self.spinDistance and robotPos[0] + 5 > ballPos[0] and ((not spinDirection and robotPos[1] >= ballPos[1]) or ((spinDirection) and robotPos[1] <= ballPos[1]))):
            self.action = Actions.SPIN
        else:
            self.action = Actions.MOVE
            return self.decideObjective(ballPos, robotPos, ballSpeed)

    def decideObjective(self, ballPos, robotPos, ballSpeed):

        destination = [0, 0]

        destination[0] = Field.goalMin[0] + self.lineOffset

        p2 = (ballPos[0] + ballSpeed[0]*5, ballPos[1] + ballSpeed[1]*5)
        p1 = ballPos

        if((p1[0] != p2[0] or p1[1] != p2[1]) and ballSpeed[0] > 10.0 and ballSpeed[0] < Field.middle[0]):
            var_x = p2[0] - p1[0]
            var_y = p2[1] - p1[1]
            var2_x = destination[0] - p2[0]
            var2_y = (var2_x*var_y)/var_x
            destination[1] = p2[1] + var2_y
            if (destination[1] < Field.m_min[1]):
                if (destination[1] > Field.m_min[1]-Field.size[1]):
                    destination[1] = Field.m_min[1] + \
                        Field.m_min[1] - destination[1]
                else:
                    destination[1] = ballPos[1]

            if (destination[1] > Field.m_max[1]):
                if (destination[1] < Field.m_max[1] + Field.size[1]):
                    destination[1] = Field.m_max[1] - \
                        (destination[1] - Field.m_max[1])
                else:
                    destination[1] = ballPos[1]

            p3 = (ballPos[0] + ballSpeed[0]*500, ballPos[1] + ballSpeed[1]*500)

            distance = utils.distancePointSegment(p1, p3, robotPos)

            if(math.fabs(distance) < 2.75):
                destination = [robotPos[0], robotPos[1]]
        else:
            destination[1] = ballPos[1]

        destination[0] = Field.goalMin[0] + self.lineOffset
        destination[1] = utils.bound(
            destination[1], Field.goalMin[1]+self.boundGK, Field.goalMax[1]-self.boundGK)

        if (ballPos[1] >= Field.goalMin[1] - Field.goalAreaWidth * (2.0/3.0) and ballPos[1] <= Field.goalMax[1] + Field.goalAreaWidth * (2.0/3.0)):
            destination[1] = utils.bound(
                destination[1], Field.goalMin[1] + self.boundGK*1.53, Field.goalMax[1] - self.boundGK*1.53)

        return destination
        # self().setObjective(destination)
