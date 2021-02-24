import math

from rc_gym.vss.env_coach.deterministic_vss import Field, utils


def calculateVectAngle(source, target, desiredPos):
    derisedAngle = math.atan2(target[1] - source[1],  target[0] - source[0])
    rx = target[0] + 3 * math.cos(derisedAngle)
    ry = target[1] + 3 * math.sin(derisedAngle)
    anglePG = math.atan2(source[1] - target[1], source[0] - target[0])
    angleRG = math.atan2(target[1] - ry, target[0] - rx)
    anglePR = math.atan2(source[1] - ry, source[0] - rx)
    n = 8.0
    alpha = utils.to180range(anglePR - anglePG)
    ans = (anglePG - (n * alpha))
    return ans + math.pi


def getSurroundings(robotPos, ballPos, desiredPos):
    i = math.floor(robotPos[0])
    j = math.floor(robotPos[1])
    Nij = Ni1j = Ni1j1 = Nij1 = N = aux = [0, 0]
    da = db = dc = dd = 0.0
    ANij = calculateVectAngle((i, j), ballPos, desiredPos)
    Nij[0] = math.cos(ANij)
    Nij[1] = math.sin(ANij)
    ANi1j = calculateVectAngle((i + 1, j), ballPos, desiredPos)
    Ni1j[0] = math.cos(ANi1j)
    Ni1j[1] = math.sin(ANi1j)
    ANij1 = calculateVectAngle((i, j + 1), ballPos, desiredPos)
    Nij1[0] = math.cos(ANij1)
    Nij1[1] = math.sin(ANij1)
    ANi1j1 = calculateVectAngle((i + 1, j + 1), ballPos, desiredPos)
    Ni1j1[0] = math.cos(ANi1j1)
    Ni1j1[1] = math.sin(ANi1j1)
    da = utils.euclideanDistance(robotPos, (i + 1, j + 1))
    db = utils.euclideanDistance(robotPos, (i, j + 1))
    dc = utils.euclideanDistance(robotPos, (i + 1, j))
    dd = utils.euclideanDistance(robotPos, (i + 1, j + 1))
    N[0] = ((db * dc * dd * Nij[0]) + (da * dc * dd * Nij1[0]) + (da * db * dd * Ni1j[0]) +
            (da * db * dc * Ni1j1[0])) / (db * dc * dd + da * dc * dd + da * db * dd + da * db * dc)
    N[1] = ((db * dc * dd * Nij[1]) + (da * dc * dd * Nij1[1]) + (da * db * dd * Ni1j[1]) +
            (da * db * dc * Ni1j1[1])) / (db * dc * dd + da * dc * dd + da * db * dd + da * db * dc)
    norma = math.sqrt(N[0] * N[0] + N[1] * N[1])
    aux[0] = N[0] / norma
    aux[1] = N[1] / norma
    return math.atan2(aux[1], aux[0])


def adjustToObstacle(source, direction, obstaclePosition):
    Ro = 10.0
    M = 4.0
    distance = utils.euclideanDistance(source, obstaclePosition)
    length = math.fabs((obstaclePosition[0] - source[0]) * math.sin(
        direction) + (source[1] - obstaclePosition[1]) * math.cos(direction))
    angle = math.atan2(
        obstaclePosition[1] - source[1], obstaclePosition[0] - source[0])
    diff_angle = utils.smallestAngleDiff(direction, angle)

    if (length < Ro + M and math.fabs(diff_angle) < math.pi / 2.0):
        if (distance <= Ro):
            direction = angle - math.pi
        elif (distance <= Ro + M):
            alfa = 0.5

            if (diff_angle > 0.0):
                alfa = 1.5

            tmpx = ((distance - Ro) * math.cos(angle - alfa * math.pi) +
                    (Ro + M - distance) * math.cos(angle - math.pi)) / M
            tmpy = ((distance - Ro) * math.sin(angle - alfa * math.pi) +
                    (Ro + M - distance) * math.sin(angle - math.pi)) / M
            direction = math.atan2(tmpy, tmpx)
        else:
            multiplier = -1.0

            if (diff_angle > 0.0):
                multiplier = 1.0

            direction = multiplier * \
                math.fabs(math.atan((Ro + M) / math.sqrt(distance *
                                                         distance + (Ro + M) * (Ro + M)))) + angle

    return direction


def getVectorDirection(currentPosition, objectivePosition, robotPos, allies, enemies, index):
    direction = getSurroundings(
        currentPosition, objectivePosition, objectivePosition)

    for i in range(len(allies)):
        if (i == index):
            continue
        direction = adjustToObstacle(currentPosition, direction, allies[i])

    if(not utils.insideOurArea(robotPos, 0, 0)):
        for enemy in enemies:
            direction = adjustToObstacle(currentPosition, direction, enemy)

    return direction


def update(ballPos, robotPos, objPos, allies, enemies, index):

    iterations = 3
    curPos = [robotPos[0], robotPos[1]]

    for _ in range(iterations):
        angle = getVectorDirection(
            curPos, objPos, robotPos, allies, enemies, index)
        curPos = [curPos[0] + math.cos(angle) * 5,
                  curPos[1] + math.sin(angle) * 5]

        if (utils.euclideanDistance(curPos, objPos) < 8):
            curPos = objPos
            break

    angle = getVectorDirection(
        robotPos, objPos, robotPos, allies, enemies, index)
    # if(utils.insideOurArea(ballPos, 0, 0)):
    #    if(utils.insideOurArea(curPos, 0, 0)):
    #        curPos[0] = utils.bound(curPos[0], 0, curPos[0] - Field.goalAreaMin[0])

    # self().setNextStepAngle(angle);
    return curPos
