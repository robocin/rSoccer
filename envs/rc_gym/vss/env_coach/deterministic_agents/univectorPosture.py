import math

from rc_gym.Utils.Utils import (distance, insideOurArea,
                                smallestAngleDiff, to_pi_range)


def calculateVectAngle(source, target):
    derisedAngle = math.atan2(target[1] - source[1],  target[0] - source[0])
    rx = target[0] + 3 * math.cos(derisedAngle)
    ry = target[1] + 3 * math.sin(derisedAngle)
    anglePG = math.atan2(source[1] - target[1], source[0] - target[0])
    anglePR = math.atan2(source[1] - ry, source[0] - rx)
    n = 8.0
    alpha = to_pi_range(anglePR - anglePG)
    ans = (anglePG - (n * alpha))
    return ans + math.pi


def getSurroundings(robotPos, ballPos):
    i = math.floor(robotPos[0])
    j = math.floor(robotPos[1])
    Nij = Ni1j = Ni1j1 = Nij1 = N = aux = [0, 0]
    da = db = dc = dd = 0.0
    ANij = calculateVectAngle((i, j), ballPos)
    Nij[0] = math.cos(ANij)
    Nij[1] = math.sin(ANij)
    ANi1j = calculateVectAngle((i + 1, j), ballPos)
    Ni1j[0] = math.cos(ANi1j)
    Ni1j[1] = math.sin(ANi1j)
    ANij1 = calculateVectAngle((i, j + 1), ballPos)
    Nij1[0] = math.cos(ANij1)
    Nij1[1] = math.sin(ANij1)
    ANi1j1 = calculateVectAngle((i + 1, j + 1), ballPos)
    Ni1j1[0] = math.cos(ANi1j1)
    Ni1j1[1] = math.sin(ANi1j1)
    da = distance(robotPos, (i + 1, j + 1))
    db = distance(robotPos, (i, j + 1))
    dc = distance(robotPos, (i + 1, j))
    dd = distance(robotPos, (i + 1, j + 1))
    prod1 = da * db * dc
    prod2 = da * db * dd
    prod3 = da * dc * dd
    prod4 = db * dc * dd
    prod_sum = prod1 + prod2 + prod3 + prod4
    N[0] = (prod1 * Ni1j1[0]) + (prod2 * Ni1j[0]) \
        + (prod3 * Nij1[0]) + (prod4 * Nij[0])
    N[0] = N[0]/prod_sum
    N[1] = (prod1 * Ni1j1[1]) + (prod2 * Ni1j[1]) \
        + (prod3 * Nij1[1]) + (prod4 * Nij[1])
    N[1] = N[1]/prod_sum
    norma = math.sqrt(N[0] * N[0] + N[1] * N[1])
    aux[0] = N[0] / norma
    aux[1] = N[1] / norma
    return math.atan2(aux[1], aux[0])


def adjustToObstacle(source, direction, obstaclePosition):
    Ro = 10.0
    M = 4.0
    distance = distance(source, obstaclePosition)
    length_y = (obstaclePosition[0] - source[0]) * math.sin(direction)
    length_x = (source[1] - obstaclePosition[1]) * math.cos(direction)
    length = math.fabs(length_y + length_x)
    angle = math.atan2(obstaclePosition[1] - source[1],
                       obstaclePosition[0] - source[0])
    diff_angle = smallestAngleDiff(direction, angle)

    if (length < Ro + M and math.fabs(diff_angle) < math.pi / 2.0):
        if (distance <= Ro):
            direction = angle - math.pi
        elif (distance <= Ro + M):
            alfa = 0.5

            if (diff_angle > 0.0):
                alfa = 1.5

            tmpx = ((distance - Ro) * math.cos(angle - alfa * math.pi)
                    + (Ro + M - distance) * math.cos(angle - math.pi)) / M
            tmpy = ((distance - Ro) * math.sin(angle - alfa * math.pi)
                    + (Ro + M - distance) * math.sin(angle - math.pi)) / M
            direction = math.atan2(tmpy, tmpx)
        else:
            multiplier = -1.0

            if (diff_angle > 0.0):
                multiplier = 1.0

            direction = multiplier \
                * math.fabs(math.atan((Ro + M)
                                      / math.sqrt(distance * distance
                                                  + (Ro + M) * (Ro + M))))\
                + angle

    return direction


def getVectorDirection(currentPosition, objectivePosition,
                       robotPos, allies, enemies, index):
    direction = getSurroundings(currentPosition, objectivePosition)

    for i in range(len(allies)):
        if (i == index):
            continue
        direction = adjustToObstacle(currentPosition, direction, allies[i])

    if(not insideOurArea(robotPos, 0, 0)):
        for enemy in enemies:
            direction = adjustToObstacle(currentPosition, direction, enemy)

    return direction


def update(robotPos, objPos, allies, enemies, index):

    iterations = 3
    curPos = [robotPos[0], robotPos[1]]

    for _ in range(iterations):
        angle = getVectorDirection(curPos, objPos,
                                   robotPos, allies,
                                   enemies, index)
        curPos = [curPos[0] + math.cos(angle) * 5,
                  curPos[1] + math.sin(angle) * 5]

        if (distance(curPos, objPos) < 8):
            curPos = objPos
            break

    angle = getVectorDirection(robotPos, objPos,
                               robotPos, allies,
                               enemies, index)
    return curPos
