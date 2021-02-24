import math

from rc_gym.vss.env_coach.deterministic_vss import Field

halfAxis = 3.75


def dot(p1, p2):
    return (p1[0] * p2[0]) + (p1[1] * p2[1])


def projectPointToSegment(t_a, t_b, t_c):
    p = (t_b[0] - t_a[0], t_b[1] - t_a[1])
    r = dot(p, p)

    if (math.fabs(r) < 0.0001):
        return t_a

    r = dot((t_c[0] - t_a[0], t_c[1] - t_a[1]),
            (t_b[0] - t_a[0], t_b[1] - t_a[1])) / r

    if (r < 0):
        return t_a

    if (r > 1):
        return t_b
    aux = (t_b[0] - t_a[0], t_b[1] - t_a[1])
    return [t_a[0] + aux[0] * r, t_a[1] + aux[1] * r]


def distancePointSegment(t_a, t_b, t_c):
    return euclideanDistance(t_c, projectPointToSegment(t_a, t_b, t_c))


def euclideanDistance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)


def bound(x, m_min, m_max):
    if x < m_min:
        return m_min
    elif x > m_max:
        return m_max
    return x


def to180range(angle):
    a = math.fmod(angle, 2.0*math.pi)
    if (a < -math.pi):
        a = a + 2.0 * math.pi
    elif (a > math.pi):
        a = a - 2.0 * math.pi
    return a


def to_positive_angle(angle):
    return math.fmod(math.fmod(angle, 2*math.pi), math.pi)


def smallestAngleDiff(target, source):
    a = to_positive_angle(target) - to_positive_angle(source)

    if (a > math.pi):
        a = a - 2.0 * math.pi
    elif (a < -math.pi):
        a = a + 2.0 * math.pi
    return a


def insideOurArea(pos, sumX, sumY):
    return bool(pos[0] > Field.goalAreaMin[0]-sumX) and (pos[1] > Field.goalAreaMin[1]-sumY) and (pos[1] < Field.goalAreaMax[1]+sumY)


def PointInPolygon(p, q):
    c = False
    p_size = len(p)
    for i in range(p_size):
        j = (i + 1) % p_size
        if((p[i][1] <= q[1] and q[1] < p[j][1] or p[j][1] <= q[1] and q[1] < p[i][1]) and q[0] < p[i][0] + (p[j][0] - p[i][0]) * (q[1] - p[i][1]) / (p[j][1] - p[i][1])):
            c = not c

    return c


def spin(robotPos, ballPos, ballSpeed):
    spinDirection = False
    if (robotPos[1] > ballPos[1]):
        spinDirection = False
    else:
        spinDirection = True
    if(ballPos[0] > Field.middle[0] - 10):
        if(ballPos[1] > Field.middle[1]):
            if(ballPos[1] < robotPos[1] and ballPos[0] > robotPos[0]):
                spinDirection = not spinDirection
        else:
            if(ballPos[1] > robotPos[1] and ballPos[0] > robotPos[0]):
                spinDirection = not spinDirection

    if (ballPos[0] < 20):
        if (ballPos[0] < robotPos[0]):
            if (ballPos[1] < Field.middle[1]):
                spinDirection = False
            else:
                spinDirection = True

    if(robotPos[0] > Field.m_max[0] - 3.75):
        if(ballPos[0] < robotPos[0]):
            p1 = ballPos
            p2 = (ballPos[0] + ballSpeed[0]*5, ballPos[1] + ballSpeed[1]*5)
            angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
            if(math.sin(angle) > 0):
                spinDirection = True
            elif(math.sin(angle) < 0):
                spinDirection = False

    if(spinDirection):
        return(-70, 70)
    else:
        return (70, -70)


def isNearToWall(position, alpha):
    res = 0
    if(position[1] <= Field.m_min[1]+3.75*alpha and res != 3):
        res = 1
    if(position[1] >= Field.m_max[1]-3.75*alpha and res != 3):
        res = 2
    if (position[1] >= Field.goalMin[1] + (3.75*alpha) and position[1] <= Field.goalMax[1] - (3.75*alpha)):
        pass
    else:
        if (position[0] <= Field.m_min[0] + (3.75*alpha)):
            res = 4
        if (position[0] >= Field.m_max[0] - (3.75*alpha)):
            res = 3
    return res
