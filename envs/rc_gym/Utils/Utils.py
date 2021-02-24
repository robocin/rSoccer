import math
import numpy as np
import gym
from rc_gym.Entities.Field import VSSField


# Math methods
# ----------------------------
def distance(pointA, pointB):

    x1 = pointA[0]
    y1 = pointA[1]

    x2 = pointB[0]
    y2 = pointB[1]

    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return distance


def mod(x, y):
    return (math.sqrt(x*x + y*y))


def angle(x, y):
    return math.atan2(y, x)


def toPositiveAngle(angle):
    return math.fmod(angle + 2 * math.pi, 2 * math.pi)


def smallestAngleDiff(target, source):
    a = toPositiveAngle(target) - toPositiveAngle(source)

    if a > math.pi:
        a = a - 2 * math.pi
    elif a < -math.pi:
        a = a + 2 * math.pi

    return a


def to_pi_range(angle):
    angle = math.fmod(angle, 2 * math.pi)
    if angle < -math.pi:
        angle = angle + 2 * math.pi
    elif angle > math.pi:
        angle = angle - 2 * math.pi

    return angle


def clip(val, vmin, vmax):
    return min(max(val, vmin), vmax)


def normX(x):
    return clip(x / 0.85, -1.2, 1.2)


def normVx(v_x):
    return clip(v_x / 0.8, -1.25, 1.25)


def normVt(vt):
    return clip(vt / 573, -1.2, 1.2)


def roundTo5(x, base=5):
    return int(base * round(float(x) / base))


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
    return np.array([t_a[0] + aux[0] * r, t_a[1] + aux[1] * r])


def distancePointSegment(t_a, t_b, t_c):
    return distance(t_c, projectPointToSegment(t_a, t_b, t_c))


def insideOurArea(pos, sumX, sumY):
    field = VSSField()
    check_x = pos[0] > field.goal_area_min[0]-sumX
    check_y = pos[1] > field.goal_area_min[1]-sumY
    check_y &= pos[1] < field.goal_area_max[1]+sumY
    return check_x and check_y


def spin(robot_pos, ball_pos, ball_speed):
    field = VSSField()
    spinDirection = False
    if (robot_pos[1] > ball_pos[1]):
        spinDirection = False
    else:
        spinDirection = True
    if(ball_pos[0] > field.middle[0] - 10):
        if(ball_pos[1] > field.middle[1]):
            if(ball_pos[1] < robot_pos[1] and ball_pos[0] > robot_pos[0]):
                spinDirection = not spinDirection
        else:
            if(ball_pos[1] > robot_pos[1] and ball_pos[0] > robot_pos[0]):
                spinDirection = not spinDirection

    if (ball_pos[0] < 20):
        if (ball_pos[0] < robot_pos[0]):
            if (ball_pos[1] < field.middle[1]):
                spinDirection = False
            else:
                spinDirection = True

    if(robot_pos[0] > field.m_max[0] - 3.75):
        if(ball_pos[0] < robot_pos[0]):
            p1 = ball_pos
            p2 = (ball_pos[0] + ball_speed[0]*5, ball_pos[1] + ball_speed[1]*5)
            angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
            if(math.sin(angle) > 0):
                spinDirection = True
            elif(math.sin(angle) < 0):
                spinDirection = False

    if(spinDirection):
        return(-70, 70)
    else:
        return (70, -70)


def is_near_to_wall(position, alpha):
    field = VSSField()
    res = 0
    if(position[1] <= field.m_min[1]+3.75*alpha and res != 3):
        res = 1
    if(position[1] >= field.m_max[1]-3.75*alpha and res != 3):
        res = 2
    if (position[1] >= field.goal_min[1] + (3.75*alpha)
            and position[1] <= field.goal_max[1] - (3.75*alpha)):
        pass
    else:
        if (position[0] <= field.m_min[0] + (3.75*alpha)):
            res = 4
        if (position[0] >= field.m_max[0] - (3.75*alpha)):
            res = 3
    return res


def point_in_polygon(p, q):
    c = False
    p_size = len(p)
    for i in range(p_size):
        j = (i + 1) % p_size
        if((p[i][1] <= q[1] and q[1] < p[j][1]
            or p[j][1] <= q[1] and q[1] < p[i][1])
           and q[0] < p[i][0] + (p[j][0] - p[i][0])
           * (q[1] - p[i][1]) / (p[j][1] - p[i][1])):
            c = not c

    return c

# Base on baselines implementation


class OrnsteinUhlenbeckAction(object):
    def __init__(self, action_space, theta=.17, dt=0.032, x0=None):
        self.theta = theta
        self.mu = (action_space.high + action_space.low) / 2
        self.sigma = (action_space.high - self.mu) / 2
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
