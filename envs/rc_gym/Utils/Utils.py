import math
import numpy as np
import gym


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
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
