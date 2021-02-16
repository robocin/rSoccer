import math

import numpy as np
from rc_gym.Utils import clip, smallestAngleDiff, to_pi_range


class PID:
    def __init__(self):
        self.pidLastAngularError = 0.0
        self.lastDistanceError = 0.0
        self.kp = 20.0
        self.kd = 2.5
        self.baseSPD = 5.0
        self.dbaseSPD = 0.0

    def run(self, angle_rob, obj_pos, robot_pos):

        reverse = False

        angle_obj = math.atan2(
            obj_pos[1] - robot_pos[1], obj_pos[0] - robot_pos[0])

        error = smallestAngleDiff(angle_rob, angle_obj)

        reverse_angle_rob = to_pi_range(angle_rob+math.pi)

        if math.fabs(error) > (math.pi/2.0 + math.pi/20.0):
            reverse = True
            angle_rob = reverse_angle_rob
            error = smallestAngleDiff(angle_rob, angle_obj)

        motorSpeed = self.kp*error + \
            (self.kd * (error - self.pidLastAngularError))
        self.pidLastAngularError = error

        distance = np.linalg.norm(obj_pos - robot_pos)
        baseSpeed = (self.baseSPD * distance) + self.dbaseSPD * \
            (distance - self.lastDistanceError)
        self.lastDistanceError = distance

        baseSpeed = clip(baseSpeed, 0, 45)

        motorSpeed = clip(motorSpeed, -30, 30)

        if motorSpeed > 0:
            leftMotorSpeed = baseSpeed
            rightMotorSpeed = baseSpeed - motorSpeed
        else:
            leftMotorSpeed = baseSpeed + motorSpeed
            rightMotorSpeed = baseSpeed

        if reverse:
            if motorSpeed > 0:
                motorSpeed = -baseSpeed + motorSpeed
                rightMotorSpeed = -baseSpeed
            else:
                leftMotorSpeed = -baseSpeed
                rightMotorSpeed = - baseSpeed - motorSpeed

        return np.array([leftMotorSpeed, rightMotorSpeed])
