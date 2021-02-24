import math

import numpy as np
from rc_gym.Utils import is_near_to_wall, smallestAngleDiff
from rc_gym.Utils import clip as bound
from rc_gym.Utils import to_pi_range as to180range
from rc_gym.Utils import distance as euclideanDistance


class PID:
    def __init__(self):
        self.pidLastAngularError = 0.0
        self.lastDistanceError = 0.0
        self.kp = 36.0  # 20#36.0
        self.kd = 4.2  # 2.5#4.2
        self.baseSPD = 5.8  # 5#5.8
        self.dbaseSPD = 0.0
        self.motors = (0, 0)
        self.oldRobotPos = (0, 0)
        self.steps = 0
        self.stepsExit = 0
        self.stuck = False

    def run(self, angle_rob, obj_pos, robot_pos):

        reverse = False

        angle_obj = math.atan2(
            obj_pos[1] - robot_pos[1], obj_pos[0] - robot_pos[0])

        error = smallestAngleDiff(angle_rob, angle_obj)

        aux = [math.cos(angle_rob)*5+robot_pos[0],
               math.sin(angle_rob)*5+robot_pos[1]]

        reverse_angle_rob = to180range(angle_rob+math.pi)

        aux2 = [math.cos(reverse_angle_rob)*5+robot_pos[0],
                math.sin(reverse_angle_rob)*5+robot_pos[1]]

        if(is_near_to_wall(aux, 0.5) != 0):
            reverse = True
            error = smallestAngleDiff(reverse_angle_rob, angle_obj)
        elif(is_near_to_wall(aux2, 0.5) != 0):
            error = smallestAngleDiff(angle_rob, angle_obj)
        else:
            if math.fabs(error) > (math.pi/2.0 + math.pi/20.0):
                reverse = True
                angle_rob = reverse_angle_rob
                error = smallestAngleDiff(angle_rob, angle_obj)

        motorSpeed = self.kp*error + \
            (self.kd * (error - self.pidLastAngularError))
        self.pidLastAngularError = error

        distance = euclideanDistance(obj_pos, robot_pos)
        baseSpeed = (self.baseSPD * distance) + self.dbaseSPD * \
            (distance - self.lastDistanceError)
        self.lastDistanceError = distance

        if math.fabs(math.fabs(error) - (math.pi/2)) < math.pi/12:
            baseSpeed = 0
        else:
            baseSpeed = bound(baseSpeed, 0, 45)

        motorSpeed = bound(motorSpeed, -30, 30)

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

        self.motors = (leftMotorSpeed, rightMotorSpeed)
        self.oldRobotPos = robot_pos
        print(self.motors)

        return np.array(self.motors)*0.02
