import math

from rc_gym.vss.env_coach.deterministic_vss import Field, utils


class PID:
	def __init__(self):
		self.pidLastAngularError = 0.0
		self.lastDistanceError = 0.0
		self.kp = 36.0#20#36.0
		self.kd = 4.2#2.5#4.2
		self.baseSPD = 5.8#5#5.8
		self.dbaseSPD = 0.0
		self.motors = (0,0)
		self.oldRobotPos = (0,0)
		self.steps = 0
		self.stepsExit = 0
		self.stuck = False

	def turn_in_place(self,angle_rob,angle_obj):
		reverse = False

		error = utils.smallestAngleDiff(angle_rob,angle_obj)

		reverse_angle_rob = utils.to180range(angle_rob+math.pi)

		if math.fabs(error) > (math.pi/2.0 + math.pi/20.0):
			reverse = True
			angle_rob = reverse_angle_rob
			error = utils.smallestAngleDiff(angle_rob,angle_obj)

		motorSpeed = self.kp*error + (self.kd * (error - self.pidLastAngularError))
		self.pidLastAngularError = error
		motorSpeed = utils.bound(motorSpeed,-30,30)

		
		leftMotorSpeed = motorSpeed
		rightMotorSpeed = -motorSpeed
		
		if reverse:
			motorSpeed = motorSpeed
			rightMotorSpeed = -motorSpeed
		
		return (leftMotorSpeed,rightMotorSpeed)
			

    
	def run(self,angle_rob,obj_pos,robot_pos):

		if(utils.euclideanDistance(robot_pos,obj_pos) < 4):
			return self.turn_in_place(angle_rob,math.pi/2)

		reverse = False

		angle_obj = math.atan2(obj_pos[1] - robot_pos[1], obj_pos[0] - robot_pos[0])


		error = utils.smallestAngleDiff(angle_rob,angle_obj)

		aux = [math.cos(angle_rob)*5+robot_pos[0], math.sin(angle_rob)*5+robot_pos[1]]

		reverse_angle_rob = utils.to180range(angle_rob+math.pi)

		aux2 = [math.cos(reverse_angle_rob)*5+robot_pos[0], math.sin(reverse_angle_rob)*5+robot_pos[1]]

		if(utils.isNearToWall(aux, 0.5)!=0):
			reverse = True
			error = utils.smallestAngleDiff(reverse_angle_rob,angle_obj)
		elif(utils.isNearToWall(aux2, 0.5)!=0):
			error = utils.smallestAngleDiff(angle_rob,angle_obj)
		else:
			if math.fabs(error) > (math.pi/2.0 + math.pi/20.0):
				reverse = True
				angle_rob = reverse_angle_rob
				error = utils.smallestAngleDiff(angle_rob,angle_obj)

		#print(self.motors)
		#if((abs(self.motors[0])>5.0 or abs(self.motors[1])>5.0) and utils.euclideanDistance(robot_pos, self.oldRobotPos)<0.02):
		#	self.steps +=1
		#	if(self.steps>30):
		#		self.stuck = True
		#		#print(self.steps)
		#		#print("preso ", reverse)
		#		reverse = not reverse
		#		angle_rob = utils.to180range(angle_rob+math.pi)
		#		error = utils.smallestAngleDiff(angle_rob,angle_obj)
		#		self.stepsExit = 0
		#	else:
		#			#print("nao preso ainda ", self.steps, reverse)
		#			pass
		#
		#else:
		#	if(self.stuck):
		#		reverse = not reverse
		#		angle_rob = utils.to180range(angle_rob+math.pi)
		#		error = utils.smallestAngleDiff(angle_rob,angle_obj)
		#		self.stepsExit +=1
		#		if(self.stepsExit>30):
		#			self.stuck=False
		#			self.stepsExit = 0
		#			self.steps = 0
		#		#print(self.stepsExit)
		#		#print("tentando sair")
		#	else:	
		#		self.stepsExit = 0
		#		self.stuck= False
		#		self.steps = 0
		#
				#print("normal")
				
		


		motorSpeed = self.kp*error + (self.kd * (error - self.pidLastAngularError))
		self.pidLastAngularError = error

		distance = utils.euclideanDistance(obj_pos,robot_pos)
		baseSpeed = (self.baseSPD * distance) + self.dbaseSPD*(distance - self.lastDistanceError)
		self.lastDistanceError = distance

		if math.fabs(math.fabs(error) - (math.pi/2)) < math.pi/12:
			baseSpeed = 0
		else:
			baseSpeed = utils.bound(baseSpeed,0,45)

		motorSpeed = utils.bound(motorSpeed,-30,30)

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
		#self.motors = (15,45)
		self.oldRobotPos = robot_pos

		return self.motors




