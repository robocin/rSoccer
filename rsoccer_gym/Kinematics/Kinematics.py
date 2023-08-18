import numpy as np

M_TO_MM = 1/1000

class Wheel:
    def __init__(self,
                 radius = 0,
                 distance_to_axis = 0,
                 alpha = 0,
                 beta = 0,
                 mm_deviation = 0,
                 angle_deviation = 0):
        self.build_wheel(radius, 
                         distance_to_axis, 
                         alpha,
                         beta,
                         mm_deviation,
                         angle_deviation)

    def build_wheel(self, radius, distance_to_axis, alpha, beta, mm_deviation, angle_deviation):
        self.radius = np.random.normal(radius, mm_deviation*M_TO_MM, 1)
        self.distance_to_axis = np.random.normal(distance_to_axis, mm_deviation*M_TO_MM, 1)
        self.alpha = np.random.normal(alpha, angle_deviation, 1)
        self.beta = np.random.normal(beta, angle_deviation, 1)

    def get_constraint_to_kinematics(self):
        return np.array([-np.sin(np.deg2rad(self.alpha + self.beta)), 
                         np.cos(np.deg2rad(self.alpha + self.beta)), 
                         np.cos(np.deg2rad(self.beta))*self.distance_to_axis])

class Robot:
    def __init__(self,
                 number_of_wheels = 4,
                 id = 0,
                 wheel_radius = 0.02475,
                 axis_length = 0.081,
                 wheels_alphas = [0, 0, 0, 0],
                 wheels_betas = [0, 0, 0, 0],
                 mm_deviation = 0,
                 angle_deviation = 0):
        
        self.id = id
        self.wheels = []
        self.number_of_wheels = len(wheels_alphas)
        for alpha, beta in zip(wheels_alphas, wheels_betas):
            wheel = Wheel(wheel_radius, 
                        axis_length, 
                        alpha,
                        beta,
                        mm_deviation,
                        angle_deviation)
            self.wheels.append(wheel)

    def get_J1(self):
        J1 = []
        for wheel in self.wheels:
            J1_line = wheel.get_constraint_to_kinematics()
            J1.append(J1_line)
        
        return np.squeeze(J1)
    
    def get_J2(self):
        J2_diagonal = []
        for wheel in self.wheels:
            J2_diagonal.append(wheel.radius)
        J2 = np.diag(np.squeeze(J2_diagonal))

        return J2
    
    def get_J1_inv(self):
        J1 = self.get_J1()
        J1_inv = np.linalg.pinv(J1)
        return J1_inv

    def get_J2_inv(self):
        J2 = self.get_J2()
        J2_inv = np.linalg.inv(J2)
        return J2_inv

if __name__ == "__main__":

    robot_sim = Robot(number_of_wheels = 4,
                      wheel_radius = 0.02475,
                      axis_length = 0.081,
                      wheels_alphas = [60, 135, -135, -60],
                      wheels_betas = [0, 0, 0, 0],
                      mm_deviation = 0,
                      angle_deviation = 0)
    
    robot_real = Robot(number_of_wheels = 4,
                       wheel_radius = 0.025,
                       axis_length = 0.081,
                       wheels_alphas = [60, 135, -135, -60],
                       wheels_betas = [0, 0, 0, 0],
                       mm_deviation = 0,
                       angle_deviation = 0)

    # VELOCITY COMMAND
    vx = 1
    vy = 0
    w = 0
    v = np.array([vx, vy, w])

    # WHEELS' DESIRED ROTATION
    phi = robot_sim.get_J2_inv()@robot_sim.get_J1()@v
    print(f'Desired wheel rotations in rad/s: {phi}')
    
    # ROBOT MOVEMENT
    v_real = robot_real.get_J1_inv()@robot_real.get_J2()@phi
    print(f'Real robot velocities in m/s: {v_real}')

