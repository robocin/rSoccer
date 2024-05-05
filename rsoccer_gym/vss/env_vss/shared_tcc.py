import math
import numpy as np

distancia = lambda o1, o2: np.sqrt((o1.x - o2.x)**2 + (o1.y - o2.y)**2)


def close_to_x(x, range=.15):
    return np.clip(x + np.random.uniform(-range,range,1)[0], -.5, .5)

def close_to_y(x, range=.15):
    return np.clip(x + np.random.uniform(-range,range,1)[0], -.5, .5)

def menor_angulo(v1, v2):
    angle = math.acos(np.dot(v1,v2))

    if np.cross(v1,v2) > 0:
        return -angle

    return angle

def transform(v1, ang):
    mod = np.sqrt(v1[0]**2 + v1[1]**2)
    v1 = (v1[0]/mod, v1[1]/mod)

    mn = menor_angulo(v1, (math.cos(ang), math.sin(ang)))

    return mn, (math.cos(mn)*mod, math.sin(mn)*mod)

def observations(self):
    observation = []

    max_comprimento = self.field.length + self.field.penalty_length
    max_altura = self.field.width

    vetor_x_gol_oponente = (max_comprimento/2 - self.frame.robots_blue[0].x) / max_comprimento
    vetor_y_gol_oponente = (0 - self.frame.robots_blue[0].y) / (max_altura/2)

    # de cima pra baixo verde e azul -> 4,71 rad -> 270 graus
    # significa que azul e verde e 90 graus
    # azul e a frente do robo, quando alinhada no eixo x ela e 0 graus
    ang = np.deg2rad(self.frame.robots_blue[0].theta)

    angle, (v1_x, v1_y) = transform((vetor_x_gol_oponente, vetor_y_gol_oponente), ang)

    distance_rg = np.sqrt(v1_x * v1_x + v1_y * v1_y)

    observation.append(distance_rg) # vetor robo -> gol oponente
    observation.append(angle/math.pi) # vetor robo -> gol oponente

    # observacao para bola
    vetor_x_bola = (self.frame.ball.x - self.frame.robots_blue[0].x) / max_comprimento
    vetor_y_bola = (self.frame.ball.y - self.frame.robots_blue[0].y) / max_altura

    angle, (v1_x, v1_y) = transform((vetor_x_bola, vetor_y_bola), ang)
    distance_rb = np.sqrt(v1_x * v1_x + v1_y * v1_y) # vetor robo -> bola
    
    observation.append(distance_rb)
    observation.append(angle/math.pi) # vetor robo -> bola


    # observação do inimigo 1 para o goleiro
    for i in range(self.n_robots_yellow):
        amigo_x = (self.frame.robots_yellow[i].x - self.frame.robots_blue[0].x) / max_comprimento
        amigo_y = (self.frame.robots_yellow[i].y - self.frame.robots_blue[0].y) / max_altura

        angle, (v1_x, v1_y) = transform((amigo_x, amigo_y), ang)
        distance_amigo = np.sqrt(v1_x * v1_x + v1_y * v1_y)
        
        observation.append(distance_amigo)
        observation.append(angle/math.pi)

    # angulo dele
    observation.append(np.cos(ang))
    observation.append(np.sin(ang))

    #velocidades da bola
    observation.append(self.norm_v(self.frame.ball.v_x))
    observation.append(self.norm_v(self.frame.ball.v_y))


    # velocidades dele
    observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
    observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
    observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))

    # print("".join([str(round(x,2)).ljust(6) for x in observation]),end='')
    return np.array(observation, dtype=np.float32)

w_move_tcc = .25
w_ball_grad_tcc = 2
w_energy_tcc = 1e-3
goal_reward = 500
own_goal_reward = 100

def observations_da(self):
    observation = []

    observation.append(self.norm_pos(self.frame.ball.x))
    observation.append(self.norm_pos(self.frame.ball.y))
    observation.append(self.norm_v(self.frame.ball.v_x))
    observation.append(self.norm_v(self.frame.ball.v_y))

    observation.append(self.norm_pos(self.frame.robots_blue[0].x))
    observation.append(self.norm_pos(self.frame.robots_blue[0].y))
    observation.append(
        np.sin(np.deg2rad(self.frame.robots_blue[0].theta))
    )
    observation.append(
        np.cos(np.deg2rad(self.frame.robots_blue[0].theta))
    )
    observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
    observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
    observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))

    return np.array(observation, dtype=np.float32)