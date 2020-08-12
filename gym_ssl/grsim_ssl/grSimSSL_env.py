import gym


class GrSimSSLEnv(gym.Env):

    def __init__(self):
        print('Environment initialized')
    def step(self):
        print('step sucessful!')
    def reset(self):
        print('Environment reset')
