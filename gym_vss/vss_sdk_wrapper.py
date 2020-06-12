import gym
from gym_vss.single_agent_soccer_env_wrapper import SingleAgentSoccerEnvWrapper

ENVIRONMENT_PARAMS = {
    'env_name': "vss_soccer_cont-v0",
    'run_name': '%',  # '%' in this field sets the run_name to the parameter's file name
    'n_agents': 1,

    # simulator config:
    'ip': '127.0.0.1',  # real: '224.5.23.2'
    'port': 5555,  # real: 10006
    'ia': False,
    'init_agents': 0,  # 0: default positions, 1: random positions, 2: random base positions, 3: one agent, 4: goal_keeper, 5: penalty left, 6: penalty right
    'init_ball': 2,  # 0: stopped at the center, 1: random slow, 2: towards left goal, 3: towards right goal, 4: towards a random goal
    'random_cmd': False,
    'random_init': True,
    'is_team_yellow': True,  # True here requires ia = False
    'frame_step_size': (1000 / 60.0),
    'frame_skip': 1,
    # 5min in ms, to disable time limit set time_limit_ms < 0
    'time_limit_ms':  (5 * 60 * 1000),

    # Robot configuration
    'robot_center_wheel_dist': 4.88,
    'robot_wheel_radius': 2.0,

    # actions (this parameter is optional). None for 2D continuous space:
    'actions': None,
    'range_linear': 75.,   # linear speed range
    'range_angular': 15.,  # angular speed range

    # reward shaping/scaling parameters (per step):
    'w_rescaling': 10,
    'w_goal': (1.0, 1.0),  # (pro, against)
    'w_move': (0.02, 1),  # (-1,1)
    'w_ball_grad': (0.08, 1),  # (-1,1)
    'w_collision': (0.000, 1),  # {-1,0}
    'w_ball_pot': (0.0, 1),  # (-1,0)
    'w_energy': (1e-5, 1),  # (-1,0)
    'min_dist_move_rw': 0.0,  # cm
    'collision_distance': 12,
    'alpha_base': [0.25, 0.5, 0.6],

    # Tracking and log
    'track_rewards': [True, False, False],
}


class SDKWrapper:

    def __init__(self, env):
        self.env = SingleAgentSoccerEnvWrapper(env)
        self.env.setup(0, 1, False)
        self.env.soccer_env.set_parameters(ENVIRONMENT_PARAMS)

    def change_params(self, new_params):
        '''
        Changes environment parameters of your choice
        You always have to reset your environment after calling
        this method.

        Params -
            new_params: dict with which parameters you want to update

        Usage - 
            env.change_params({'frame_skip': 5})
            env.reset()
        '''
        ENVIRONMENT_PARAMS.update(new_params)
        self.env.soccer_env.set_parameters(ENVIRONMENT_PARAMS)

    def stop(self):
        self.env.stop()

    def reset(self):
        return self.env.reset()

    def step(self, command):
        return self.env.step(command)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, mode='human', close=False):
        self.env.render(mode, close)
