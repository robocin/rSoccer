from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
import rc_gym
import gym
import numpy as np
from tensorboardX import SummaryWriter
import time
env_name = 'VSS3v3-v0'
n_procs = 5
TRAIN_STEPS = 200000

class NormalizedWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(NormalizedWrapper, self).__init__(env)

        assert isinstance(self.env.action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        assert isinstance(self.env.observation_space,
                          gym.spaces.Box), "This wrapper only works with continuous observation space (spaces.Box)"

        # We modify the wrapper action space, so all actions will lie in [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.env.action_space.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.env.observation_space.shape, dtype=np.float32)


    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.env.action_space.low + (
                0.5 * (scaled_action + 1.0) * (self.env.action_space.high - self.env.action_space.low))

    def scale_observation(self, observation):
        """
        Scale the observation to bounds [-1, 1]
        """
        return (2 * ((observation - self.env.observation_space.low) /
                     (self.env.observation_space.high - self.env.observation_space.low))) - 1

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        return self.scale_observation(self.env.reset())

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, done, info = self.env.step(rescaled_action)
        
        return self.scale_observation(obs), reward, done, info
    
writer = SummaryWriter(log_dir="log/ppo", comment="-agent")

eval_env = gym.make(env_name)
train_env = make_vec_env(rc_gym.vss.env_3v3.vss_gym_3v3.VSS3v3Env, n_envs=n_procs, wrapper_class=NormalizedWrapper, vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': 'fork'})
model = PPO('MlpPolicy', train_env, verbose=0)
try:
    steps = 0
    while True:
        train_env.reset()
        mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print("steps {}, evaluation mean_rewards: {:.6f}".format(steps, mean_reward))
        writer.add_scalar("eval/mean_rewards", mean_reward, steps)
        seconds_start = time.perf_counter()
        model.learn(total_timesteps=TRAIN_STEPS, )
        writer.add_scalar("learn/steps_s", TRAIN_STEPS / (time.perf_counter() - seconds_start), steps)
        model.save("model/ppo")
        steps += TRAIN_STEPS
except KeyboardInterrupt:
    train_env.close()
    eval_env.close()