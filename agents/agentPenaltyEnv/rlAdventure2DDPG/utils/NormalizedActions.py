import gym

import numpy as np


class NormalizedWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        # Retrieve the observation space
        observation_space = env.observation_space

        assert isinstance(action_space,
                          gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
        assert isinstance(observation_space,
                          gym.spaces.Box), "This wrapper only works with continuous observation space (spaces.Box)"

        # Retrieve the max/min values
        self.actLow, self.actHigh = action_space.low, action_space.high
        self.obsLow, self.obsHigh = observation_space.low, observation_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)
        env.observation_space = gym.spaces.Box(low=-1, high=1, shape=observation_space.shape, dtype=np.float32)

        # Call the parent constructor, so we can access self.env later
        super(NormalizedWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.actLow + (0.5 * (scaled_action + 1.0) * (self.actHigh - self.actLow))

    def scale_observation(self, observation):
        """
        Scale the observation to bounds [-1, 1]
        """
        return (2 * ((observation - self.obsLow) / (self.obsHigh - self.obsLow))) - 1

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


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return actions