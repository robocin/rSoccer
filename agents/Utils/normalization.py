import gym

import numpy        as np

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


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def _reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return action