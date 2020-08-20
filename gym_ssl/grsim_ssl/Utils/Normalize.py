import numpy as np

class NormalizedObservation():
    def __init__(self, observationSpace):
        self.observationSpace = observationSpace

    def _observation(self, observation):
        low_bound   = self.observationSpace.low
        upper_bound = self.observationSpace.high
        
        observation = low_bound + (observation + 1.0) * 0.5 * (upper_bound - low_bound)
        observation = np.clip(observation, low_bound, upper_bound)
        
        return observation

    def _reverse_observation(self, observation):
        low_bound   = self.observationSpace.low
        upper_bound = self.observationSpace.high
        
        observation = 2 * (observation - low_bound) / (upper_bound - low_bound) - 1
        observation = np.clip(observation, low_bound, upper_bound)
        
        return observation


class NormalizedActions():
    def __init__(self, action_space):
        self.action_space = action_space

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
        
        return actions