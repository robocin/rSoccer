# RoboCup Very Small Size League Gym Environment
This Environment is used in [RobôCIn's](https://github.com/robocin/deepvss) project. Check for some Reinforcement Learning Techniques applied for the environment. 

## Project based on:
* [VSS-SDK/VSS-Simulator & VSS-Viewer](https://github.com/VSS-SDK/VSS-Simulator) (x64 binaries provided here)
* [OpenAI Gym Environments](https://github.com/openai/gym)

# Install
```bash
$ pip install -e .
```

# Example of agent
```python
import gym
import gym_vss

from gym_vss import SingleAgentSoccerEnvWrapper

n_agents = 1
random_cmd = False
env = gym.make('vss_soccer_cont-v0')
env = SingleAgentSoccerEnvWrapper(env, simulator='sdk')
# If you want FIRASim
# env = SingleAgentSoccerEnvWrapper(env, simulator='fira')
env.setup(0, n_agents, random_cmd)
env.reset()
for i in range(100):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
env.close()
```


