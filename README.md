# RoboCup Very Small Size League Gym Environment
This Environment is used in [RobôCIn's](https://github.com/robocin/deepvss) project. Check for some Reinforcement Learning Techniques applied for the environment. 

## Project based on:
* [VSS-SDK/VSS-Simulator & VSS-Viewer](https://github.com/VSS-SDK/VSS-Simulator) (x64 binaries provided here)
* [OpenAI Gym Environments](https://github.com/openai/gym)

# Requirements

- FIRASim
    - Clone the repo
        - https://github.com/robocin/FIRASim/releases/tag/deep_train
    - Follow it's install guide
        - https://github.com/robocin/FIRASim/blob/master/INSTALL.md
    - Once builded, change the binary file in gym_vss/binaries_envs/fira_sim/bin

- VSS SDK
    - Add the following line to your ~/.bashrc or ~/.bash_profile
    - source /home/$USER/path/to/envs/gym_vss/binaries_envs/vss_sdk/exportlibs

FIRASim requires much more and heavy libs linked than SDK, that's why we couldn't do as in SDK guide.

# Install
First change the 5th line of gym_vss/gym_real_soccer/comm/Makefile to your python include path.
If you are using anaconda, you need to change only the python version in the path.
```bash
$ sudo apt-get install swig freeglut3-dev -y
$ cd gym_vss/gym_real_soccer/comm
$ make
$ cd ../../../
$ pip install -e .
```

# Example of agent
```python
import gym
import gym_vss

from gym_vss import SingleAgentSoccerEnvWrapper


env = gym.make('vss_soccer_cont-v0')
# env = SingleAgentSoccerEnvWrapper(env, simulator='sdk')
# If you want FIRASim
env = SingleAgentSoccerEnvWrapper(env, simulator='fira')
env.reset()
for i in range(1):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
env.close()
```


