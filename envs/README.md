# RoboSim VSSS and SSL gym environments

## Install environments

```bash
$ pip install -e .
```
# Available Envs
- **VSS3v3-v0**
- **VSS3v3FIRA-v0**
    - Needs to run with FIRASIm
- **VSSMA-v0**
- **VSSMAOpp-v0**
    - Needs a attacker model trained on VSS3v3-v0
- **VSSGk-v0**
    - Needs a attacker model trained on VSS3v3-v0

# Example code
```python
import gym
import rc_gym

# Using penalty env
env = gym.make('VSS3v3-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
    print(reward)
```
