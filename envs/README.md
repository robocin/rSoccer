# RoboSim VSSS and SSL gym environments

## Install environments

```bash
$ pip install -e .
```
# Available Envs
## VSS
- **VSS-v0**
![Alt Text](resources/vss.gif)

- **VSSFIRA-v0** [Needs to run with FIRASIm]
- **VSSMA-v0**
- **VSSMAOpp-v0** [Needs a attacker model trained on VSS-v0]
- **VSSGk-v0** [Needs a attacker model trained on VSS-v0]
## SSL
- **SSLGoToBall-v0**
- **SSLGoToBall-v1**
- **SSLGoToBallIR-v0**
- **SSLGoToBallIR-v1**
![Alt Text](resources/gotoball.gif)

- **SSLGoToBallShoot-v0**
- **SSLGoToBallShoot-v1**
- **SSLGoToBallShoot-v2**
- **SSLGoToBallShoot-v3**
![Alt Text](resources/gotoshoot.gif)

- **SSLHWStaticDefenders-v0**
![Alt Text](resources/static.gif)

- **SSLHWDribbling-v0**
![Alt Text](resources/dribbling.gif)

- **SSLContestedPossessionEnv-v0**
![Alt Text](resources/contested_possession.gif)

- **SSLPassEndurance-v0**
![Alt Text](resources/pass_endurance.gif)

# Example code
```python
import gym
import rc_gym

# Using VSS 3v3 env
env = gym.make('VSS3v3-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
    print(reward)
```
