# RoboSim SSL and VSSS gym environments
IEE VSSS                   |  IEE VSSS Multi-Agent     |        GoTo Ball          | Static Defenders          |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
![](resources/vss.gif)     | ![](resources/vss_ma.gif) | ![](resources/gotoball.gif) | ![](resources/static.gif)     |

  Contested Possession     |        Dribbling          |  Pass Endurance     |        Pass Endurance MA          |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:| 
 ![](resources/contested_possession.gif) | ![](resources/dribbling.gif)|![](resources/pass_endurance.gif) | ![](resources/pass_endurance_ma.gif)|
## Install environments

```bash
$ pip install -e .
```
# Available Envs

|       Environment Id                                                       | Observation Space | Action Space | Step limit |
|:--------------------------------------------------------------------------:|:-----------------:|:------------:|:----------:|
|[VSS-v0](rc_gym/vss/README.md#vss-v0)                                       |      Box(40,)     |    Box(2,)   |    1200    |
|[VSSMA-v0](rc_gym/vss/README.md#vssma-v0)                                   |      Box(3,40)    |    Box(3,2)  |    1200    |
|[VSSGk-v0](rc_gym/vss/README.md#vssgk-v0)                                   |      Box(40,)     |    Box(2,)   |    1200    |
|[SSLGoToBall-v0](rc_gym/ssl/README.md#sslgotoball-v0)                   |      Box(24,)     |    Box(3,)   |    2400        |
|[SSLGoToBallShoot-v0](rc_gym/ssl/README.md#sslgotoballshoot-v0)             |      Box(12,)     |    Box(5,)   |    1200    |
|[SSLStaticDefenders-v0](rc_gym/ssl/README.md#sslstaticdefenders-v0)         |      Box(24,)     |    Box(5,)   |    1000    |
|[SSLDribbling-v0](rc_gym/ssl/README.md#ssldribbling-v0)                     |      Box(21,)     |    Box(4,)   |    4800    |
|[SSLContestedPossession-v0](rc_gym/ssl/README.md#sslcontestedpossession-v0) |      Box(14,)     |    Box(5,)   |    1200    |
|[SSLPassEndurance-v0](rc_gym/ssl/README.md#sslpassendurance-v0)             |      Box(18,)     |    Box(3,)   |    1200    |
|[SSLPassEnduranceMA-v0](rc_gym/ssl/README.md#sslpassendurancema-v0)         |      Box(18,)     |    Box(2,3)  |    1200    |
# Example code
```python
import gym
import rc_gym

# Using VSS 3v3 env
env = gym.make('VSS-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
    print(reward)
```
