# ssl-env

# install
```bash
$ pip install -e .
```

# Test code
```python
import gym
import gym_ssl

env = gym.make('grSimSSL-v0')
env.step()
env.reset()
```