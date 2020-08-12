# ssl-en

# Requirements
```bash
$ sudo apt-get install libprotobuf-dev protobuf-compiler -y
$ cd gym_ssl/grsim_ssl/pb/
$ protoc --python_out=../ *.proto
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