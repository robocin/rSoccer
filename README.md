# Robocup SSL OpenAi gym environments

# Requirements
## Compile protobuf files
```bash
$ sudo apt-get install libprotobuf-dev protobuf-compiler -y
$ cd gym_ssl/grsim_ssl/Communication/pb/proto
$ protoc --python_out=../ *.proto
```
## Fix protobuf compiled files from relative reference to absolute
On file **gym_ssl/grsim_ssl/Communication/pb/messages_robocup_ssl_wrapper_pb2.py**:


``` python
'before:'
15 - import messages_robocup_ssl_detection_pb2 as messages__robocup__ssl__detection__pb2
16 - import messages_robocup_ssl_geometry_pb2 as messages__robocup__ssl__geometry__pb2

'after:'
15 + import gym_ssl.grsim_ssl.Communication.pb.messages_robocup_ssl_detection_pb2 as messages__robocup__ssl__detection__pb2
16 + import gym_ssl.grsim_ssl.Communication.pb.messages_robocup_ssl_geometry_pb2 as messages__robocup__ssl__geometry__pb2
```

On file **gym_ssl/grsim_ssl/Communication/pb/grSim_Packet_pb2.py**:

``` python
'before:'
15 - import grSim_Commands_pb2 as grSim__Commands__pb2
16 - import grSim_Replacement_pb2 as grSim__Replacement__pb2
'after:'
15 + import gym_ssl.grsim_ssl.Communication.pb.grSim_Commands_pb2 as grSim__Commands__pb2
16 + import gym_ssl.grsim_ssl.Communication.pb.grSim_Replacement_pb2 as grSim__Replacement__pb2
```
## Install environments

```bash
$ pip install -e .
```
# Available Envs
- **grSimSSLPenalty-v0**

# Example code
```python
import gym
import gym_ssl

# Using penalty env
env = gym.make('grSimSSLPenalty-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
    print(reward)
```