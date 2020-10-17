# AgentDDPG
Based on [RL-Adventure-2 Github Repository](https://github.com/higgsfield/RL-Adventure-2)
## How to run
### Training
```bash
../rlAdventure2DDPG $ conda activate env
../rlAdventure2DDPG $ python agentDDPG.py {name} train
```
- Training checkpoints will be save in **./runs/_{name}_/**
- If checkpoint already exists in **./runs/_{name}_/** it will resume training

### Playing
```bash
../rlAdventure2DDPG $ conda activate env
../rlAdventure2DDPG $ python agentDDPG.py {name} play
```
- Uses trained network if checkpoint exists in **./runs/_{name}_/**
- Play with random network if no checkpoint is found


# Requirements
\#TODO