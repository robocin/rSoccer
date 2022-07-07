# This is the experiment made to obtain the maximum ball potential gradient in
# the VSS5v5Env. There's only one robot in the 5v5 field, which is carrying the
# ball from the left goal to the right goal

import gym
from rsoccer_gym.vss.env_vss.vss5v5_gym import VSS5v5Env


debug = dict()


if __name__ == '__main__':
    done = False
    info = None
    env = VSS5v5Env()
    # env = gym.make("VSS-v0")
    env.reset()
    env.render()
    left = 1
    while not done:
        _, _, done, info = env.step([1,1])
        env.render()
        input()
        # left += 0.05
    
    print(info)
    print(env.debug)

# Results:

# {'goal_score': 1, 'move': 42.35095445086161, 'ball_grad': 90.6138495367951, 'energy': -1.3823007675795078, 'goals_blue': 1, 'goals_yellow': 0}
# {'max_ball_grad': 1.6342460522826219}

# The value of interest is 'max_ball_grad', which will be applied soon to normalize the rewards in VSS5v5Env