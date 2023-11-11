import gym
import rsoccer_gym
import numpy as np

env = gym.make("VSS-v0")
for i in range(10):
    env.reset()
    env.render()
    done = False
    rew = 0
    while not done:
        # action = np.array(
        #     [
        #         # 0,
        #         1,
        #         1,
        #         np.sin(env.target_angle),
        #         np.cos(env.target_angle),
        #         # env.target_velocity.x / 2.5,
        #         # env.target_velocity.y / 2.5,
        #     ]
        # )
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        rew += reward
        env.render()
    print(rew)
