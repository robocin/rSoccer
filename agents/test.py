import gym
import rc_gym
import PIL


# Using penalty env
env = gym.make('SSLPassEndurance-v0')

# Run for 1 episode and print reward at the end
for i in range(1300):
    done = False
    env.reset()
    for _ in range(1200):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
