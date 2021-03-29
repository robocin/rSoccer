import gym
import rc_gym

# Using penalty env
env = gym.make('VSS-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
    print(reward)