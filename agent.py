import gym
import numpy as np
import rsoccer_gym    

def get_action_from_desired_global_velocity(env, vx, vy, w):
    vx = vx/env.max_v
    vy = vy/env.max_v
    w = w/env.max_w
    return np.array([vx, vy, w], dtype=np.float32)

# Using SSL Single Agent env
env = gym.make('SSLGoToBall-v0', 
               n_robots_blue=7, 
               n_robots_yellow=0,
               mm_deviation=0.2,
               angle_deviation=0.1)

env.reset()

# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        # Step using random actions
        action = get_action_from_desired_global_velocity(env, 0.5, 0, 0)
        next_state, reward, done, _ = env.step(action)
        env.render()
