from gym.envs.registration import register
from gym_vss.gym_real_soccer.vss_soccer_real import RealSoccerEnv
from gym_vss.gym_real_soccer.vss_soccer_real_continuous import RealSoccerContinuousEnv

register(
    id='vss_soccer_real-v0',
    entry_point='gym_vss.gym_real_soccer:RealSoccerEnv'
)

register(
    id='vss_soccer_real_continuous-v0',
    entry_point='gym_vss.gym_real_soccer:RealSoccerContinuousEnv'
)