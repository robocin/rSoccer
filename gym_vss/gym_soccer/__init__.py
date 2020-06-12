from gym.envs.registration import register
from gym_vss.gym_soccer.vss_soccer_sim import SimSoccerEnv
from gym_vss.gym_soccer.vss_soccer_sim_continuous import SimSoccerContinuousEnv
from gym_vss.gym_soccer.vss_soccer_sim_continuous_ma import SimSoccerContinuousMultiAgentEnv

register(
    id='vss_soccer-v0',
    entry_point='gym_vss.gym_soccer.vss_soccer_sim:SimSoccerEnv'
)

register(
    id='vss_soccer_cont-v0',
    entry_point='gym_vss.gym_soccer.vss_soccer_sim_continuous:SimSoccerContinuousEnv'
)

register(
    id='vss_soccer_cont_ma-v0',
    entry_point='gym_vss.gym_soccer.vss_soccer_sim_continuous_ma:SimSoccerContinuousMultiAgentEnv'
)
