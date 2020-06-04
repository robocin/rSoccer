from gym.envs.registration import register
from envs.gym_soccer.vss_soccer_sim import SimSoccerEnv
from envs.gym_soccer.vss_soccer_sim_continuous import SimSoccerContinuousEnv
from envs.gym_soccer.vss_soccer_sim_continuous_ma import SimSoccerContinuousMultiAgentEnv
from envs.single_agent_soccer_env_wrapper import SingleAgentSoccerEnvWrapper

register(
    id='vss_soccer-v0',
    entry_point='envs.gym_soccer.vss_soccer_sim:SimSoccerEnv'
)

register(
    id='vss_soccer_cont-v0',
    entry_point='envs.gym_soccer.vss_soccer_sim_continuous:SimSoccerContinuousEnv'
)

register(
    id='vss_soccer_cont_ma-v0',
    entry_point='envs.gym_soccer.vss_soccer_sim_continuous_ma:SimSoccerContinuousMultiAgentEnv'
)

register(
    id='vss_soccer_agent-v0',
    entry_point='envs.single_agent_soccer_env_wrapper:SingleAgentSoccerEnvWrapper'
)
