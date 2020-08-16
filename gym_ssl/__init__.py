from gym.envs.registration import register

register(id='grSimSSL-v0',
    entry_point='gym_ssl.grsim_ssl:GrSimSSLPenaltyEnv'
    )
