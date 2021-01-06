from gym.envs.registration import register

register(id='grSimSSLPenalty-v0',
         entry_point='rc_gym.grsim_ssl:GrSimSSLPenaltyEnv'
         )

register(id='grSimSSLShootGoalie-v0',
         entry_point='rc_gym.grsim_ssl:shootGoalieEnv'
         )

register(id='grSimSSLGoToBall-v0',
         entry_point='rc_gym.grsim_ssl:goToBallEnv'
         )

register(id='VSS3v3-v0',
         entry_point='rc_gym.vss:VSS3v3Env'
         )

register(id='VSSReach-v0',
         entry_point='rc_gym.vss:VSSReachEnv'
         )
