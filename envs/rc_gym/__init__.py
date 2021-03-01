from gym.envs.registration import register

register(id='VSS3v3-v0',
         entry_point='rc_gym.vss:VSS3v3Env'
         )

register(id='VSSMA-v0',
         entry_point='rc_gym.vss:VSSMAEnv',
         )

register(id='VSSMAOpp-v0',
         entry_point='rc_gym.vss:VSSMAOpp',
         )
