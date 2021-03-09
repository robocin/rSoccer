from gym.envs.registration import register

register(id='VSS3v3-v0',
         entry_point='rc_gym.vss.env_3v3:VSS3v3Env'
         )

register(id='VSSMA-v0',
         entry_point='rc_gym.vss.env_ma:VSSMAEnv',
         )

register(id='VSSMAOpp-v0',
         entry_point='rc_gym.vss.env_ma:VSSMAOpp',
         )

register(id='VSSGk-v0',
         entry_point='rc_gym.vss.env_gk:rSimVSSGK'
         )

register(id='VSS3v3FIRA-v0',
         entry_point='rc_gym.vss.env_3v3:VSS3v3FIRAEnv'
         )
