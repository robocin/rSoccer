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

register(id='SSLGoToBall-v0',
         entry_point='rc_gym.ssl.ssl_go_to_ball:SSLGoToBallEnv',
         kwargs={'field_type' : 1, 'n_robots_yellow' : 0}
         )

register(id='SSLGoToBall-v1',
         entry_point='rc_gym.ssl.ssl_go_to_ball:SSLGoToBallEnv',
         kwargs={'field_type' : 1, 'n_robots_yellow' : 6}
         )
