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
         kwargs={'field_type' : 2, 'n_robots_yellow' : 0}
         )

register(id='SSLGoToBall-v1',
         entry_point='rc_gym.ssl.ssl_go_to_ball:SSLGoToBallEnv',
         kwargs={'field_type' : 2, 'n_robots_yellow' : 6}
         )

register(id='SSLGoToBallIR-v0',
         entry_point='rc_gym.ssl.ssl_go_to_ball:SSLGoToBallIREnv',
         kwargs={'field_type' : 2, 'n_robots_yellow' : 0}
         )

register(id='SSLGoToBallIR-v1',
         entry_point='rc_gym.ssl.ssl_go_to_ball:SSLGoToBallIREnv',
         kwargs={'field_type' : 2, 'n_robots_yellow' : 6}
         )

register(id='SSLGoToBallShoot-v0',
         entry_point='rc_gym.ssl.ssl_go_to_ball_shoot:SSLGoToBallShootEnv',
         kwargs={'field_type' : 2, 'random_init' : False, 
                 'enter_goal_area' : False}
         )

register(id='SSLGoToBallShoot-v1',
         entry_point='rc_gym.ssl.ssl_go_to_ball_shoot:SSLGoToBallShootEnv',
         kwargs={'field_type' : 2, 'random_init' : True, 
                 'enter_goal_area' : False}
         )

register(id='SSLGoToBallShoot-v2',
         entry_point='rc_gym.ssl.ssl_go_to_ball_shoot:SSLGoToBallShootEnv',
         kwargs={'field_type' : 2, 'random_init' : False, 
                 'enter_goal_area' : True}
         )

register(id='SSLGoToBallShoot-v3',
         entry_point='rc_gym.ssl.ssl_go_to_ball_shoot:SSLGoToBallShootEnv',
         kwargs={'field_type' : 2, 'random_init' : True, 
                 'enter_goal_area' : True}
         )

register(id='SSLHWStaticDefenders-v0',
         entry_point='rc_gym.ssl.ssl_hw_challenge.static_defenders:SSLHWStaticDefendersEnv',
         kwargs={'field_type' : 2}
         )

register(id='SSLHWDribbling-v0',
         entry_point='rc_gym.ssl.ssl_hw_challenge.dribbling:SSLHWDribblingEnv',
         )

register(id='SSLContestedPossessionEnv-v0',
         entry_point='rc_gym.ssl.ssl_hw_challenge.contested_possession:SSLContestedPossessionEnv',
         kwargs={'random_init' : False}
         )

register(id='SSLContestedPossessionEnv-v1',
         entry_point='rc_gym.ssl.ssl_hw_challenge.contested_possession:SSLContestedPossessionEnv',
         kwargs={'random_init' : True}
         )
