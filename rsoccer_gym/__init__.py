from gym.envs.registration import register

register(id='VSS-v0',
         entry_point='rsoccer_gym.vss.env_vss:VSSEnv',
         max_episode_steps=1200
         )

register(id='VSSMA-v0',
         entry_point='rsoccer_gym.vss.env_ma:VSSMAEnv',
         max_episode_steps=1200
         )

register(id='VSSMAOpp-v0',
         entry_point='rsoccer_gym.vss.env_ma:VSSMAOpp',
         max_episode_steps=1200
         )

register(id='VSSGk-v0',
         entry_point='rsoccer_gym.vss.env_gk:rSimVSSGK',
         max_episode_steps=1200
         )

register(id='VSSFIRA-v0',
         entry_point='rsoccer_gym.vss.env_vss:VSSFIRAEnv',
         max_episode_steps=1200
         )

register(id='SSLGoToBall-v0',
         entry_point='rsoccer_gym.ssl.ssl_go_to_ball:SSLGoToBallEnv',
         kwargs={'field_type': 2, 'n_robots_yellow': 6},
         max_episode_steps=1200
         )

register(id='SSLGoToBallIR-v0',
         entry_point='rsoccer_gym.ssl.ssl_go_to_ball:SSLGoToBallIREnv',
         kwargs={'field_type': 2, 'n_robots_yellow': 6},
         max_episode_steps=1200
         )

register(id='SSLGoToBallShoot-v0',
         entry_point='rsoccer_gym.ssl.ssl_go_to_ball_shoot:SSLGoToBallShootEnv',
         kwargs={'field_type': 2, 'random_init': True,
                 'enter_goal_area': False},
         max_episode_steps=2400
         )

register(id='SSLStaticDefenders-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge.static_defenders:SSLHWStaticDefendersEnv',
         kwargs={'field_type': 2},
         max_episode_steps=1000
         )

register(id='SSLDribbling-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge.dribbling:SSLHWDribblingEnv',
         max_episode_steps=4800
         )

register(id='SSLContestedPossession-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge.contested_possession:SSLContestedPossessionEnv',
         max_episode_steps=1200
         )

register(id='SSLPassEndurance-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge:SSLPassEnduranceEnv',
         max_episode_steps=120
         )

register(id='SSLPassEnduranceMA-v0',
         entry_point='rsoccer_gym.ssl.ssl_hw_challenge:SSLPassEnduranceMAEnv',
         max_episode_steps=1200
         )
