from rsoccer_gym.vss.env_vss_progressive_attacker_with_goalkeeper.VSSProgressiveAttackerVSRandomGoalkeeper import (
    VSSProgressiveAttackerVSRandomGoalkeeper,
)

import numpy as np


class VSSProgressiveAttackerVSGoalkeeper(VSSProgressiveAttackerVSRandomGoalkeeper):

    def _get_goalkeeper_vels(self):
        # Obter a posição atual do goleiro
        gk_pos = self.frame.robots_yellow[0]

        # Obter a posição da bola
        ball_pos = self.frame.ball.y

        max_gk_pos = self.field.goal_width / 2

        parsed_ball_pos = np.clip(ball_pos, -max_gk_pos, max_gk_pos)

        # Calcular a diferença entre a posição Y do goleiro e a posição Y da bola
        diff_y = parsed_ball_pos - gk_pos.y

        if diff_y > 0:
            return 1, 1
        elif diff_y < 0:
            return -1, -1
        else:
            return 0, 0
