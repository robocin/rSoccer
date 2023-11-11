import pygame

from rsoccer_gym.Render.utils import COLORS


class RenderField:
    _scale = 1
    margin = 0
    length = 0
    width = 0
    margin = 0
    center_circle_r = 0
    penalty_length = 0
    penalty_width = 0
    goal_area_length = 0
    goal_area_width = 0
    goal_width = 0
    goal_depth = 0
    corner_arc_r = 0
    center_x = 0
    center_y = 0

    @property
    def scale(self):
        return self._scale

    def _screen_width(self):
        return self.length + 2 * self.margin

    def _screen_height(self):
        return self.width + 2 * self.margin

    def _transform_params(self):
        self.length *= self._scale
        self.width *= self._scale
        self.penalty_length *= self._scale
        self.penalty_width *= self._scale
        self.goal_width *= self._scale
        self.goal_depth *= self._scale
        self.center_circle_r *= self._scale
        self.goal_area_length *= self._scale
        self.goal_area_width *= self._scale
        self.corner_arc_r *= self._scale

    def draw_background(self, screen):
        screen.fill(COLORS["BG_GREEN"])

    def draw_field_bounds(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["WHITE"],
            (
                self.margin,
                self.margin,
                self.length,
                self.width,
            ),
            1,
        )

    def draw_penalty_area_left(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["WHITE"],
            (
                self.margin,
                (self.screen_height - self.penalty_width) // 2,
                self.penalty_length,
                self.penalty_width,
            ),
            1,
        )

    def draw_penalty_area_right(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["WHITE"],
            (
                self.screen_width - self.margin - self.penalty_length,
                (self.screen_height - self.penalty_width) // 2,
                self.penalty_length,
                self.penalty_width,
            ),
            1,
        )

    def draw_goal_area_left(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["WHITE"],
            (
                self.margin,
                (self.screen_height - self.goal_width) // 2,
                self.goal_depth,
                self.goal_width,
            ),
            1,
        )

    def draw_goal_area_right(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["WHITE"],
            (
                self.screen_width - self.margin - self.goal_depth,
                (self.screen_height - self.goal_width) // 2,
                self.goal_depth,
                self.goal_width,
            ),
            1,
        )

    def draw_goal_left(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["BLACK"],
            (
                self.margin - self.goal_depth,
                (self.screen_height - self.goal_width) // 2,
                self.goal_depth,
                self.goal_width,
            ),
        )

    def draw_goal_right(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["BLACK"],
            (
                self.screen_width - self.margin,
                (self.screen_height - self.goal_width) // 2,
                self.goal_depth,
                self.goal_width,
            ),
        )

    def draw_central_circle(self, screen):
        pygame.draw.circle(
            screen,
            COLORS["WHITE"],
            (int(self.screen_width / 2), int(self.screen_height / 2)),
            int(self.center_circle_r),
            1,
        )

    def draw_central_line(self, screen):
        midfield_x = self.screen_width / 2
        pygame.draw.line(
            screen,
            COLORS["WHITE"],
            (midfield_x, self.margin),
            (midfield_x, self.screen_height - self.margin),
            1,
        )

    def draw(self, screen):
        self.draw_background(screen)
        self.draw_field_bounds(screen)
        self.draw_central_line(screen)
        self.draw_central_circle(screen)
        self.draw_penalty_area_left(screen)
        self.draw_penalty_area_right(screen)
        self.draw_goal_area_left(screen)
        self.draw_goal_area_right(screen)
        self.draw_goal_left(screen)
        self.draw_goal_right(screen)


class Sim2DRenderField(RenderField):
    length = 840.0
    width = 544.0
    margin = 40.0
    center_circle_r = 73.2
    penalty_length = 132.0
    penalty_width = 322.56
    goal_area_length = 44.0
    goal_area_width = 146.56
    goal_width = 112.16
    goal_depth = 19.52
    corner_arc_r = 8.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.screen_width = self.length + 2 * self.margin
        self.screen_height = self.width + 2 * self.margin
        self.window_size = (int(self.screen_width), int(self.screen_height))


class VSSRenderField(RenderField):
    length = 1.5
    width = 1.3
    margin = 0.1
    center_circle_r = 0.2
    penalty_length = 0.15
    penalty_width = 0.7
    goal_area_length = 0
    goal_area_width = 0
    goal_width = 0.4
    goal_depth = 0.1
    corner_arc_r = 0.01
    _scale = 500

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_x = self.length / 2 + self.margin
        self.center_y = self.width / 2 + self.margin
        self.margin *= self._scale
        self.center_x *= self._scale
        self.center_y *= self._scale
        self._transform_params()
        self.screen_width = self._screen_width()
        self.screen_height = self._screen_height()
        self.window_size = (int(self.screen_width), int(self.screen_height))

    def draw(self, screen):
        self.draw_background(screen)
        self.draw_field_bounds(screen)
        self.draw_central_line(screen)
        self.draw_central_circle(screen)
        self.draw_penalty_area_left(screen)
        self.draw_penalty_area_right(screen)
        self.draw_goal_left(screen)
        self.draw_goal_right(screen)

    def draw_goal_left(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["WHITE"],
            (
                self.margin - self.goal_depth,
                (self.screen_height - self.goal_width) // 2,
                self.goal_depth,
                self.goal_width,
            ),
            1,
        )

    def draw_goal_right(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["WHITE"],
            (
                self.screen_width - self.margin,
                (self.screen_height - self.goal_width) // 2,
                self.goal_depth,
                self.goal_width,
            ),
            1,
        )


class SSLRenderField(VSSRenderField):
    length = 9
    width = 6
    margin = 0.35
    center_circle_r = 1
    penalty_length = 1
    penalty_width = 2
    goal_area_length = 0
    goal_area_width = 0
    goal_width = 1
    goal_depth = 0.18
    corner_arc_r = 0.01
    _scale = 100


if __name__ == "__main__":
    field = Sim2DRenderField()
    pygame.display.init()
    pygame.display.set_caption("SSL Environment")
    window = pygame.display.set_mode(field.window_size)
    clock = pygame.time.Clock()
    while True:
        field.draw(window)
        pygame.event.pump()
        pygame.display.update()
        clock.tick(60)
