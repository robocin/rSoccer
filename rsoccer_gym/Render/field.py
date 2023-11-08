import pygame
from rsoccer_gym.Entities.Field import Field
from rsoccer_gym.Render.utils import COLORS


class RenderField:
    _scale = 1
    margin = 0
    window_surface = None
    window_size = None
    clock = None
    screen_height = None
    screen_width = None
    length = None
    width = None
    margin = None
    center_circle_r = None
    penalty_length = None
    penalty_width = None
    goal_area_length = None
    goal_area_width = None
    goal_width = None
    goal_depth = None
    corner_arc_r = None

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.caption = "rSoccer Environment"

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

    def get_window(self):
        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption(self.caption)
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        return self.window_surface

    def draw_background(self):
        self.window_surface.fill(COLORS["BG_GREEN"])

    def draw_field_bounds(self):
        pygame.draw.rect(
            self.window_surface,
            COLORS["WHITE"],
            (
                self.margin,
                self.margin,
                self.length,
                self.width,
            ),
            2,
        )

    def draw_penalty_area_left(self):
        pygame.draw.rect(
            self.window_surface,
            COLORS["WHITE"],
            (
                self.margin,
                (self.screen_height - self.penalty_width) / 2,
                self.penalty_length,
                self.penalty_width,
            ),
            2,
        )

    def draw_penalty_area_right(self):
        pygame.draw.rect(
            self.window_surface,
            COLORS["WHITE"],
            (
                self.screen_width - self.margin - self.penalty_length,
                (self.screen_height - self.penalty_width) / 2,
                self.penalty_length,
                self.penalty_width,
            ),
            2,
        )

    def draw_goal_area_left(self):
        pygame.draw.rect(
            self.window_surface,
            COLORS["WHITE"],
            (
                self.margin,
                (self.screen_height - self.goal_width) / 2,
                self.goal_depth,
                self.goal_width,
            ),
            2,
        )

    def draw_goal_area_right(self):
        pygame.draw.rect(
            self.window_surface,
            COLORS["WHITE"],
            (
                self.screen_width - self.margin - self.goal_depth,
                (self.screen_height - self.goal_width) / 2,
                self.goal_depth,
                self.goal_width,
            ),
            2,
        )

    def draw_goal_left(self):
        pygame.draw.rect(
            self.window_surface,
            COLORS["BLACK"],
            (
                self.margin / 2,
                (self.screen_height - self.goal_width) / 2,
                self.goal_depth,
                self.goal_width,
            ),
        )

    def draw_goal_right(self):
        pygame.draw.rect(
            self.window_surface,
            COLORS["BLACK"],
            (
                self.screen_width - self.margin,
                (self.screen_height - self.goal_width) / 2,
                self.goal_depth,
                self.goal_width,
            ),
        )

    def draw_central_circle(self):
        pygame.draw.circle(
            self.window_surface,
            COLORS["WHITE"],
            (int(self.screen_width / 2), int(self.screen_height / 2)),
            int(self.center_circle_r),
            2,
        )

    def draw_central_line(self):
        midfield_x = self.screen_width / 2
        pygame.draw.line(
            self.window_surface,
            COLORS["WHITE"],
            (midfield_x, self.margin),
            (midfield_x, self.screen_height - self.margin),
            2,
        )

    def draw_field(self):
        self.draw_background()
        self.draw_field_bounds()
        self.draw_central_line()
        self.draw_central_circle()
        self.draw_penalty_area_left()
        self.draw_penalty_area_right()
        self.draw_goal_area_left()
        self.draw_goal_area_right()
        self.draw_goal_left()
        self.draw_goal_right()


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
    goal_depth = 0.05
    corner_arc_r = 0.01
    _scale = 500

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin *= self._scale
        self._transform_params()
        self.screen_width = self._screen_width()
        self.screen_height = self._screen_height()
        self.window_size = (int(self.screen_width), int(self.screen_height))

    def draw_field(self):
        self.draw_background()
        self.draw_field_bounds()
        self.draw_central_line()
        self.draw_central_circle()
        self.draw_penalty_area_left()
        self.draw_penalty_area_right()
        self.draw_goal_left()
        self.draw_goal_right()


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    field = SSLRenderField(render_mode="human")
    window = field.get_window()
    clock = pygame.time.Clock()
    while True:
        field.draw_field()
        pygame.event.pump()
        pygame.display.update()
        clock.tick(60)
