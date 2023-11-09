import numpy as np
import pygame
from utils import COLORS, TAG_ID_COLORS


class Robot:
    size = 1

    def __init__(self, x, y, direction, scale, team_color=COLORS["BLUE"]):
        self.team_color = team_color
        self.x = x
        self.y = y
        self.direction = direction
        self.size *= scale
        self.scale = scale

    def draw(self, screen):
        raise NotImplementedError("Implement this method in a child class")


class Sim2DRobot(Robot):
    size = 10
    vision_size = 2

    def __init__(
        self, x, y, direction, vision_direction, vision_openning, scale, team_color
    ):
        super().__init__(x, y, direction, scale, team_color=team_color)
        self.vision_direction = vision_direction
        self.vision_openning = vision_openning

    def draw(self, screen):
        pygame.draw.circle(screen, self.team_color, (self.x, self.y), self.size)
        pygame.draw.circle(
            screen, COLORS["ROBOT_BLACK"], (self.x, self.y), self.size, 1
        )
        pygame.draw.line(
            screen,
            COLORS["BLACK"],
            (self.x, self.y),
            (
                self.x + self.size * np.cos(np.deg2rad(self.direction)),
                self.y + self.size * np.sin(np.deg2rad(self.direction)),
            ),
            min(max(self.scale, 1), 3),
        )
        pygame.draw.polygon(
            screen,
            COLORS["WHITE"],
            [
                (
                    self.x,
                    self.y,
                ),
                (
                    self.x
                    + self.size
                    * self.vision_size
                    * np.cos(np.deg2rad(self.vision_direction + self.vision_openning)),
                    self.y
                    + self.size
                    * self.vision_size
                    * np.sin(np.deg2rad(self.vision_direction + self.vision_openning)),
                ),
                (
                    self.x
                    + self.size
                    * self.vision_size
                    * np.cos(np.deg2rad(self.vision_direction - self.vision_openning)),
                    self.y
                    + self.size
                    * self.vision_size
                    * np.sin(np.deg2rad(self.vision_direction - self.vision_openning)),
                ),
            ],
            1,
        )


class VSSRobot(Robot):
    size = 8

    def __init__(self, x, y, direction, scale, id, team_color):
        super().__init__(x, y, direction, scale, team_color=team_color)
        self.id = id

    def draw_robot(self, screen):
        rect = pygame.Rect(0, 0, self.size, self.size)
        rect.center = (self.x, self.y)
        pygame.draw.rect(screen, self.team_color, rect)

    def draw_direction(self, screen):
        pygame.draw.line(
            screen,
            COLORS["WHITE"],
            (self.x, self.y),
            (
                self.x + self.size * np.cos(np.deg2rad(self.direction)),
                self.y + self.size * np.sin(np.deg2rad(self.direction)),
            ),
        )

    def draw_team_tag(self, screen):
        pygame.draw.rect(
            screen,
            COLORS["ROBOT_BLACK"],
            (
                self.x - self.size,
                self.y - self.size,
                self.size * 2,
                self.size * 2,
            ),
        )

    def draw(self, screen):
        self.draw_robot(screen)
        # self.draw_direction(screen)
        # self.draw_team_tag(screen)
        # self.draw_id_tag(screen)


# class SSLRobot(Robot):
#     direction_size = 1.8
#     direction_openning = 30

#     def __init__(self, x, y, direction, tag_id):
#         super().__init__(x, y, size, direction)
#         self.tag_id = tag_id

#     def draw(self, screen):
#         super().draw(screen)
#         pygame.draw.circle(screen, TAG_ID_COLORS[self.tag_id], (self.x, self.y), 0.8)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    robot = VSSRobot(250, 250, 30, 10, 0, COLORS["BLUE"])
    while True:
        screen.fill(COLORS["GREEN"])
        robot.draw(screen)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
