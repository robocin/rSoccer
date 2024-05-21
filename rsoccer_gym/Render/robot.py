import numpy as np
import pygame
from rsoccer_gym.Render.utils import COLORS, TAG_ID_COLORS


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
    size = 0.072

    def __init__(self, x, y, direction, scale, id, team_color):
        super().__init__(x, y, direction, scale, team_color=team_color)
        tag_id_colors = {0: COLORS["GREEN"], 1: COLORS["PURPLE"], 2: COLORS["RED"]}
        self.id = id
        self.id_color = tag_id_colors[id]

    def draw_robot(self, screen):
        rotated_surface = pygame.Surface(
            (self.size * 2, self.size * 2), pygame.SRCALPHA
        )

        pygame.draw.rect(
            rotated_surface,
            COLORS["ROBOT_BLACK"],
            rect=(self.size // 2, self.size // 2, self.size, self.size),
        )
        self.draw_team_tag(rotated_surface)
        self.draw_id_tag(rotated_surface)

        rotated_surface = pygame.transform.rotate(rotated_surface, -self.direction)
        new_rect = rotated_surface.get_rect(center=(self.x, self.y))

        screen.blit(rotated_surface, new_rect.topleft)

    def draw_direction(self, screen):
        pygame.draw.line(
            screen,
            COLORS["WHITE"],
            (self.x, self.y),
            (
                self.x + self.size * np.cos(np.deg2rad(self.direction)),
                self.y + self.size * np.sin(np.deg2rad(self.direction)),
            ),
            1
        )

    def draw_team_tag(self, surface):
        tag_size = (0.03 * self.scale, 0.068 * self.scale)
        tag_offset = (
            -1 + (self.size - 2 * tag_size[0]) // 2,
            (self.size - tag_size[1]) // 2,
        )
        tag_position = (self.size // 2 + tag_offset[0], self.size // 2 + tag_offset[1])

        pygame.draw.rect(
            surface,
            self.team_color,
            rect=(*tag_position, *tag_size),
        )

    def draw_id_tag(self, surface):
        tag_size = (0.03 * self.scale, 0.068 * self.scale)
        tag_offset = (self.size - tag_size[1]) // 2
        tag_position = (self.size + 1, self.size // 2 + tag_offset)

        pygame.draw.rect(
            surface,
            self.id_color,
            rect=(*tag_position, *tag_size),
        )

    def draw(self, screen):
        self.draw_robot(screen)
        # self.draw_direction(screen)


class SSLRobot(Robot):
    size = 0.09

    def __init__(self, x, y, direction, scale, id, team_color):
        super().__init__(x, y, direction, scale, team_color=team_color)
        self.id = id
        self.id_color = TAG_ID_COLORS[id]

    def draw(self, screen):
        self.draw_robot(screen)
        self.draw_direction(screen)

    def draw_robot(self, screen):
        rotated_surface = pygame.Surface(
            (self.size * 2, self.size * 2), pygame.SRCALPHA
        )

        pygame.draw.circle(
            rotated_surface, COLORS["ROBOT_BLACK"], (self.size, self.size), self.size
        )
        self.draw_team_tag(rotated_surface)
        self.draw_id_tag(rotated_surface)

        rotated_surface = pygame.transform.rotate(rotated_surface, -self.direction)
        new_rect = rotated_surface.get_rect(center=(self.x, self.y))
        
        screen.blit(rotated_surface, new_rect.topleft)
    
    def draw_team_tag(self, surface):
        tag_radius = 0.025 * self.scale
        pygame.draw.circle(
            surface, self.team_color, (self.size, self.size), tag_radius
        )
        
    def draw_id_tag(self, surface):
        tag_radius = 0.02 * self.scale
        translations = np.array([
            [0.035, 0.054772],
            [-0.054772, 0.035],
            [-0.054772, -0.035],
            [0.035, -0.054772],
        ])
        translations *= self.scale
        tags_position = translations + np.array([self.size, self.size])
        for i, tag_position in enumerate(tags_position):
            pygame.draw.circle(
                surface, self.id_color[i], tag_position.astype(int), tag_radius
            )
        
    
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


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((750, 650))
    robots = []
    robots.append(Sim2DRobot(750 // 2, 650 // 2, 0, 0, 100, 10, COLORS["BLUE"]))
    robots.append(VSSRobot(150, 300, 0, 500, 0, COLORS["BLUE"]))
    robots.append(SSLRobot(600, 300, 0, 500, 0, COLORS["BLUE"]))
    while True:
        clock.tick(60)
        screen.fill(COLORS["GREEN"])
        for robot in robots:
            robot.draw(screen)
            robot.direction += 1
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
