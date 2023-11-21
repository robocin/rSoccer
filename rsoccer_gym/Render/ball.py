import pygame
from rsoccer_gym.Render.utils import COLORS


class Ball:
    radius = 0.0215

    def __init__(self, x, y, scale) -> None:
        self.x = x
        self.y = y
        self.radius *= scale
        self.scale = scale

    def update(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, COLORS["ORANGE"], (self.x, self.y), self.radius)
        pygame.draw.circle(screen, COLORS["BLACK"], (self.x, self.y), self.radius, 1)
