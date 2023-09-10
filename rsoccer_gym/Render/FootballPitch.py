import pygame
import math

WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLACK = (0, 0, 0)

# Original parameters

# PITCH_LENGTH = 315.0
# PITCH_WIDTH = 204.0
# PITCH_MARGIN = 15.0 
# CENTER_CIRCLE_R = 27.45
# PENALTY_AREA_LENGTH = 49.5
# PENALTY_AREA_WIDTH = 120.96
# GOAL_AREA_LENGTH = 16.5
# GOAL_AREA_WIDTH = 54.96
# GOAL_WIDTH = 42.06
# GOAL_DEPTH = 7.32
# CORNER_ARC_R = 3.0
# PLAYER_SIZE = 0.3

PITCH_LENGTH = 840.0
PITCH_WIDTH = 544.0
PITCH_MARGIN = 40.0
CENTER_CIRCLE_R = 73.2
PENALTY_AREA_LENGTH = 132.0
PENALTY_AREA_WIDTH = 322.56
GOAL_AREA_LENGTH = 44.0
GOAL_AREA_WIDTH = 146.56
GOAL_WIDTH = 112.16
GOAL_DEPTH = 19.52
CORNER_ARC_R = 8.0

screen_width = PITCH_LENGTH + 2 * PITCH_MARGIN
screen_height = PITCH_WIDTH + 2 * PITCH_MARGIN

pygame.init()

screen = pygame.display.set_mode((int(screen_width), int(screen_height)))
pygame.display.set_caption("Campo de Futebol")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(GREEN)

    # field
    pygame.draw.rect(screen, WHITE, (PITCH_MARGIN, PITCH_MARGIN, PITCH_LENGTH, PITCH_WIDTH), 2)

    # central line
    midfield_x = screen_width / 2
    pygame.draw.line(screen, WHITE, (midfield_x, PITCH_MARGIN), (midfield_x, screen_height - PITCH_MARGIN), 2)

    # central circle
    pygame.draw.circle(screen, WHITE, (int(screen_width / 2), int(screen_height / 2)), int(CENTER_CIRCLE_R), 2)
    
    # penalty area
    pygame.draw.rect(screen, WHITE, (PITCH_MARGIN, (screen_height - PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH), 2)
    pygame.draw.rect(screen, WHITE, (screen_width - PITCH_MARGIN - PENALTY_AREA_LENGTH, (screen_height - PENALTY_AREA_WIDTH) / 2, PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH), 2)
    
    # goal area
    pygame.draw.rect(screen, WHITE, (PITCH_MARGIN, (screen_height - GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, GOAL_AREA_WIDTH), 2)
    pygame.draw.rect(screen, WHITE, (screen_width - PITCH_MARGIN - GOAL_AREA_LENGTH, (screen_height - GOAL_AREA_WIDTH) / 2, GOAL_AREA_LENGTH, GOAL_AREA_WIDTH), 2)
    
    # goal
    pygame.draw.rect(screen, BLACK, (PITCH_MARGIN/2, (screen_height - GOAL_WIDTH) / 2, GOAL_DEPTH, GOAL_WIDTH))
    pygame.draw.rect(screen, BLACK, (screen_width - PITCH_MARGIN, (screen_height - GOAL_WIDTH) / 2 , GOAL_DEPTH, GOAL_WIDTH))

    # arc
    pygame.draw.arc(screen, WHITE, (PITCH_MARGIN - CORNER_ARC_R, PITCH_MARGIN - CORNER_ARC_R, 2 * CORNER_ARC_R, 2 * CORNER_ARC_R), 1.5 * math.pi, 0, 2) #ok
    pygame.draw.arc(screen, WHITE, (PITCH_MARGIN - CORNER_ARC_R, screen_height - PITCH_MARGIN - CORNER_ARC_R, 2 * CORNER_ARC_R, 2 * CORNER_ARC_R),0, 0.5 * math.pi, 2) # ok
    pygame.draw.arc(screen, WHITE, (screen_width - PITCH_MARGIN - CORNER_ARC_R, PITCH_MARGIN - CORNER_ARC_R, 2 * CORNER_ARC_R, 2 * CORNER_ARC_R), math.pi, 1.5 * math.pi, 2)
    pygame.draw.arc(screen, WHITE, (screen_width - PITCH_MARGIN - CORNER_ARC_R, screen_height - PITCH_MARGIN - CORNER_ARC_R, 2 * CORNER_ARC_R, 2 * CORNER_ARC_R), 0.5 * math.pi, math.pi, 2)# ok
    
    pygame.display.flip()

pygame.quit()
