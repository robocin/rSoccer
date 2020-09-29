import os

import numpy as np
import pygame
from pygame.constants import QUIT


class RCRender:
    '''
    Rendering Class to RoboSim Simulator
    '''

    def __init__(self, n_robots_blue: int,
                 n_robots_yellow: int,
                 field_params: dict) -> None:
        '''
        Creates our View object.

        Parameters
        ----------
        n_robots_blue : int
            Number of blue robots
        
        n_robots_yellow : int
            Number of yellow robots

        field_params : dict
            field_width, field_length,
            penalty_width, penalty_length,
            goal_width

        Returns
        -------
        None

        '''
        pygame.init()
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.field_params = field_params
        self.scale = 400
        self.screen_width = int(
            self.field_params['field_width']*self.scale)+50
        self.screen_height = int(
            self.field_params['field_length']*self.scale)+50
        self.screen = None

    def draw_field(self) -> None:
        '''
        Auxiliary function to draw the background.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        penalty_width = int(self.field_params['penalty_width']*self.scale)
        penalty_length = int(self.field_params['penalty_length']*self.scale)

        goal_width = int(self.field_params['goal_width']*self.scale)
        goal_length = int(-0.05*self.scale)

        self.screen.fill((0, 254, 19))
        # Out lines
        pygame.draw.rect(self.screen, (255, 255, 255),
                         pygame.Rect(25, 25,
                                     self.screen_height - 50,
                                     self.screen_width - 50), 1)
        # Half field
        pygame.draw.line(self.screen, (255, 255, 255),
                         (self.screen_height//2, 25),
                         (self.screen_height//2, self.screen_width-25), 1)
        pygame.draw.circle(self.screen, (255, 255, 255),
                           (self.screen_height//2,
                            self.screen_width//2), 20, 1)
        # Penalty Area Left
        penalty_x = 25
        penalty_y = self.screen_width//2 - penalty_width//2
        pygame.draw.rect(self.screen, (255, 255, 255),
                         pygame.Rect(25, penalty_y,
                                     penalty_length, penalty_width), 1)
        # Penalty Area Right
        penalty_x = self.screen_height - penalty_length - 25
        pygame.draw.rect(self.screen, (255, 255, 255),
                         pygame.Rect(penalty_x, penalty_y,
                                     penalty_length, penalty_width), 1)

        # goal Area Left
        goal_x = 25
        goal_y = self.screen_width//2 - goal_width//2
        pygame.draw.rect(self.screen, (255, 255, 255),
                         pygame.Rect(25, goal_y,
                                     goal_length, goal_width), 1)
        # goal Area Right
        goal_x = self.screen_height - goal_length - 25
        pygame.draw.rect(self.screen, (255, 255, 255),
                         pygame.Rect(goal_x, goal_y,
                                     goal_length, goal_width), 1)

    def view(self, ball: np.ndarray,
             blues: np.ndarray, yellows: np.ndarray) -> None:
        '''
        Draws the field, ball and players.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        if self.screen is None:
            self.screen = pygame.display.set_mode(
                (self.screen_height, self.screen_width))
        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
        self.draw_field()
        ball_r = int(0.015*self.scale)
        pygame.draw.circle(self.screen, (254, 139, 0),
                           self.pos_transform(*ball), ball_r)
        robot_r = int(0.025*self.scale)
        for blue in blues:
            pygame.draw.circle(self.screen, (26, 43, 218),
                               self.pos_transform(*blue),
                               robot_r, robot_r-3)
        for yellow in yellows:
            pygame.draw.circle(self.screen, (249, 255, 55),
                               self.pos_transform(*yellow),
                               robot_r, robot_r-3)
        pygame.display.update()

    def pos_transform(self, x: float, y: float) -> np.ndarray:
        '''
        Transforms original position of ball and players
        to the desired pixel position.

        Parameters
        ----------
        x: float
            Original x position

        y: float
            Original y position

        Returns
        -------
        np.ndarray
            Pixel position of object

        '''
        pos = np.array([x+self.field_params['field_length']/2,
                        y+self.field_params['field_width']/2])
        pos *= self.scale
        pos = np.array(pos, dtype=np.int) + 25
        return pos

    def close(self) -> None:
        '''
        Closes the view

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        pygame.display.quit()
        self.screen = None


if __name__ == "__main__":
    keys = ['field_width', 'field_length',
            'penalty_width', 'penalty_length',
            'goal_width']
    params = [1.3, 1.5, 0.7, 0.15, 0.4]
    render = RCRender(3, 3, {key: param for key, param in zip(keys, params)})
    while True:
        ball = (np.random.rand()*1.5, np.random.rand()*1.3)
        blues = list()
        for i in range(3):
            blues.append((np.random.rand()*1.5, np.random.rand()*1.3))
        yellows = list()
        for i in range(3):
            yellows.append((np.random.rand()*1.5, np.random.rand()*1.3))
        render.view(ball, blues, yellows)
