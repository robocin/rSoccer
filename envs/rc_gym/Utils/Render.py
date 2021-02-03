import os

import numpy as np
import pygame
from pygame.constants import QUIT
from rc_gym.Entities import Frame


class RCRender:
    '''
    Rendering Class to RoboSim Simulator
    '''

    def __init__(self, n_robots_blue: int,
                 n_robots_yellow: int,
                 field_params: dict,
                 simulator: str = 'vss') -> None:
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

        simulator : str


        Returns
        -------
        None

        '''
        pygame.init()

        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.field_params = field_params
        self.simulator = simulator
        if simulator == 'vss':
            self.scale = 400
            self.object_scale = 400
        else:
            self.scale = 60
            self.object_scale = 400
        self.screen_width = int(
            self.field_params['field_width']*self.scale) + 50
        self.screen_height = int(
            self.field_params['field_length']*self.scale) + 50
        self.screen = None
        resources_path = os.path.dirname(
            os.path.abspath(os.path.expanduser(__file__)))
        resources_path = os.path.join(resources_path, "resources")
        resources_path = os.path.join(resources_path, self.simulator)
        self.robot_blue_image = pygame.image.load(os.path.join(resources_path,
                                                               'blue.png'))
        self.robot_yellow_image = pygame.image.load(
            os.path.join(resources_path, 'yellow.png'))
        self.field = pygame.image.load(os.path.join(resources_path,
                                                    'field.jpeg'))

        self.screen = pygame.display.set_mode(
            (self.screen_height, self.screen_width))

    def __del__(self):
        pygame.display.quit()

    def debug_line(self, center, robot_r, frame) -> None:
        '''
        Draws Lines in robot movement direction.

        Parameters: 
            Robot center array
            Robot radius
            Frame
        '''

        x_end = center[0] + (robot_r / (2 ** (1/2))) *\
            np.cos(np.deg2rad(frame.theta))
        y_end = center[1] + (robot_r / (2 ** (1/2))) *\
            np.sin(np.deg2rad(-frame.theta))

        pygame.draw.line(self.screen, (200, 0, 0),
                         center, (x_end, y_end), 2)

    def render_frame(self, frame: Frame) -> None:
        '''
        Draws the field, ball and players.

        Parameters
        ----------
        Frame

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

        self.screen.blit(self.field, (0, 0))
        ball_r = int(0.015*self.object_scale)

        pygame.draw.circle(self.screen, (254, 139, 0),
                           self.pos_transform(x=frame.ball.x, y=frame.ball.y),
                           ball_r)

        robot_r = int(0.04*self.object_scale)

        for blue in frame.robots_blue.values():

            center = self.pos_transform(x=blue.x, y=blue.y)

            l = int(robot_r * (2 ** (1/2)))

            surface_from_image = pygame.transform.\
                scale(self.robot_blue_image,
                      (int(l * 1.1), int(l * 1.1)))

            new_image = pygame.transform.rotate(surface_from_image,
                                                blue.theta - 90)

            self.screen.blit(new_image,
                             (center[0] - robot_r, center[1] - robot_r))

            # self.debug_line(center, robot_r, blue)

        for yellow in frame.robots_yellow.values():
            center = self.pos_transform(x=yellow.x, y=yellow.y)

            l = int(robot_r * (2 ** (1/2)))
            surface_from_image = pygame.transform.scale(self.robot_yellow_image,
                                                        (int(l * 1.1), int(l * 1.1)))

            new_image = pygame.transform.rotate(
                surface_from_image, yellow.theta - 90)

            self.screen.blit(
                new_image, (center[0] - robot_r, center[1] - robot_r))

            # self.debug_line(center, robot_r, yellow)

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
        pos = np.array([x + (self.field_params['field_length']/2),
                        (self.field_params['field_width']/2) - y])
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
    params = [9, 12, 3.6, 1.8, 1.8]
    # params = [1.3, 1.5, 0.7, 0.15, 0.4]

    # {keys: params}
    render = RCRender(3, 3, {key: param for key, param in zip(keys, params)})
    while True:
        render.draw_field()
