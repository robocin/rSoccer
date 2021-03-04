import os

import numpy as np
from rc_gym.Entities import Frame
from gym.envs.classic_control import rendering
from typing import Dict, List, Tuple

# COLORS RGB
BG_GREEN =      (11  /255, 102 /255, 35  /255)
LINES_WHITE =   (220 /255, 220 /255, 220 /255)
ROBOT_BLACK =   (25  /255, 25  /255, 25  /255)
BALL_ORANGE =   (253 /255, 106 /255, 2   /255)
TAG_BLUE =      (0   /255, 64  /255, 255 /255)
TAG_YELLOW =    (250 /255, 218 /255, 94  /255)
TAG_GREEN =     (57  /255, 255 /255, 20  /255)
TAG_RED =       (151 /255, 21  /255, 0   /255)
TAG_PURPLE =    (102 /255, 51  /255, 153 /255)


class RCGymRender:
    '''
    Rendering Class to RoboSim Simulator, based on gym classic control rendering
    '''

    def __init__(self, n_robots_blue: int,
                 n_robots_yellow: int,
                 field_params: dict,
                 simulator: str = 'vss',
                 width: int = 750,
                 height: int = 650) -> None:
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
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow
        self.field = field_params
        self.ball: rendering.Transform = None
        self.blue_robots: List[rendering.Transform] = []
        self.yellow_robots: List[rendering.Transform] = []

        # Window dimensions in pixels
        screen_width = width
        screen_height = height

        # Window margin
        margin = 0.1 if simulator == "vss" else 0.4
        # Half field width
        h_len = (self.field["field_length"]
                 + 2*self.field["goal_depth"]) / 2
        # Half field height
        h_wid = (self.field["field_width"]) / 2

        # Window dimensions in meters
        self.screen_dimensions = {
            "left": -(h_len + margin),
            "right": (h_len + margin),
            "bottom": -(h_wid + margin),
            "top": (h_wid + margin)
        }

        # Init window
        self.screen = rendering.Viewer(screen_width, screen_height)

        # Set window bounds, will scale objects accordingly
        self.screen.set_bounds(**self.screen_dimensions)

        # add backgrond
        self._add_background()

        if simulator == "vss":
            # add field_lines
            self._add_field_lines_vss()

            # add robots
            self._add_vss_robots()
        if simulator == "ssl":
            # add field_lines
            self._add_field_lines_ssl()
            # add robots
            self._add_ssl_robots()
        
        # add ball
        self._add_ball()

    def __del__(self):
        self.screen.close()
        del(self.screen)
        self.screen = None

    def render_frame(self, frame: Frame, return_rgb_array: bool = False) -> None:
        '''
        Draws the field, ball and players.

        Parameters
        ----------
        Frame

        Returns
        -------
        None

        '''

        self.ball.set_translation(frame.ball.x, frame.ball.y)

        for i, blue in enumerate(frame.robots_blue.values()):
            self.blue_robots[i].set_translation(blue.x, blue.y)
            self.blue_robots[i].set_rotation(np.deg2rad(blue.theta))

        for i, yellow in enumerate(frame.robots_yellow.values()):
            self.yellow_robots[i].set_translation(yellow.x, yellow.y)
            self.yellow_robots[i].set_rotation(np.deg2rad(yellow.theta))

        return self.screen.render(return_rgb_array=return_rgb_array)

    def _add_background(self) -> None:
        back_ground = rendering.FilledPolygon([
            (self.screen_dimensions["right"], self.screen_dimensions["top"]),
            (self.screen_dimensions["right"],
             self.screen_dimensions["bottom"]),
            (self.screen_dimensions["left"], self.screen_dimensions["bottom"]),
            (self.screen_dimensions["left"], self.screen_dimensions["top"]),
        ])
        back_ground.set_color(*BG_GREEN)
        self.screen.add_geom(back_ground)

    #----------VSS-----------#
    
    def _add_field_lines_vss(self) -> None:
        # Vertical Lines X
        x_border = self.field["field_length"] / 2
        x_goal = x_border + self.field["goal_depth"]
        x_penalty = x_border - self.field["penalty_length"]
        x_center = 0

        # Horizontal Lines Y
        y_border = self.field["field_width"] / 2
        y_penalty = self.field["penalty_width"] / 2
        y_goal = self.field["goal_width"] / 2

        # add field borders
        field_border_points = [
            (x_border, y_border),
            (x_border, -y_border),
            (-x_border, -y_border),
            (-x_border, y_border)
        ]
        field_border = rendering.PolyLine(field_border_points, close=True)
        field_border.set_color(*LINES_WHITE)

        # Center line and circle
        center_line = rendering.Line(
            (x_center, y_border), (x_center, -y_border))
        center_line.set_color(*LINES_WHITE)
        center_circle = rendering.make_circle(0.2, filled=False)
        center_circle.set_color(*LINES_WHITE)

        # right side penalty box
        penalty_box_right_points = [
            (x_border, y_penalty),
            (x_penalty, y_penalty),
            (x_penalty, -y_penalty),
            (x_border, -y_penalty)
        ]
        penalty_box_right = rendering.PolyLine(
            penalty_box_right_points, close=False)
        penalty_box_right.set_color(*LINES_WHITE)

        # left side penalty box
        penalty_box_left_points = [
            (-x_border, y_penalty),
            (-x_penalty, y_penalty),
            (-x_penalty, -y_penalty),
            (-x_border, -y_penalty)
        ]
        penalty_box_left = rendering.PolyLine(
            penalty_box_left_points, close=False)
        penalty_box_left.set_color(*LINES_WHITE)

        # Right side goal line
        goal_line_right_points = [
            (x_border, y_goal),
            (x_goal, y_goal),
            (x_goal, -y_goal),
            (x_border, -y_goal)
        ]
        goal_line_right = rendering.PolyLine(
            goal_line_right_points, close=False)
        goal_line_right.set_color(*LINES_WHITE)

        # Left side goal line
        goal_line_left_points = [
            (-x_border, y_goal),
            (-x_goal, y_goal),
            (-x_goal, -y_goal),
            (-x_border, -y_goal)
        ]
        goal_line_left = rendering.PolyLine(goal_line_left_points, close=False)
        goal_line_left.set_color(*LINES_WHITE)

        self.screen.add_geom(field_border)
        self.screen.add_geom(center_line)
        self.screen.add_geom(center_circle)
        self.screen.add_geom(penalty_box_right)
        self.screen.add_geom(penalty_box_left)
        self.screen.add_geom(goal_line_right)
        self.screen.add_geom(goal_line_left)

    def _add_vss_robots(self) -> None:
        tag_id_colors: Dict[int, Tuple[float, float, float]] = {
            0 : TAG_GREEN,
            1 : TAG_PURPLE,
            2 : TAG_RED
        }
        
        # Add blue robots
        for id in range(self.n_robots_blue):
            self.blue_robots.append(
                self._add_vss_robot(team_color=TAG_BLUE, id_color=tag_id_colors[id])
            )
            
        # Add yellow robots
        for id in range(self.n_robots_yellow):
            self.yellow_robots.append(
                self._add_vss_robot(team_color=TAG_YELLOW, id_color=tag_id_colors[id])
            )

    def _add_vss_robot(self, team_color, id_color) -> rendering.Transform:
        robot_transform:rendering.Transform = rendering.Transform()
        
        # Robot dimensions
        robot_x: float = 0.075
        robot_y: float = 0.075
        # Tag dimensions
        tag_x: float = 0.030
        tag_y: float = 0.065
        tag_x_offset: float = (0.065 / 2) / 2
        
        # Robot vertices (zero at center)
        robot_vertices: List[Tuple[float,float]]= [
            (robot_x/2, robot_y/2),
            (robot_x/2, -robot_y/2),
            (-robot_x/2, -robot_y/2),
            (-robot_x/2, robot_y/2)
        ]
        # Tag vertices (zero at center)
        tag_vertices: List[Tuple[float,float]]= [
            (tag_x/2, tag_y/2),
            (tag_x/2, -tag_y/2),
            (-tag_x/2, -tag_y/2),
            (-tag_x/2, tag_y/2)
        ]
        
        # Robot object
        robot = rendering.FilledPolygon(robot_vertices)
        robot.set_color(*ROBOT_BLACK)
        robot.add_attr(robot_transform)

        # Team tag object
        team_tag = rendering.FilledPolygon(tag_vertices)
        team_tag.set_color(*team_color)
        team_tag.add_attr(rendering.Transform(translation=(tag_x_offset, 0)))
        team_tag.add_attr(robot_transform)
        
        # Id tag object
        id_tag = rendering.FilledPolygon(tag_vertices)
        id_tag.set_color(*id_color)
        id_tag.add_attr(rendering.Transform(translation=(-tag_x_offset, 0)))
        id_tag.add_attr(robot_transform)

        # Add objects to screen
        self.screen.add_geom(robot)
        self.screen.add_geom(team_tag)
        self.screen.add_geom(id_tag)

        # Return the transform class to change robot position
        return robot_transform

    #----------SSL-----------#
    
    def _add_field_lines_ssl(self) -> None:
        # Vertical Lines X
        x_border = self.field["field_length"] / 2
        x_goal = x_border + self.field["goal_depth"]
        x_penalty = x_border - self.field["penalty_length"]
        x_center = 0

        # Horizontal Lines Y
        y_border = self.field["field_width"] / 2
        y_penalty = self.field["penalty_width"] / 2
        y_goal = self.field["goal_width"] / 2

        # add field borders
        field_border_points = [
            (x_border, y_border),
            (x_border, -y_border),
            (-x_border, -y_border),
            (-x_border, y_border)
        ]
        field_border = rendering.PolyLine(field_border_points, close=True)
        field_border.set_color(*LINES_WHITE)

        # Center line and circle
        center_line = rendering.Line(
            (x_center, y_border), (x_center, -y_border))
        center_line.set_color(*LINES_WHITE)
        center_circle = rendering.make_circle(0.2, filled=False)
        center_circle.set_color(*LINES_WHITE)

        # right side penalty box
        penalty_box_right_points = [
            (x_border, y_penalty),
            (x_penalty, y_penalty),
            (x_penalty, -y_penalty),
            (x_border, -y_penalty)
        ]
        penalty_box_right = rendering.PolyLine(
            penalty_box_right_points, close=False)
        penalty_box_right.set_color(*LINES_WHITE)

        # left side penalty box
        penalty_box_left_points = [
            (-x_border, y_penalty),
            (-x_penalty, y_penalty),
            (-x_penalty, -y_penalty),
            (-x_border, -y_penalty)
        ]
        penalty_box_left = rendering.PolyLine(
            penalty_box_left_points, close=False)
        penalty_box_left.set_color(*LINES_WHITE)

        # Right side goal line
        goal_line_right_points = [
            (x_border, y_goal),
            (x_goal, y_goal),
            (x_goal, -y_goal),
            (x_border, -y_goal)
        ]
        goal_line_right = rendering.PolyLine(
            goal_line_right_points, close=False)
        goal_line_right.set_color(*LINES_WHITE)

        # Left side goal line
        goal_line_left_points = [
            (-x_border, y_goal),
            (-x_goal, y_goal),
            (-x_goal, -y_goal),
            (-x_border, -y_goal)
        ]
        goal_line_left = rendering.PolyLine(goal_line_left_points, close=False)
        goal_line_left.set_color(*LINES_WHITE)

        self.screen.add_geom(field_border)
        self.screen.add_geom(center_line)
        self.screen.add_geom(center_circle)
        self.screen.add_geom(penalty_box_right)
        self.screen.add_geom(penalty_box_left)
        self.screen.add_geom(goal_line_right)
        self.screen.add_geom(goal_line_left)

    def _add_ssl_robots(self) -> None:
        
        # Add blue robots
        for id in range(self.n_robots_blue):
            self.blue_robots.append(
                self._add_ssl_robot(team_color=TAG_BLUE)
            )
            
        # Add yellow robots
        for id in range(self.n_robots_yellow):
            self.yellow_robots.append(
                self._add_ssl_robot(team_color=TAG_YELLOW)
            )

    def _add_ssl_robot(self, team_color, id_color=0) -> rendering.Transform:
        robot_transform:rendering.Transform = rendering.Transform()
        
        # Robot dimensions
        robot_radius: float = 0.09
        distance_center_kicker: float = 0.073
        kicker_angle = 2 * np.arccos(distance_center_kicker / robot_radius)
        res = 30

        points = []
        for i in range(res + 1):
            ang = (2*np.pi - kicker_angle)*i / res
            ang += kicker_angle/2
            points.append((np.cos(ang)*robot_radius, np.sin(ang)*robot_radius))

        # Robot object
        robot = rendering.FilledPolygon(points)
        robot.set_color(*team_color)
        robot.add_attr(robot_transform)
        
        # Robot outline
        robot_outline = rendering.PolyLine(points, True)
        robot_outline.set_color(*ROBOT_BLACK)
        robot_outline.add_attr(robot_transform)

        # Add objects to screen
        self.screen.add_geom(robot)
        self.screen.add_geom(robot_outline)

        # Return the transform class to change robot position
        return robot_transform

    def _add_ball(self):
        ball_radius: float = 0.0215
        ball_transform:rendering.Transform = rendering.Transform()
        
        ball: rendering.Geom = rendering.make_circle(ball_radius, filled=True)
        ball.set_color(*BALL_ORANGE)
        ball.add_attr(ball_transform)
        
        ball_outline: rendering.Geom = rendering.make_circle(ball_radius, filled=False)
        ball_outline.set_color(*ROBOT_BLACK)
        ball_outline.add_attr(ball_transform)
        
        self.screen.add_geom(ball)
        self.screen.add_geom(ball_outline)
        
        self.ball = ball_transform