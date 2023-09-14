from dataclasses import Field, dataclass
from typing import Final
from rsoccer_gym.Render.BaseRender import BaseRender, RenderParam2D

@dataclass
class RenderParam2D:
    DEFAULT_WIDTH: int = 1080
    DEFAULT_HEIGHT: int = 720

class Render2D(BaseRender):
    def __init__(self, n_allies: int,
                 n_adversaries: int,
                 field_params: Field,
                 width: float = RenderParam2D.DEFAULT_WIDTH,
                 height: float = RenderParam2D.DEFAULT_HEIGHT) -> None:
        '''
        Docstrings to write.
        '''
        self.n_allies: Final[int] = n_allies
        self.n_adversaries: Final[int] = n_adversaries
        self.field_params: Final[Field] = field_params

        # todo(bonna.borsoi): implement required params to render

        margin = 40.0
        goal_width = 146.56
        goal_length = 44
        pitch_length = 840
        pitch_width = 544 

        # half dimensions
        h_length = (pitch_length + 2*goal_length) / 2
        h_width = pitch_width / 2

        self.screen_dimensions = {
            "left" : -(h_length + margin),
            "right" : (h_length + margin),
            "bottom" : -(h_width + margin),
            "top" : (h_width + margin),
        }

        self.screen = None

        self._add_background()
        self._add_field_lines()
        self._add_multiple_agents()
        self._add_ball()

    def __del__(self):
        self.screen.close()
        del(self.screen)
        self.screen = None

    def render_frame(self) -> None:
        raise NotImplementedError
    
    def _add_background(self) -> None:
        raise NotImplementedError

    def _add_field_lines(self) -> None:
        raise NotImplementedError

    def _add_multiple_agents(self) -> None:
        raise NotImplementedError
    
    def _add_agent(self) -> None:
        raise NotImplementedError
    
    def _add_ball() -> None:
        raise NotImplementedError
