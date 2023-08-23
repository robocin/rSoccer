from dataclasses import Field
from typing import Final
from rsoccer_gym.Render.BaseRender import BaseRender, RenderParam2D

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
        self.screen_dimensions = None
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
