from abc import ABC, abstractmethod
from dataclasses import dataclass
from rsoccer_gym.Entities import Frame, Field

class BaseRender(ABC):
    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    def render_frame(self, frame: Frame, return_rgb_array: bool = False) -> None:
        pass

    @abstractmethod
    def _add_background(self) -> None:
        pass

    @abstractmethod
    def _add_field_lines(self) -> None:
        pass

    @abstractmethod
    def _add_multiple_agents(self) -> None:
        pass

    @abstractmethod
    def _add_agent(self) -> None:
        pass

    @abstractmethod
    def _add_ball() -> None:
        pass

@dataclass
class RenderParam2D:
    DEFAULT_WIDTH: int = 1080
    DEFAULT_HEIGHT: int = 720