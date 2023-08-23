from abc import ABC, abstractmethod
from rsoccer_gym.Entities import Frame

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
