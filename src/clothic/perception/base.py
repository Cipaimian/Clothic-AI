"""Abstract perception backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clothic.schemas import PersonObservation


class PerceptionBackend(ABC):
    """Maps a single frame to a list of per-person observations.

    Implementations encapsulate the whole perception stack (person detect,
    track, pose, parsing, garment, attributes, exposure). The rest of Clothic AI
    only depends on the returned ``PersonObservation`` objects.
    """

    name: str = "base"

    @abstractmethod
    def observe(self, frame: Any, frame_index: int = 0) -> list[PersonObservation]:
        """Return observations for every tracked person in ``frame``.

        ``frame`` is backend-defined (an ndarray for real backends; ignored by
        the mock backend, which is driven by a scripted scenario).
        """
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass
