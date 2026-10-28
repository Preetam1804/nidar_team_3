"""Provider interfaces for the survivor_tagging module.

These abstract the inputs Teams 1/2/5 deliver.  Core module code (the
pipeline) depends only on these interfaces, not on any transport, so
real-team-module adapters can be swapped in at integration time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import schemas


@dataclass(frozen=True)
class FrameHandle:
    """Identifies a detection frame that has been consumed by the pipeline."""
    frame_id: str
    stamp: float


class DetectionsProvider(ABC):
    """Provides the latest camera-frame detection frame (Team 2)."""

    @abstractmethod
    def latest_frame(self) -> schemas.DetectionFrame | None:
        """The newest detection frame, or None if none is available.

        Called at most once per pipeline tick.  Should not block.
        """

    @abstractmethod
    def ack(self, frame_id: str) -> None:
        """Mark a frame consumed (so this same frame is not re-ingested)."""


class PoseProvider(ABC):
    """Provides Team 5 pose samples in the map frame."""

    @abstractmethod
    def latest(self) -> schemas.PoseSample | None:
        """The newest pose sample (no guarantee of age), or None."""


class MapProvider(ABC):
    """Provides Team 1's occupancy grid in the map frame."""

    @abstractmethod
    def latest(self) -> schemas.OccupancyGridMsg | None:
        """The newest map, or None if none is available."""
