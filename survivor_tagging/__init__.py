"""survivor_tagging — NiDAR AirMouse Team 3 module.

Survivor localisation and map tagging: consumes detections (Team 2),
pose (Team 5) and the occupancy map (Team 1), and emits map-frame grid
cells with confidence to the GCS.

Public API:
    SurvivorTaggingPipeline  - core orchestration
    load_config              - config yaml + ENV overrides
    adapters                 - in-memory / JSONL playback providers
"""

from .config import load_config
from .pipeline import SurvivorTaggingPipeline

__all__ = ["SurvivorTaggingPipeline", "load_config"]
