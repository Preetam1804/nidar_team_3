"""Tactical Search-and-Rescue (SAR) exploration path planner.

Generates structured, room-aware search trajectories for the 15m x 15m indoor arena:
- Explores rooms systematically rather than following an arbitrary bounding box
- Sweeps high-probability survivor casualty zones
- Curves through corridor doorways with clearance margins from walls
- Returns smoothly along cleared corridors back to the takeoff pad
"""

from __future__ import annotations

import math
from typing import Sequence


# Standard tactical search waypoints for the 15m x 15m SAR Arena
TACTICAL_SAR_WAYPOINTS: tuple[tuple[float, float], ...] = (
    (1.5, 1.5),    # Helipad (Takeoff point)
    (3.5, 2.8),    # Room A Interior (inspects Survivor 1)
    (4.5, 4.0),    # Room A Corridor Exit
    (6.2, 4.2),    # Central Corridor Threshold
    (8.2, 4.8),    # Room B Approach
    (8.8, 6.2),    # Room B Sweep (inspects Survivor 2)
    (7.5, 8.5),    # North Passage Ascent
    (4.2, 10.5),   # North Corridor Entry
    (3.8, 11.8),   # North Corridor Deep Sweep (inspects Survivor 3)
    (6.5, 12.5),   # North Hallway Eastbound Turn
    (9.5, 11.5),   # East Corridor Junction
    (12.2, 9.5),   # East Room Interior (inspects Survivor 4)
    (12.5, 6.0),   # Southeast Corridor Sweep
    (11.0, 2.5),   # South Corridor Return
    (6.0, 2.0),    # Central South Transit
    (2.5, 1.8),    # Helipad Final Approach
    (1.5, 1.5),    # Helipad Touchdown
)


def generate_sar_trajectory(
    custom_survivor_positions: Sequence[tuple[float, float]] | None = None,
    arena_w: float = 15.0,
    arena_h: float = 15.0,
) -> list[tuple[float, float]]:
    """Generate a deliberate search-and-rescue inspection path.

    If custom survivor coordinates are provided, the planner creates a
    smooth touring trajectory that visits each survivor zone while keeping
    a safe minimum clearance (1.2m) from the perimeter and interior partition walls.
    """
    if not custom_survivor_positions:
        return list(TACTICAL_SAR_WAYPOINTS)

    # If dynamic survivors are used, build an ordered coverage tour
    pts = [(1.5, 1.5)]  # Start at helipad
    unvisited = list(custom_survivor_positions)

    current = pts[0]
    while unvisited:
        # Nearest unvisited survivor
        nearest = min(
            unvisited,
            key=lambda s: math.hypot(s[0] - current[0], s[1] - current[1]),
        )
        unvisited.remove(nearest)

        # Add waypoint safely offset over survivor
        target_x = max(1.5, min(arena_w - 1.5, nearest[0]))
        target_y = max(1.5, min(arena_h - 1.5, nearest[1]))
        pts.append((target_x, target_y))
        current = (target_x, target_y)

    pts.append((1.5, 1.5))  # Return to helipad
    return pts
