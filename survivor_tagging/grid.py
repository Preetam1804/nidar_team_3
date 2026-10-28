"""Grid indexing and occupancy helpers for the survivor_tagging module.

Conventions (frozen in the plan and in the README):
* Map frame is right-handed, z-up, metres.
* Cell (col, row) centre is at map-frame position
      x = origin_x + (col + 0.5) * res
      y = origin_y + (row + 0.5) * res
* Cell (0, 0) is the map's lower-left cell; col grows along +x and
  row grows along +y.
* Cell index into the row-major data array: index = row * w + col.
* Scoring grid uses res_m = 1.0 (mission brief: 1 m x 1 m reference
  grid); the raw Team 1 map may have a different res and origin, so all
  conversions here are explicit about which grid is being used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import schemas


@dataclass(frozen=True)
class GridSpec:
    """Geometry of a rectangular occupancy grid in the map frame."""
    w: int
    h: int
    origin_x: float
    origin_y: float
    res: float

    def cell_center(self, col: int, row: int) -> tuple[float, float]:
        return (
            self.origin_x + (col + 0.5) * self.res,
            self.origin_y + (row + 0.5) * self.res,
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Map-frame (x, y) -> (col, row).  Returns (-1, -1) if outside."""
        col = int(math.floor((x - self.origin_x) / self.res))
        row = int(math.floor((y - self.origin_y) / self.res))
        if not (0 <= col < self.w and 0 <= row < self.h):
            return (-1, -1)
        return (col, row)

    def contains(self, col: int, row: int) -> bool:
        return 0 <= col < self.w and 0 <= row < self.h

    def cell_bounds(self, col: int, row: int) -> tuple[float, float, float, float]:
        """min_x, min_y, max_x, max_y of the cell footprint."""
        return (
            self.origin_x + col * self.res,
            self.origin_y + row * self.res,
            self.origin_x + (col + 1) * self.res,
            self.origin_y + (row + 1) * self.res,
        )


def grid_from_msg(msg: schemas.OccupancyGridMsg) -> GridSpec:
    return GridSpec(w=msg.w, h=msg.h, origin_x=msg.origin_x, origin_y=msg.origin_y, res=msg.res)


def cell_state(grid_msg: schemas.OccupancyGridMsg, col: int, row: int) -> int:
    """3-state value for the cell, or UNKNOWN if out of bounds."""
    idx = row * grid_msg.w + col
    if idx < 0 or idx >= len(grid_msg.data):
        return schemas.UNKNOWN
    return grid_msg.data[idx]


def snap_to_free(
    grid_msg: schemas.OccupancyGridMsg,
    col: int,
    row: int,
    max_radius: int = 2,
) -> tuple[int, int, bool]:
    """Snap (col, row) to the nearest explored-FREE cell within radius.

    Returns (col, row, snapped_flag).
    If the input cell is already FREE, returns it unchanged with flag
    False (no snap needed).
    If nothing FREE is within max_radius, returns the input unchanged
    with flag True — the caller must decide whether to publish a snap-
    questionable tag.
    """
    if cell_state(grid_msg, col, row) == schemas.FREE:
        return (col, row, False)
    best: tuple[int, int] | None = None
    best_d = None
    for dr in range(-max_radius, max_radius + 1):
        for dc in range(-max_radius, max_radius + 1):
            cc, rr = col + dc, row + dr
            if not grid_from_msg(grid_msg).contains(cc, rr):
                continue
            if cell_state(grid_msg, cc, rr) != schemas.FREE:
                continue
            d = dc * dc + dr * dr
            if best_d is None or d < best_d:
                best_d = d
                best = (cc, rr)
    if best is None:
        return (col, row, True)   # nothing to snap onto; flag urged
    return (best[0], best[1], True)


def nearest_free_cell(
    grid_msg: schemas.OccupancyGridMsg,
    x: float,
    y: float,
    max_radius: int = 2,
) -> tuple[int, int, float, bool]:
    """Snap a map-frame position to the nearest FREE cell.

    Returns (col, row, snap_distance_m, was_snapped).
    was_snapped is True when the projected position needed to move.
    """
    col, row = grid_from_msg(grid_msg).world_to_cell(x, y)
    if col < 0:
        return (-1, -1, 0.0, True)
    cx, cy = grid_from_msg(grid_msg).cell_center(col, row)
    ccol, crow, _ = snap_to_free(grid_msg, col, row, max_radius)
    fcol, frow = grid_from_msg(grid_msg).cell_center(ccol, crow)
    dist = math.hypot(fcol - cx, frow - cy)
    return (ccol, crow, dist, dist > 1e-9)
