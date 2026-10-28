import pytest

from survivor_tagging import grid, schemas
from survivor_tagging.grid import GridSpec


class TestGridSpec:
    def test_zero_cell(self):
        g = GridSpec(w=10, h=10, origin_x=0.0, origin_y=0.0, res=1.0)
        assert g.world_to_cell(0.5, 0.5) == (0, 0)
        assert g.cell_center(0, 0) == (0.5, 0.5)

    def test_negative_origin(self):
        g = GridSpec(w=10, h=10, origin_x=-5.0, origin_y=-5.0, res=1.0)
        assert g.world_to_cell(0.0, 0.0) == (5, 5)
        assert g.world_to_cell(-4.5, -4.5) == (0, 0)

    def test_out_of_bounds(self):
        g = GridSpec(w=10, h=10, origin_x=0.0, origin_y=0.0, res=1.0)
        assert g.world_to_cell(15.0, 5.0) == (-1, -1)
        assert g.world_to_cell(-0.1, 5.0) == (-1, -1)
        assert g.world_to_cell(5.0, -0.1) == (-1, -1)

    def test_cell_bounds(self):
        g = GridSpec(w=10, h=10, origin_x=0.0, origin_y=0.0, res=1.0)
        assert g.cell_bounds(1, 1) == (1.0, 1.0, 2.0, 2.0)


class TestSnap:
    def _map(self, occupied_cells):
        w, h = 5, 5
        data = [schemas.FREE] * (w * h)
        for (c, r) in occupied_cells:
            data[r * w + c] = schemas.OCCUPIED
        return schemas.OccupancyGridMsg(
            w=w, h=h, origin_x=0.0, origin_y=0.0, res=1.0, data=tuple(data)
        )

    def test_no_snap_needed_on_free(self):
        m = self._map([])
        c, r, snapped = grid.snap_to_free(m, 2, 2)
        assert (c, r, snapped) == (2, 2, False)

    def test_snaps_occupied_to_nearest_free(self):
        m = self._map([(2, 2)])
        c, r, snapped = grid.snap_to_free(m, 2, 2, max_radius=1)
        assert snapped is True
        assert (c, r) != (2, 2)

    def test_no_free_within_radius_returns_original_flagged(self):
        # Occupy a full 5-radius block around origin.
        occupied = [(c, r) for c in range(5) for r in range(5)]
        m = self._map(occupied)
        c, r, snapped = grid.snap_to_free(m, 0, 0, max_radius=2)
        assert snapped is True

    def test_nearest_free_cell_distance(self):
        m = self._map([(1, 1)])
        c, r, dist, snapped = grid.nearest_free_cell(m, 1.5, 1.5)
        assert snapped is True
        assert c >= 0
