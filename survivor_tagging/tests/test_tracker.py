import time

import pytest

from survivor_tagging.tracker import Tracker


def cfg(**over):
    base = {
        "tracker": {
            "gate_dist_m": 1.5,
            "hit_radius_m": 0.5,
            "confirm_hits": 2,
            "confirm_window": 3,
            "max_missed_ticks": 8,
            "max_map_units": 64,
        },
    }
    base["tracker"].update(over)
    return base


class TestAssociation:
    def test_two_close_detections_same_track(self):
        tr = Tracker(cfg())
        tr.ingest([(1.0, 1.0, 0.9, False, False, False)], now=1.0)
        tr.ingest([(1.05, 1.05, 0.9, False, False, False)], now=2.0)
        assert len(tr.active_tracks()) == 1
        (t,) = tr.active_tracks()
        assert t.hits == 2

    def test_two_far_detections_two_tracks(self):
        tr = Tracker(cfg())
        tr.ingest([(1.0, 1.0, 0.9, False, False, False)], now=1.0)
        tr.ingest([(5.0, 5.0, 0.9, False, False, False)], now=2.0)
        assert len(tr.active_tracks()) == 2

    def test_gated_but_not_hit_counts_as_miss(self):
        tr = Tracker(cfg())
        tr.ingest([(1.0, 1.0, 0.9, False, False, False)], now=1.0)
        # 1.0 m away: inside gate_dist (1.5) but outside hit_radius (0.5)
        tr.ingest([(2.0, 1.0, 0.9, False, False, False)], now=2.0)
        (t,) = tr.active_tracks()
        assert t.hits == 1
        assert t.misses == 1
        # and the far one did not spawn a competing track
        assert len(tr.active_tracks()) == 1


class TestLifecycle:
    def test_confirm_after_two_hits(self):
        tr = Tracker(cfg())
        tr.ingest([(1.0, 1.0, 0.9, False, False, False)], now=1.0)
        (t,) = tr.active_tracks()
        assert t.state == "TENTATIVE"
        tr.ingest([(1.0, 1.0, 0.9, False, False, False)], now=2.0)
        (t,) = tr.active_tracks()
        assert t.state == "CONFIRMED"

    def test_removed_after_miss_ticks(self):
        tr = Tracker(cfg(max_missed_ticks=2))
        tr.ingest([(1.0, 1.0, 0.9, False, False, False)], now=1.0)
        # last_seen stays at 1.0; 2.5 s > max_missed_ticks = 2 -> removed
        tr.ingest([], now=3.5)
        assert len(tr.active_tracks()) == 0
        assert len(tr.removed_track_ids()) == 1

    def test_withdraw_if_occupied(self):
        tr = Tracker(cfg())
        tr.ingest([(3.5, 2.5, 0.9, False, False, False)], now=1.0)
        tr.ingest([(3.5, 2.5, 0.9, False, False, False)], now=2.0)
        # pretend the map cell at (3, 2) became OCCUPIED
        withdrawn = tr.withdraw_if_occupied(lambda c, r: (c, r) == (3, 2))
        assert withdrawn and withdrawn[0].startswith("tag_")
        assert len(tr.active_tracks()) == 0
