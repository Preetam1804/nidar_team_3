import copy

import pytest

from survivor_tagging.config import load_config


@pytest.fixture
def cfg() -> dict:
    """A valid config with sensible synthetic intrinsics."""
    c = load_config(None)
    c = copy.deepcopy(c)
    cam = c["camera"]
    cam["fx"] = cam["fy"] = 700.0
    cam["cx"] = 768.0
    cam["cy"] = 432.0
    cam["dist_model"] = "none"
    cam["dist_params"] = []
    return c
