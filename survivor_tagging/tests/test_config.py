import os
from pathlib import Path

import pytest
import yaml

from survivor_tagging.config import (
    ENV_PREFIX, _env_overrides, _parse_scalar, load_config
)


class TestParsing:
    def test_scalar_parse(self):
        assert _parse_scalar("0.5") == 0.5
        assert _parse_scalar("true") is True
        assert _parse_scalar("str") == "str"
        assert _parse_scalar("[1,2]") == [1, 2]


class TestEnvOverrides:
    def test_simple_top_level(self, monkeypatch):
        monkeypatch.setenv("SURVIVOR_TAGGING_TAGS_MIN_PUBLISH_CONF", "0.5")
        tree, extras = _env_overrides()
        assert tree == {"tags": {"min_publish_conf": 0.5}}
        assert extras == {}

    def test_deep_key(self, monkeypatch):
        monkeypatch.setenv("SURVIVOR_TAGGING_TRACKER_GATE_DIST_M", "2.5")
        tree, _ = _env_overrides()
        assert tree == {"tracker": {"gate_dist_m": 2.5}}

    def test_unknown_key_goes_to_extras(self, monkeypatch):
        monkeypatch.setenv("SURVIVOR_TAGGING_ERO_JAMOOM_XXX", "3")
        tree, extras = _env_overrides()
        assert tree == {}
        assert extras == {"ero.jamoom.xxx": 3}

    def test_full_pipeline(self, monkeypatch, cfg):
        monkeypatch.setenv("SURVIVOR_TAGGING_TRACKER_HIT_RADIUS_M", "0.9")
        c = load_config(None)
        assert c["tracker"]["hit_radius_m"] == 0.9


class TestLoadConfig:
    def test_defaults_when_no_file(self):
        c = load_config(None)
        assert c["tracker"]["confirm_hits"] == 2
        assert c["grid"]["res_m"] == 1.0

    def test_yaml_file_merges(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump({"tracker": {"confirm_hits": 5}}))
        c = load_config(str(p))
        assert c["tracker"]["confirm_hits"] == 5
        assert c["tracker"]["gate_dist_m"] == 1.5   # default kept
