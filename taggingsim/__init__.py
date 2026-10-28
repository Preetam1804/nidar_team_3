"""Simulator package for the survivor_tagging benchmark harness."""
from .cli import main
from .sim import run_mission

__all__ = ["run_mission", "main"]
