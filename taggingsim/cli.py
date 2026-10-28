"""CLI for the taggingsim harness.

Usage:
    python -m taggingsim --help
    python -m taggingsim run --seed 7 --survivors 6
    python -m taggingsim --playback mission.jsonl
    python -m taggingsim soak --minutes 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from survivor_tagging.adapters import JSONLPlaybackAdapter
from survivor_tagging.config import load_config
from survivor_tagging.pipeline import SurvivorTaggingPipeline
from .sim import SimConfig, run_mission, run_soak


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "command", nargs="?", default="run",
        choices=["run", "soak", "playback"],
        help="Command to run: 'run' (default), 'soak', or 'playback'",
    )
    p.add_argument(
        "--config", default=None,
        help="Path to config.yaml (defaults to the shipped one)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible benchmark runs",
    )
    p.add_argument(
        "--survivors", type=int, default=6,
        help="Number of synthetic survivors in the run",
    )
    p.add_argument(
        "--playback", default=None,
        help="Path to JSONL log file for playback mode",
    )
    p.add_argument(
        "--soak", action="store_true",
        help="Run 30-minute soak test",
    )
    p.add_argument(
        "--minutes", type=float, default=30.0,
        help="Duration of soak in simulated minutes (default: 30.0)",
    )
    p.add_argument(
        "--out", default=None, help="Path to write the JSON report",
    )
    p.add_argument(
        "--gui", action="store_true",
        help="Launch the interactive real-time visual simulation dashboard",
    )
    p.add_argument(
        "--save-anim", default=None,
        help="Save simulation animation to file (.gif or .mp4) instead of showing window",
    )


def run_playback_mode(playback_path: str, cfg: dict) -> dict:
    """Execute playback of recorded JSONL frames through the pipeline."""
    adapter = JSONLPlaybackAdapter(playback_path)
    pipeline_cfg = dict(cfg)
    cam = dict(pipeline_cfg.get("camera", {}))
    if cam.get("fx", 0.0) == 0.0:
        w = cam.get("width", 1536)
        h = cam.get("height", 864)
        fov = float(cfg.get("sim", {}).get("fov_deg", 120.0))
        fx_fy = (w / 2.0) / math.tan(math.radians(fov / 2.0))
        cam["fx"] = cam["fy"] = fx_fy
        cam["cx"] = w / 2.0
        cam["cy"] = h / 2.0
        cam["dist_model"] = "none"
        pipeline_cfg["camera"] = cam

    pipeline = SurvivorTaggingPipeline(
        pipeline_cfg, adapter.detectors, adapter.pose, adapter.map
    )
    stamps = sorted({float(r.get("stamp", 0.0)) for r in adapter._rows})
    t_start = stamps[0] if stamps else 0.0
    t_end = stamps[-1] if stamps else 0.0
    gcs_payloads: list[dict] = []

    t = t_start
    dt = 0.05
    while t <= t_end + 0.001:
        adapter.advance_to(t)
        payloads = pipeline.tick()
        gcs_payloads.extend(payloads)
        t += dt

    return {
        "mode": "playback",
        "playback_file": str(playback_path),
        "total_rows": len(adapter._rows),
        "metrics": pipeline.metrics.summary(),
        "gcs_payload_count": len(gcs_payloads),
        "gcs_payloads": gcs_payloads,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="taggingsim",
        description="Synthetic mission benchmark for survivor_tagging",
    )
    _add_common(ap)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    # 1. Playback mode (--playback <file> or 'playback' positional command)
    playback_file = args.playback
    if args.command == "playback" and not playback_file:
        ap.error("playback mode requires --playback <file.jsonl>")
    if playback_file:
        report = run_playback_mode(playback_file, cfg)
        text = json.dumps(report, indent=2, default=list)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"report written to {args.out}")
        else:
            print(text)
        print(f"Playback complete: {report['gcs_payload_count']} tags emitted from {report['total_rows']} log events.")
        return 0

    # 2. Soak mode (--soak or 'soak' positional command)
    if args.command == "soak" or args.soak:
        simcfg = SimConfig(seed=args.seed, n_survivors=args.survivors)
        duration_s = float(args.minutes) * 60.0
        report = run_soak(simcfg, cfg, duration_s=duration_s)
        text = json.dumps(report, indent=2, default=list)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"report written to {args.out}")
        else:
            print(text)
        print(
            f"Soak benchmark complete: {report['simulated_minutes']:.1f} simulated minutes "
            f"({report['ticks']} ticks), max active tracks: {report['max_active_tracks']}/"
            f"{report['max_allowed_tracks']} (bounded={report['bounded_memory']}), "
            f"payloads={report['payload_count']}."
        )
        return 0

    # 3. Standard synthetic mission run
    simcfg = SimConfig(seed=args.seed, n_survivors=args.survivors)

    if args.gui or args.save_anim:
        from .gui import MissionVisualizer
        vis = MissionVisualizer(simcfg, cfg, fps=15, speedup=1.5)
        vis.run(save_path=args.save_anim)
        return 0

    result = run_mission(simcfg, cfg)

    report = {
        "sim_config": {
            "seed": simcfg.seed,
            "n_survivors": simcfg.n_survivors,
            "arena_w": simcfg.arena_w,
            "arena_h": simcfg.arena_h,
            "pose_sigma_xy": simcfg.pose_sigma_xy,
        },
        "per_survivor": [
            {"sid": k if isinstance(k, int) else -1,
             "n_tags": v["n_tags"],
             "correct": v["correct"],
             "incorrect": v["incorrect"],
             "cells": [list(c) for c in sorted(v["cells"])]}
            for k, v in result.per_survivor.items()
        ],
        "metrics": result.metrics_summary,
        "gcs_payload_count": len(result.gcs_payloads),
    }
    text = json.dumps(report, indent=2, default=list)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(text)
    correct = report["metrics"]["sim_scoring"]["correct"] or 0
    scoreable = correct + (report["metrics"]["sim_scoring"]["incorrect"] or 0)
    rate = (correct / scoreable) if scoreable else 0.0
    print(
        f"correct-cell rate: {rate:.1%} "
        f"({correct}/{scoreable} tags)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
