#!/usr/bin/env python3
"""Register Sweep-Policy tasks and run Isaac Lab's RSL-RL player."""

import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweeping_policy  # noqa: E402,F401

root = Path(os.environ.get("ISAACLAB_PATH", str(Path(__file__).resolve().parents[2] / "IsaacLab")))
script = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
if not script.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL player does not exist: {script}")
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
