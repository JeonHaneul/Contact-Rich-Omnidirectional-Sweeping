"""Central Omniverse USD paths and object metadata for the shelf scene."""

from __future__ import annotations

import os

from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR
from pathlib import Path as _AssetPath
from os import environ as _asset_environment

SHELF_ASSET_ROOT = _asset_environment.get("INSPIRE_SWEEP_ASSET_ROOT") or str(_AssetPath(__file__).resolve().parents[5] / "assets" / "shelf")


# -----------------------------------------------------------------------------
# USD path configuration: edit only this section when assets are supplied.
# Environment variables remain available for machine-specific overrides.
# -----------------------------------------------------------------------------
UR5E_USD_PATH = os.environ.get(
    "SWEEP_SHELVE_FORCE_UR5E_USD_PATH",
    (
        f"{NUCLEUS_ASSET_ROOT_DIR}/"
        "Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
    ),
)
ROBOTIQ_USD_PATH = os.environ.get(
    "SWEEP_SHELVE_FORCE_ROBOTIQ_USD_PATH",
    (
        f"{NUCLEUS_ASSET_ROOT_DIR}/"
        "Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
    ),
)
SHELF_USD_PATH = f"{SHELF_ASSET_ROOT}/Arena/Collected_speedrack_shape/speedrack_shape.usd"

OBJECT_NAMES = ("bottle_1", "cup_1", "cup_2", "mug_1", "mug_2", "can_1")

OBJECT_USD_PATHS = {
    "bottle_1": f"{SHELF_ASSET_ROOT}/Objects/Bottle_6/Bottle_6.usd",
    "cup_1": f"{SHELF_ASSET_ROOT}/Objects/Collected_Cup_1/Cup_1.usd",
    "cup_2": f"{SHELF_ASSET_ROOT}/Objects/Collected_Cup_4/Cup_4.usd",
    "mug_1": f"{SHELF_ASSET_ROOT}/Objects/Collected_Mug_2/Mug_2.usd",
    "mug_2": f"{SHELF_ASSET_ROOT}/Objects/Collected_Mug_3/Mug_3.usd",
    "can_1": f"{SHELF_ASSET_ROOT}/Objects/Can_6/Can_6.usd",
}

# Original example arrangement: two x rows by three y columns.
OBJECT_POSES = {
    "bottle_1": (-0.75, -0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
    "cup_1": (-0.75, 0.00, 1.05, 1.0, 0.0, 0.0, 0.0),
    "cup_2": (-0.75, 0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
    "mug_1": (-0.60, -0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
    "mug_2": (-0.60, 0.00, 1.05, 1.0, 0.0, 0.0, 0.0),
    "can_1": (-0.60, 0.20, 1.05, 1.0, 0.0, 0.0, 0.0),
}

OBJECT_WIDTHS = {
    "bottle_1": 0.06,
    "cup_1": 0.06,
    "cup_2": 0.05,
    "mug_1": 0.09,
    "mug_2": 0.09,
    "can_1": 0.06,
}

OBJECT_WIDTH_TUPLE = tuple(OBJECT_WIDTHS[name] for name in OBJECT_NAMES)
OBJECT_POSE_TUPLE = tuple(OBJECT_POSES[name] for name in OBJECT_NAMES)
