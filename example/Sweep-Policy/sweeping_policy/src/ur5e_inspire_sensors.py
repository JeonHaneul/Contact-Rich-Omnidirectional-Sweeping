"""v2 sensor assembly reusing the workspace's physical Inspire/Axia implementation."""

from pathlib import Path
import sys

# Sweep-Policy's launcher need not install the sibling project as a package.
_HAND_SOURCE = Path(__file__).resolve().parents[4] / "src" / "trash" / "hand_manipulation_rl"
if not (_HAND_SOURCE / "hand_manipulation_rl" / "sensors.py").is_file():
    raise FileNotFoundError(f"Inspire sensor implementation is missing: {_HAND_SOURCE}")
if str(_HAND_SOURCE) not in sys.path:
    sys.path.insert(0, str(_HAND_SOURCE))

from hand_manipulation_rl.assets.robot import Ur5eAxia80InspireSpawnerCfg
from hand_manipulation_rl.sensors import (
    FixedJointWrenchReader,
    PALM_CHANNEL_NAMES,
    contact_magnitudes_and_bits,
)

from .ur5e_inspire_hand import make_movable_inspire_cfg


def make_sensor_inspire_cfg():
    """Insert a cylindrical F/T housing; retain v1 arm and hand actuator settings."""
    cfg = make_movable_inspire_cfg()
    cfg.spawn = Ur5eAxia80InspireSpawnerCfg(
        usd_path=cfg.spawn.usd_path,
        rigid_props=cfg.spawn.rigid_props.copy(),
        articulation_props=cfg.spawn.articulation_props.copy(),
    )
    return cfg
