"""URI-safe paths for the external Shelf USD asset collection."""

from __future__ import annotations

import os
import re
from pathlib import Path


DEFAULT_ASSET_ROOT = str(Path(__file__).resolve().parents[4] / "assets" / "shelf")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _is_uri(value: str | os.PathLike[str]) -> bool:
    """Return whether *value* is an absolute URI rather than a local path."""

    return _URI_PATTERN.match(os.fspath(value)) is not None


def _asset_root() -> str:
    """Return the configured asset root without damaging URI separators."""

    configured_root = os.environ.get("SWEEP_POLICY_ASSET_ROOT") or os.environ.get("INSPIRE_SWEEP_ASSET_ROOT")
    root = configured_root if configured_root else DEFAULT_ASSET_ROOT
    if _is_uri(root):
        return root.rstrip("/")
    return str(Path(root).expanduser().resolve())


ASSET_ROOT = _asset_root()
ENVIRONMENT_YAML_PATH = Path(__file__).with_name("environment.yaml")


def _path_below_asset_root(relative_path: str | os.PathLike[str]) -> str:
    """Join a relative asset path to either a URI or local asset root."""

    suffix = os.fspath(relative_path)
    if _is_uri(ASSET_ROOT):
        # pathlib collapses ``omniverse://`` to ``omniverse:/``.  Join URI
        # components as POSIX-style strings so the Nucleus address is intact.
        suffix = suffix.replace("\\", "/").lstrip("/")
        return f"{ASSET_ROOT}/{suffix}"
    return str(Path(ASSET_ROOT) / suffix)


def _configured_asset_path(environment_variable: str, relative_path: str) -> str:
    """Return an individual override or a path below :data:`ASSET_ROOT`."""

    configured_path = os.environ.get(environment_variable)
    if not configured_path:
        return _path_below_asset_root(relative_path)
    if _is_uri(configured_path):
        return configured_path
    return str(Path(configured_path).expanduser())


ROBOT_USD_PATH = _configured_asset_path(
    "SWEEP_POLICY_ROBOT_USD_PATH",
    "Robots/UR5e/Collected_UR5e_v4/UR5e_v4.usd",
)

SHELF_USD_PATH = _configured_asset_path(
    "SWEEP_POLICY_SHELF_USD_PATH",
    "Arena/Collected_speedrack_shape/speedrack_shape.usd",
)

TABLE_USD_PATH = _configured_asset_path(
    "SWEEP_POLICY_TABLE_USD_PATH",
    "Arena/Collected_thor_table/thor_table.usd",
)


def object_usd_path(relative_path: str | os.PathLike[str]) -> str:
    """Resolve an object USD path against the configured asset root.

    Nucleus URIs are returned without an eager network request; Isaac Lab
    validates them when it creates the USD reference.  Local-path overrides
    retain the previous eager file-existence check.
    """

    value = os.fspath(relative_path)
    if _is_uri(value):
        return value

    path = Path(value).expanduser()
    if not path.is_absolute() and _is_uri(ASSET_ROOT):
        return _path_below_asset_root(value)

    path = path if path.is_absolute() else Path(ASSET_ROOT) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Object USD asset does not exist: {path}")
    return str(path)
