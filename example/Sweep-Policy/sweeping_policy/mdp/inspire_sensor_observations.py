"""Physical tactile bits and uncancelled F/T, used only as v2 observations."""

import torch
import isaaclab.utils.math as math_utils

from ..src.ur5e_inspire_sensors import (
    FixedJointWrenchReader,
    PALM_CHANNEL_NAMES,
    contact_magnitudes_and_bits,
)


def palm_tactile_bits(env, sensor_name="palm_tactile", threshold_n=0.05):
    """17 physical pad contacts, palm/thumb/index/middle/ring/little order."""
    fresh = env.sensor_data_fresh
    if not bool(fresh.any()):
        return torch.zeros((env.num_envs, 17), device=env.device)
    sensor = env.scene[sensor_name]
    cache = getattr(env, "_inspire_palm_indices", None)
    if cache is None:
        cache = env._inspire_palm_indices = {}
    if sensor_name not in cache:
        names = tuple(sensor.body_names)
        if len(names) != 17 or len(set(names)) != 17 or set(names) != set(PALM_CHANNEL_NAMES):
            raise RuntimeError(f"Expected the 17 physical Inspire pads, got {names}")
        cache[sensor_name] = torch.tensor(
            [names.index(name) for name in PALM_CHANNEL_NAMES], device=env.device, dtype=torch.long,
        )
    forces = torch.index_select(sensor.data.net_forces_w, 1, cache[sensor_name])
    # Reset rows may contain stale backend samples; clear them before validation.
    forces = torch.where(fresh[:, None, None], forces, torch.zeros_like(forces))
    _, bits = contact_magnitudes_and_bits(forces, threshold_n)
    return bits.to(dtype=torch.float32)


def wrist_wrench_eef(env, asset_name="robot", frame_name="ee_frame"):
    """[Fx,Fy,Fz,Mx,My,Mz], N/Nm, expressed and referenced at the EEF.

    Read the fixed-joint reaction on the hand child body in the Axia parent
    frame, rotate into the tool axes, and transport its moment reference.
    No gravity cancellation, filtering, or reward-dependent processing.
    """
    fresh = env.sensor_data_fresh
    if not bool(fresh.any()):
        return torch.zeros((env.num_envs, 6), device=env.device)
    robot = env.scene[asset_name]
    cache = getattr(env, "_inspire_wrench_readers", None)
    if cache is None:
        cache = env._inspire_wrench_readers = {}
    if asset_name not in cache:
        names = tuple(robot.body_names)
        if names.count("inspire_base_link") != 1 or names.count("axia80_link") != 1:
            raise RuntimeError("Expected one Inspire base and one Axia80 sensor body")
        cache[asset_name] = (
            FixedJointWrenchReader(robot, "inspire_base_link"), names.index("axia80_link"),
        )
    reader, f_body_id = cache[asset_name]
    frame = env.scene[frame_name].data
    c_pos_w = frame.target_pos_w[:, 0]
    c_quat_w = frame.target_quat_w[:, 0]
    f_pos_w = robot.data.body_pos_w[:, f_body_id]
    f_quat_w = robot.data.body_quat_w[:, f_body_id]
    rotation_c_from_f = (
        math_utils.matrix_from_quat(c_quat_w).transpose(-1, -2)
        @ math_utils.matrix_from_quat(f_quat_w)
    )
    c_to_f_c = math_utils.quat_apply_inverse(c_quat_w, f_pos_w - c_pos_w)
    result = reader.read(rotation_c_from_f, c_to_f_c).measured_c
    return torch.where(fresh[:, None], result, torch.zeros_like(result))


def palm_contact_bit(env, sensor_name="palm_tactile", threshold_n=0.05):
    """One actual palm pad only; finger and dorsal contacts are excluded."""
    fresh = env.sensor_data_fresh
    if not bool(fresh.any()):
        return torch.zeros((env.num_envs, 1), device=env.device)
    sensor = env.scene[sensor_name]
    if tuple(sensor.body_names) != ("inspire_palm_force_sensor",):
        raise RuntimeError(f"Expected only the palm pad, got {sensor.body_names}")
    forces = torch.where(
        fresh[:, None, None], sensor.data.net_forces_w,
        torch.zeros_like(sensor.data.net_forces_w),
    )
    _, bits = contact_magnitudes_and_bits(forces, threshold_n)
    return bits.to(dtype=torch.float32)
