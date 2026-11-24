"""Mass-sensor experiment shaping, isolated from the original Sweep rewards."""

import torch
from isaaclab.utils.math import matrix_from_quat


def reaching_at_object_height(env, reaching_z_offset=0.04, object_collection_cfg=None, ee_frame_cfg=None):
    """Original exp(-10 * distance) shaping with a bottom-origin height."""
    objects = env.scene["object_collection" if object_collection_cfg is None else object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target = objects.data.object_pos_w[rows, env.target_id[:, 0].long()].clone()
    target[:, 0] -= 0.02
    target[:, 1] -= env.target_width[:, 0] * torch.sign(env.sweep_dir[:, 1])
    target[:, 2] += reaching_z_offset
    eef = env.scene["ee_frame" if ee_frame_cfg is None else ee_frame_cfg.name].data.target_pos_w[:, 0]
    return torch.exp(-10.0 * torch.linalg.vector_norm(target - eef, dim=-1))


def align_eef_y_with_shelf_z(env, shelf_cfg=None, ee_frame_cfg=None):
    """Align physical tool Y (Inspire hand X) with physical shelf Z."""
    eef = env.scene["ee_frame" if ee_frame_cfg is None else ee_frame_cfg.name]
    shelf = env.scene["shelf" if shelf_cfg is None else shelf_cfg.name]
    eef_rotation = matrix_from_quat(eef.data.target_quat_w[:, 0])
    shelf_rotation = matrix_from_quat(shelf.data.default_root_state[:, 3:7])
    cosine = torch.sum(eef_rotation[..., :, 1] * shelf_rotation[..., :, 2], dim=-1)
    return torch.sign(cosine) * cosine.square()
