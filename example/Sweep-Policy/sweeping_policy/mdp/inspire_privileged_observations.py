"""Live simulator target state, available only to the privileged critic group."""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_unique, subtract_frame_transforms


def target_state_in_robot_root(
    env,
    robot_cfg=SceneEntityCfg("robot"),
    object_collection_cfg=SceneEntityCfg("object_collection"),
):
    """Target [position 3, quaternion 4, linear velocity 3, angular velocity 3].

    Position is the object link origin (the cylinder bottom), not its COM.
    All quantities are expressed in the fixed robot root frame. This reads
    Scene GT directly and applies no actor noise, filtering, or initial latch.
    """
    robot = env.scene[robot_cfg.name]
    objects = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    state = objects.data.object_link_state_w[rows, env.target_id[:, 0].long()]
    root = robot.data.root_state_w
    position, quaternion = subtract_frame_transforms(
        root[:, :3], root[:, 3:7], state[:, :3], state[:, 3:7],
    )
    linear = quat_apply_inverse(root[:, 3:7], state[:, 7:10])
    angular = quat_apply_inverse(root[:, 3:7], state[:, 10:13])
    return torch.cat((position, quat_unique(quaternion), linear, angular), dim=-1)
