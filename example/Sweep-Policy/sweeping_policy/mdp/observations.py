# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation, RigidObjectCollection
from isaaclab.utils.math import subtract_frame_transforms, quat_unique, matrix_from_quat
from isaaclab.sensors import FrameTransformerData, ContactSensorData
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils

from random import choice

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def MA_joint_pos_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(f"joint: {asset.data.joint_pos[:, :6]}")
    # print(f"joint: {asset.data.applied_torque[: , :6]}")
    return asset.data.joint_pos[:, :8] - asset.data.default_joint_pos[:, :8]


def MA_joint_vel_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(f"joint velocity: {asset.data.joint_vel[:, :6]}")
    return asset.data.joint_vel[:, :6] - asset.data.default_joint_vel[:, :6]


def MA_object_position_in_RRF(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
) -> torch.Tensor:
    """
    The position of the target object in the robot's root frame.
    """

    robot: RigidObject = env.scene[robot_cfg.name]
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    env_indices = torch.arange(env.scene.num_envs, device=target_ids.device)
    target_state_w = object_collection.data.object_state_w[env_indices, target_ids]

    # Compute the position of the target object relative to the robot
    # (Transforms the target object’s position from the world frame to the robot’s base frame)
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],  # Robot's position in the world frame
        robot.data.root_state_w[
            :, 3:7
        ],  # Robot's orientation in the world frame (quaternion)
        target_state_w[:, :3],  # Target object's position in the world frame
    )
    # print(f"object: {object_pos_b}")
    return object_pos_b


def MA_initial_object_position_in_RRF(
    env: ManagerBasedRLEnv,
    command_name: str = MISSING,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the episode-initial target position in the robot root frame.

    The dynamic goal command snapshots the selected target state once when an
    environment resets.  Reading that snapshot, instead of the live object
    state, keeps this observation fixed while rewards can continue to use the
    current simulator state.
    """

    robot: RigidObject = env.scene[robot_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_init_pos_w = command_term.target_init_state_w[..., :3]

    target_init_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        target_init_pos_w,
    )
    return target_init_pos_b


class InitialTargetPositionInRRF(ManagerTermBase):
    """Latch the initial target position once per environment reset."""

    def __init__(self, cfg: ObsTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)
        self._initial_target_pos_b = torch.zeros((env.num_envs, 3), device=env.device)
        self._needs_update = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        """Mark reset environments to capture their next initial value."""

        # Isaac Lab resets the observation manager before the command manager.
        # Defer capture until __call__, after the command has its new snapshot.
        if env_ids is None:
            self._needs_update.fill_(True)
        else:
            self._needs_update[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = MISSING,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        env_ids = self._needs_update.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            initial_target_pos_b = MA_initial_object_position_in_RRF(
                env,
                command_name=command_name,
                robot_cfg=robot_cfg,
            )
            self._initial_target_pos_b[env_ids] = initial_target_pos_b[env_ids]
            self._needs_update[env_ids] = False

        return self._initial_target_pos_b.clone()


def MA_object_position_in_EEF(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
) -> torch.Tensor:
    """Return the target-object position expressed in the end-effector frame."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    target_ids = env.target_id.squeeze(-1).long()
    env_indices = torch.arange(env.scene.num_envs, device=target_ids.device)
    target_pos_w = object_collection.data.object_state_w[env_indices, target_ids, :3]

    target_pos_eef, _ = subtract_frame_transforms(
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :],
        target_pos_w,
    )

    # ===== DEBUG : Visualize all object frames in the environment =====

    # if not hasattr(env, "_object_frame_markers"):
    #     marker_cfg = FRAME_MARKER_CFG.copy()
    #     marker_cfg.prim_path = "/Visuals/ObjectCollectionFrames"
    #     marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)

    #     env._object_frame_markers = VisualizationMarkers(marker_cfg)

    # object_pos_w = object_collection.data.object_link_pos_w
    # object_quat_w = object_collection.data.object_link_quat_w

    # object_pos_w_flat = object_pos_w.reshape(-1, 3)
    # object_quat_w_flat = object_quat_w.reshape(-1, 4)

    # env._object_frame_markers.visualize(
    #     translations=object_pos_w_flat,
    #     orientations=object_quat_w_flat,
    # )

    # ===== DEBUG : Visualize world frame =====

    # if not hasattr(env, "_world_frame_marker"):
    #     world_marker_cfg = FRAME_MARKER_CFG.copy()
    #     world_marker_cfg.prim_path = "/Visuals/WorldFrame"
    #     world_marker_cfg.markers["frame"].scale = (0.30, 0.30, 0.30)

    #     env._world_frame_marker = VisualizationMarkers(world_marker_cfg)

    # world_pos = torch.tensor(
    #     [[0.0, 0.0, 0.0]],
    #     dtype=torch.float32,
    #     device=env.device,
    # )

    # world_quat = torch.tensor(
    #     [[1.0, 0.0, 0.0, 0.0]],  # w, x, y, z
    #     dtype=torch.float32,
    #     device=env.device,
    # )

    # env._world_frame_marker.visualize(
    #     translations=world_pos,
    #     orientations=world_quat,
    # )

    # ==== DEBUG : Visualize EEF frame =====

    # if not hasattr(env, "_test_eef_frame_marker"):
    #     marker_cfg = FRAME_MARKER_CFG.copy()
    #     marker_cfg.prim_path = "/Visuals/TestEEFFrame"
    #     marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)

    #     env._test_eef_frame_marker = VisualizationMarkers(marker_cfg)

    # env._test_eef_frame_marker.visualize(
    #     translations=ee_frame.data.target_pos_w[..., 0, :],
    #     orientations=ee_frame.data.target_quat_w[..., 0, :],
    # )

    # ===== DEBUG : Print EEF / Target pose =====

    # eef_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    # eef_quat_w = ee_frame.data.target_quat_w[..., 0, :]

    # print("\n========== DEBUG POSE ==========")

    # print("\n[World frame 기준 EEF Pose]")
    # print("EEF position (world):")
    # print(eef_pos_w)
    # print("EEF quaternion (world):")
    # print(eef_quat_w)

    # print("\n[World frame 기준 Target Pose]")
    # print("Target position (world):")
    # print(target_pos_w)

    # print("\n[EEF frame 기준 Target Pose]")
    # print("Target position (EEF):")
    # print(target_pos_eef)

    # print("================================\n")

    return target_pos_eef


def MA_object_width(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return env.target_width


def MA_target_goal_command(
    env: ManagerBasedRLEnv,
    command_name: str = MISSING,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    robot: RigidObject = env.scene[robot_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3]
    des_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_w
    )

    # print(f"goal: {des_pos_b}")
    return des_pos_b


def MA_target_goal_command_in_EEF(
    env: ManagerBasedRLEnv,
    command_name: str = MISSING,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the commanded target position expressed in the end-effector frame."""

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :3]

    goal_pos_eef, _ = subtract_frame_transforms(
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :],
        goal_pos_w,
    )
    return goal_pos_eef


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )

    return object_pos_b


def target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )

    return torch.concat((object_pos_b, object_quat_b), dim=1)


def object_init_pos_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_init_pos_w = object.data.default_root_state[:, :3]

    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        object_init_pos_w,
    )
    # print(object_pos_b)
    return object_pos_b


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_pos_r(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos_w = ee_tf_data.target_pos_w[..., 0, :]
    ee_quat_w = ee_tf_data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        ee_pos_w,
        ee_quat_w,
    )

    ee_quat_b = quat_unique(ee_quat_b) if make_quat_unique else ee_quat_b

    return torch.cat((ee_pos_b, ee_quat_b), dim=-1)


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]

    # print(f"ee_quat:{ee_quat}")
    # make first element of quaternion positive
    return quat_unique(ee_quat) if make_quat_unique else ee_quat


def ee_quat_r(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    ee_pos_r, ee_quat_r = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        ee_tf_data.target_pos_w[..., 0, :],
        ee_quat,
    )

    return quat_unique(ee_quat_r) if make_quat_unique else ee_quat_r


def target_goal_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    robot: RigidObject = env.scene[robot_cfg.name]

    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3]
    des_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_w
    )

    # print(f"des_pos_w:{des_pos_w}")
    # print(f"des_pos_r: {des_pos_r}")
    return des_pos_b


def ee_goal_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    robot: Articulation = env.scene[robot_cfg.name]

    # command = env.command_manager.get_command(command_name)
    # des_pos_w = command[:, :3]
    # des_pos_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_w
    # )

    des_joint_pos = robot.data.default_joint_pos[:, :6]

    # print(f"des_pos_w:{des_pos_w}")
    # print(f"des_pos_r: {des_pos_b}")
    return des_joint_pos


def rl_joint_pos_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, :8] - asset.data.default_joint_pos[:, :8]


def rl_joint_vel_rel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, :6] - asset.data.default_joint_vel[:, :6]


EEF_OFFSET = (0.13, 0.0, 0.0)


def _eef_pose_w(
    robot: Articulation,
    eef_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_id = eef_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id]
    body_quat_w = robot.data.body_quat_w[:, body_id]
    offset = torch.tensor(
        EEF_OFFSET,
        dtype=torch.float32,
        device=robot.device,
    ).repeat(robot.num_instances, 1)
    eef_pos_w = body_pos_w + math_utils.quat_apply(body_quat_w, offset)
    return eef_pos_w, body_quat_w


def target_position_in_eef_frame(
    env: ManagerBasedRLEnv,
    eef_cfg: SceneEntityCfg,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Selected target position expressed in the current EEF frame."""
    robot: Articulation = env.scene[eef_cfg.name]
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_id.squeeze(-1).long()
    target_pos_w = objects.data.object_pos_w[rows, target_ids]

    # "World" frame에 대한 EEF frame의 상대 위치를 계산합니다.
    eef_pos_w, eef_quat_w = _eef_pose_w(robot, eef_cfg)
    target_pos_eef, _ = math_utils.subtract_frame_transforms(
        eef_pos_w,
        eef_quat_w,
        target_pos_w,
    )
    return target_pos_eef


def compare_target_position_in_eef(
    env: ManagerBasedRLEnv,
    eef_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Compare two methods for expressing target-object position in EEF frame."""

    # ============================================================
    # Common data
    # ============================================================
    robot: Articulation = env.scene[eef_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]

    env_indices = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_id.squeeze(-1).long()

    # ============================================================
    # Target position in World
    # ============================================================

    # Method 1에서 사용한 방식
    target_pos_w_1 = objects.data.object_pos_w[env_indices, target_ids]

    # Method 2에서 사용한 방식
    target_pos_w_2 = objects.data.object_state_w[env_indices, target_ids, :3]

    # ============================================================
    # Method 1:
    # Articulation + _eef_pose_w()
    # ============================================================

    eef_pos_w_1, eef_quat_w_1 = _eef_pose_w(robot, eef_cfg)

    target_pos_eef_1, _ = math_utils.subtract_frame_transforms(
        eef_pos_w_1,
        eef_quat_w_1,
        target_pos_w_1,
    )

    # ============================================================
    # Method 2:
    # FrameTransformer
    # ============================================================

    eef_pos_w_2 = ee_frame.data.target_pos_w[..., 0, :]
    eef_quat_w_2 = ee_frame.data.target_quat_w[..., 0, :]

    target_pos_eef_2, _ = subtract_frame_transforms(
        eef_pos_w_2,
        eef_quat_w_2,
        target_pos_w_2,
    )

    # ============================================================
    # Debug output - ENV 0
    # ============================================================

    print("\n")
    print("============================================================")
    print(" EEF TARGET POSITION COMPARISON")
    print("============================================================")

    print("\n[1] Target position in World")
    print("Method 1 - object_pos_w      :", target_pos_w_1[0])
    print("Method 2 - object_state_w    :", target_pos_w_2[0])
    print("Difference                   :", target_pos_w_1[0] - target_pos_w_2[0])
    print(
        "Difference norm              :",
        torch.norm(target_pos_w_1[0] - target_pos_w_2[0]),
    )

    print("\n[2] EEF position in World")
    print("Method 1 - _eef_pose_w       :", eef_pos_w_1[0])
    print("Method 2 - FrameTransformer  :", eef_pos_w_2[0])
    print("Difference                   :", eef_pos_w_1[0] - eef_pos_w_2[0])
    print("Difference norm              :", torch.norm(eef_pos_w_1[0] - eef_pos_w_2[0]))

    print("\n[3] EEF quaternion in World")
    print("Method 1 - _eef_pose_w       :", eef_quat_w_1[0])
    print("Method 2 - FrameTransformer  :", eef_quat_w_2[0])
    print("Raw difference               :", eef_quat_w_1[0] - eef_quat_w_2[0])

    print("\n[4] Target position in EEF")
    print("Method 1                     :", target_pos_eef_1[0])
    print("Method 2                     :", target_pos_eef_2[0])
    print("Difference                   :", target_pos_eef_1[0] - target_pos_eef_2[0])
    print(
        "Difference norm              :",
        torch.norm(target_pos_eef_1[0] - target_pos_eef_2[0]),
    )

    print("\n[5] FrameTransformer")
    print("Target frame names           :", ee_frame.data.target_frame_names)

    print("============================================================\n")

    # 기존 함수와 동일하게 하나를 return
    # 여기서는 Method 2(FrameTransformer)의 결과를 반환
    return target_pos_eef_2
