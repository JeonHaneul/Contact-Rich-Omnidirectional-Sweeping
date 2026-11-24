from __future__ import annotations
from dataclasses import MISSING
import math

import torch
import numpy as np
from random import shuffle, choice
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.utils.math import subtract_frame_transforms, random_yaw_orientation
from isaaclab.sensors import FrameTransformerData

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_array: tuple,
    asset_dict: dict = MISSING,
    object_width_dict: dict = MISSING,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
    ceiling_height: int = MISSING,
    sweep_dir_options: bool = False,
    task_mode: str = MISSING,
    target_only: bool = False,
) -> None:
    
    rows, cols = len(pose_array[0]), len(pose_array[0][0])
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]


    if task_mode == "grasping":
        target_object_name = choice(["cup_1", "bottle_1", "can_1"])
        
    elif task_mode == "sweeping_right":
        target_object_name = choice(list(asset_dict.keys()))


    target_object_id = object_collection.find_objects(name_keys=target_object_name)
    target_width = torch.tensor(object_width_dict[target_object_name], dtype=torch.float32, device=env.device)
    env.target_id[env_ids, 0] = target_object_id[0].to(env.target_id.dtype)
    env.target_width[env_ids, 0] = target_width
    
    
    asset_keys_list: list = list(asset_dict.keys())

    pose_array_tensor = torch.tensor(pose_array, device=env.device)

    # Orientation randomization 미적용
    orientations = torch.empty((env_ids.shape[0], 4), device=env.device)
    orientations[:, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    velocities = torch.zeros((env_ids.shape[0],6), device=env.device)


    shuffle(asset_keys_list)
    # asset_keys_list.append(asset_keys_list.pop(asset_keys_list.index(target_object_name))) 
    # asset_keys_list.insert(0, asset_keys_list.pop(asset_keys_list.index(target_object_name)))

    for index, asset_name in enumerate(asset_keys_list):
        if asset_name == target_object_name:
            target_index = index

        pose_instance = pose_array_tensor[0, index // cols, index % cols]
        positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
        if target_only and asset_name != target_object_name:
            # reset_scene_to_default restores every configured object before this
            # event runs. Keep non-target objects outside the active shelf area.
            positions[:, 2] = ceiling_height
        else:
            position_uncertainty = torch.empty(2, device=env.device).uniform_(
                -0.02, 0.02
            )
            positions[:, :2] = positions[:, :2] + position_uncertainty
        orientations[:, :] = random_yaw_orientation(num=len(env_ids), device=env.device)
        object_ids = object_collection.find_objects(name_keys=asset_name)

        object_collection.write_object_link_state_to_sim(
            torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
            env_ids=env_ids,
            object_ids=object_ids[0]
        )

    if target_only:
        # Parked objects must stay parked: otherwise gravity can bring them back
        # into the task and trigger inherited collision/termination terms.
        all_object_ids = torch.arange(
            object_collection.num_objects, device=env.device
        )
        view_ids = (
            all_object_ids.unsqueeze(1) * object_collection.num_instances
            + env_ids.unsqueeze(0)
        ).flatten()
        view_ids_cpu = view_ids.cpu()
        gravity_disabled = (
            object_collection.root_physx_view.get_disable_gravities().clone()
        )
        selected_gravity_disabled = (
            (all_object_ids != target_object_id[0])
            .unsqueeze(1)
            .expand(-1, len(env_ids))
            .reshape(-1, 1)
            .to(device=gravity_disabled.device, dtype=gravity_disabled.dtype)
        )
        gravity_disabled[view_ids_cpu] = selected_gravity_disabled
        object_collection.root_physx_view.set_disable_gravities(
            gravity_disabled, indices=view_ids_cpu
        )
    
    env.sweep_dir[env_ids, 1] = 0.18  # 초기화
    direction = 1.0
    if sweep_dir_options:
        direction = choice([-1.0, 1.0])  # -1.0 for left, 1.0 for right
        env.sweep_dir[env_ids, 1] = env.sweep_dir[env_ids, 1] * direction
    

    if task_mode == "sweeping_right" and not target_only:
        if direction > 0:
            adjacent_indices = sweeping_right_mode(target_index, rows, cols, env.device)
        else:
            adjacent_indices = sweeping_left_mode(target_index, rows, cols, env.device)

        for adjacent in adjacent_indices:
            object_index = adjacent[0] * cols + adjacent[1]
            object_name = asset_keys_list[object_index]

            pose_instance = pose_array_tensor[0, adjacent[0], adjacent[1]]
            positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
            positions[:, 2] = ceiling_height  # 높이 변경

            object_ids = object_collection.find_objects(name_keys=object_name)
            object_collection.write_object_link_state_to_sim(
                torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0]
            )

    if task_mode == "grasping" and not target_only:

        adjacent_indices = grasping_mode(target_index, rows, cols, env.device)

        for adjacent in adjacent_indices:
            object_index = adjacent[0] * cols + adjacent[1]
            object_name = asset_keys_list[object_index]

            pose_instance = pose_array_tensor[0, adjacent[0], adjacent[1]]
            positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
            positions[:, 2] = ceiling_height  # 높이 변경

            object_ids = object_collection.find_objects(name_keys=object_name)
            object_collection.write_object_link_pose_to_sim(
                torch.cat((positions, orientations), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0]
            )


def randomize_single_object_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_array: tuple,
    asset_dict: dict = MISSING,
    object_width_dict: dict = MISSING,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
    ceiling_height: int = MISSING,
    sweep_dir_options: bool = False,
    task_mode: str = MISSING,
) -> None:
    """Randomize a sweep scene with only the selected target on the shelf."""

    randomize_scene(
        env=env,
        env_ids=env_ids,
        pose_array=pose_array,
        asset_dict=asset_dict,
        object_width_dict=object_width_dict,
        object_collection_cfg=object_collection_cfg,
        ceiling_height=ceiling_height,
        sweep_dir_options=sweep_dir_options,
        task_mode=task_mode,
        target_only=True,
    )


def randomize_single_object_in_work_area(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_array: tuple,
    asset_dict: dict = MISSING,
    object_width_dict: dict = MISSING,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
    ceiling_height: float = MISSING,
    sweep_dir_options: bool = False,
    task_mode: str = MISSING,
    legacy_position_noise: float = 0.02,
) -> None:
    """Continuously place one target inside the legacy grid bounds.

    The configured grid is used only to recover the original work-area bounds
    and to give parked objects separated x/y slots.  Target identity, target
    position, yaw, and sweep direction are sampled independently per
    environment.  Every non-target object is parked above the active shelf and
    has gravity disabled until it becomes a target in a later reset.
    """

    if task_mode != "sweeping_right":
        raise ValueError(
            "Continuous single-object placement supports only "
            f"task_mode='sweeping_right', got {task_mode!r}."
        )
    if legacy_position_noise < 0.0:
        raise ValueError("legacy_position_noise must not be negative.")

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    object_collection: RigidObjectCollection = env.scene[
        object_collection_cfg.name
    ]
    asset_names = tuple(asset_dict.keys())
    if len(asset_names) != object_collection.num_objects:
        raise ValueError(
            "asset_dict must describe every object in object_collection: "
            f"{len(asset_names)} != {object_collection.num_objects}."
        )

    grid_poses = torch.as_tensor(
        pose_array, dtype=torch.float32, device=env.device
    ).reshape(-1, 7)
    if len(grid_poses) != len(asset_names):
        raise ValueError(
            "pose_array must contain one legacy grid pose per object: "
            f"{len(grid_poses)} != {len(asset_names)}."
        )
    if not torch.allclose(
        grid_poses[:, 2], grid_poses[:1, 2].expand_as(grid_poses[:, 2])
    ):
        raise ValueError("All legacy work-area poses must use one shelf height.")

    object_ids = torch.stack(
        [
            object_collection.find_objects(name_keys=asset_name)[0][0]
            for asset_name in asset_names
        ]
    ).to(device=env.device, dtype=torch.long)
    if torch.unique(object_ids).numel() != object_collection.num_objects:
        raise ValueError("asset_dict names must resolve to distinct objects.")

    # Start from a complete zero-velocity state.  The old grid x/y positions
    # become separated parking slots only; the active target does not use a
    # grid pose.
    states = object_collection.data.default_object_state[env_ids].clone()
    states[..., 7:] = 0.0
    env_origins = env.scene.env_origins[env_ids]
    states[:, object_ids, :3] = (
        grid_poses[:, :3].unsqueeze(0) + env_origins.unsqueeze(1)
    )
    states[:, object_ids, 2] = ceiling_height + env_origins[:, 2:3]
    states[..., 3:7] = random_yaw_orientation(
        num=num_resets * object_collection.num_objects,
        device=env.device,
    ).reshape(num_resets, object_collection.num_objects, 4)

    target_asset_indices = torch.randint(
        low=0,
        high=len(asset_names),
        size=(num_resets,),
        device=env.device,
    )
    target_ids = object_ids[target_asset_indices]
    reset_rows = torch.arange(num_resets, device=env.device)

    # Legacy randomize_scene adds +/-2 cm XY uncertainty to every grid slot.
    # Include that envelope so the continuous sampler has the same final area.
    work_area_min = torch.amin(grid_poses[:, :2], dim=0) - legacy_position_noise
    work_area_max = torch.amax(grid_poses[:, :2], dim=0) + legacy_position_noise
    target_xy = work_area_min + torch.rand(
        (num_resets, 2), device=env.device
    ) * (work_area_max - work_area_min)
    states[reset_rows, target_ids, :2] = target_xy + env_origins[:, :2]
    states[reset_rows, target_ids, 2] = grid_poses[0, 2] + env_origins[:, 2]

    widths = torch.as_tensor(
        [object_width_dict[name] for name in asset_names],
        dtype=torch.float32,
        device=env.device,
    )
    env.target_id[env_ids, 0] = target_ids.to(env.target_id.dtype)
    env.target_width[env_ids, 0] = widths[target_asset_indices]

    env.sweep_dir[env_ids] = 0.0
    if sweep_dir_options:
        direction = torch.where(
            torch.rand(num_resets, device=env.device) < 0.5,
            -torch.ones(num_resets, device=env.device),
            torch.ones(num_resets, device=env.device),
        )
    else:
        direction = torch.ones(num_resets, device=env.device)
    env.sweep_dir[env_ids, 1] = 0.18 * direction

    object_collection.write_object_link_state_to_sim(states, env_ids=env_ids)

    # The collection's PhysX view is object-major.  Update gravity for every
    # reset/object pair because each environment may select a different target.
    all_object_ids = torch.arange(
        object_collection.num_objects, device=env.device
    )
    view_ids = (
        all_object_ids.unsqueeze(1) * object_collection.num_instances
        + env_ids.unsqueeze(0)
    ).flatten()
    gravity_disabled = (
        object_collection.root_physx_view.get_disable_gravities().clone()
    )
    parked = (
        all_object_ids.unsqueeze(1) != target_ids.unsqueeze(0)
    ).reshape(-1, 1)
    view_ids_cpu = view_ids.cpu()
    gravity_disabled[view_ids_cpu] = parked.to(
        device=gravity_disabled.device,
        dtype=gravity_disabled.dtype,
    )
    object_collection.root_physx_view.set_disable_gravities(
        gravity_disabled, indices=view_ids_cpu
    )


def _sample_heading_in_cone(
    reference_heading: torch.Tensor,
    max_angle: float,
) -> torch.Tensor:
    """Sample unit headings strictly inside a cone around each reference."""

    if not 0.0 < max_angle < math.pi:
        raise ValueError("max_angle must be between zero and pi radians.")
    reference_heading = torch.nn.functional.normalize(
        reference_heading, dim=-1
    )
    helper_z = torch.zeros_like(reference_heading)
    helper_z[:, 2] = 1.0
    helper_y = torch.zeros_like(reference_heading)
    helper_y[:, 1] = 1.0
    helper = torch.where(
        (torch.abs(reference_heading[:, 2]) < 0.9).unsqueeze(-1),
        helper_z,
        helper_y,
    )
    tangent_u = torch.nn.functional.normalize(
        torch.linalg.cross(reference_heading, helper, dim=-1), dim=-1
    )
    tangent_v = torch.linalg.cross(reference_heading, tangent_u, dim=-1)

    angle = torch.rand(
        (len(reference_heading), 1), device=reference_heading.device
    ) * max_angle
    azimuth = 2.0 * math.pi * torch.rand_like(angle)
    tangent = torch.cos(azimuth) * tangent_u + torch.sin(azimuth) * tangent_v
    return torch.nn.functional.normalize(
        torch.cos(angle) * reference_heading + torch.sin(angle) * tangent,
        dim=-1,
    )


class initialize_robot_in_random_shelf_pose(ManagerTermBase):
    """Randomize the TCP pose while keeping the base and wrist-3 fixed.

    Position and tool-heading direction are solved with the first five UR5e
    joints.  Wrist-3 is excluded from the IK and restored to the inherited
    environment's default joint value after every update.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["robot_cfg"].name]
        if not self.robot.is_fixed_base:
            raise ValueError(
                "Random shelf-pose initialization requires a fixed-base robot."
            )
        self.object_collection: RigidObjectCollection = env.scene[
            params["object_collection_cfg"].name
        ]
        self.arm_joint_ids, arm_joint_names = self.robot.find_joints(
            params["arm_joint_names"], preserve_order=True
        )
        locked_joint_name = params["locked_joint_name"]
        if locked_joint_name not in arm_joint_names:
            raise ValueError(
                f"Locked joint {locked_joint_name!r} is not an arm joint."
            )
        self.locked_joint_id = self.arm_joint_ids[
            arm_joint_names.index(locked_joint_name)
        ]
        self.solve_joint_ids = [
            joint_id
            for joint_id in self.arm_joint_ids
            if joint_id != self.locked_joint_id
        ]

        body_ids, body_names = self.robot.find_bodies(params["eef_body_name"])
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one EEF body named {params['eef_body_name']!r}; "
                f"found {body_names}."
            )
        self.eef_body_id = int(body_ids[0])
        self.jacobian_body_id = self.eef_body_id - 1
        self.jacobian_joint_ids = self.solve_joint_ids

        if not hasattr(env, "random_init_target_eef_pos_w"):
            env.random_init_target_eef_pos_w = torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            )
        if not hasattr(env, "random_init_target_heading_w"):
            env.random_init_target_heading_w = torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            )
        if not hasattr(env, "random_init_ik_success"):
            env.random_init_ik_success = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
        if not hasattr(env, "random_init_pose_attempt_count"):
            env.random_init_pose_attempt_count = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )
        if not hasattr(env, "random_init_used_pose_bank"):
            env.random_init_used_pose_bank = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

        # Successful randomized poses are retained by workspace cell.  The
        # bank is never seeded with one hand-written posture: every entry was
        # produced by the same full reset distribution and passed all hard
        # constraints.  It is used only after bounded online resampling.
        self._pose_bank: dict[int, torch.Tensor] = {}

    def _sample_desired_eef_positions(
        self,
        *,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_rows: torch.Tensor,
        desired_eef_pos: torch.Tensor,
        base_pos_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        x_delta: torch.Tensor,
        available_x: torch.Tensor,
        env_origins: torch.Tensor,
        base_x_clearance: float,
        eef_y_range: tuple[float, float],
        eef_z_range: tuple[float, float],
    ) -> None:
        """Sample the inherited base-to-target collision-safe workspace."""

        del env, env_ids, target_pos_w
        sampled_base_distance = base_x_clearance + torch.rand(
            len(pose_rows), device=desired_eef_pos.device
        ) * available_x[pose_rows]
        desired_eef_pos[pose_rows, 0] = base_pos_w[pose_rows, 0] + torch.sign(
            x_delta[pose_rows]
        ) * sampled_base_distance
        desired_eef_pos[pose_rows, 1] = (
            env_origins[pose_rows, 1]
            + eef_y_range[0]
            + torch.rand(len(pose_rows), device=desired_eef_pos.device)
            * (eef_y_range[1] - eef_y_range[0])
        )
        desired_eef_pos[pose_rows, 2] = (
            env_origins[pose_rows, 2]
            + eef_z_range[0]
            + torch.rand(len(pose_rows), device=desired_eef_pos.device)
            * (eef_z_range[1] - eef_z_range[0])
        )

    def _is_object_clearance_safe(
        self,
        *,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        rows: torch.Tensor,
        current_eef_pos: torch.Tensor,
        target_pos_w: torch.Tensor,
        base_distance: torch.Tensor,
        x_distance: torch.Tensor,
        x_delta: torch.Tensor,
        object_x_clearance: float,
    ) -> torch.Tensor:
        """Accept positions that remain in front of the target object."""

        del env, env_ids, current_eef_pos, target_pos_w, x_delta
        return base_distance <= x_distance[rows] - object_x_clearance

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg,
        object_collection_cfg: SceneEntityCfg,
        arm_joint_names: tuple[str, ...],
        locked_joint_name: str,
        eef_body_name: str,
        eef_offset: tuple[float, float, float],
        tool_heading_axis: tuple[float, float, float],
        base_x_clearance: float,
        object_x_clearance: float,
        eef_y_range: tuple[float, float],
        eef_z_range: tuple[float, float],
        max_heading_angle: float,
        heading_limit: float,
        max_iterations: int,
        damping: float,
        step_size: float,
        position_tolerance: float,
        heading_tolerance: float,
        acceptance_position_tolerance: float,
        acceptance_heading_tolerance: float,
        max_pose_attempts: int,
        pose_bank_fallback_attempts: int,
        pose_bank_capacity_per_cell: int,
        joint_seed_offsets: tuple[tuple[float, ...], ...],
    ) -> None:
        del (
            robot_cfg,
            object_collection_cfg,
            arm_joint_names,
            locked_joint_name,
            eef_body_name,
        )
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        num_resets = len(env_ids)
        if num_resets == 0:
            return
        if base_x_clearance <= 0.0 or object_x_clearance <= 0.0:
            raise ValueError("EEF x clearances must be positive.")
        if eef_y_range[0] >= eef_y_range[1] or eef_z_range[0] >= eef_z_range[1]:
            raise ValueError("EEF y/z ranges must have increasing bounds.")
        if max_heading_angle + acceptance_heading_tolerance > heading_limit:
            raise ValueError(
                "The sampled heading cone plus IK acceptance tolerance must "
                "not exceed heading_limit."
            )
        if max_pose_attempts < 1:
            raise ValueError("max_pose_attempts must be at least one.")
        if pose_bank_fallback_attempts < 0:
            raise ValueError("pose_bank_fallback_attempts must not be negative.")
        if pose_bank_capacity_per_cell < 1:
            raise ValueError("pose_bank_capacity_per_cell must be positive.")

        target_ids = env.target_id[env_ids, 0].long()
        target_pos_w = self.object_collection.data.object_link_state_w[
            env_ids, target_ids, :3
        ]
        base_pos_w = self.robot.data.root_pos_w[env_ids]
        x_delta = target_pos_w[:, 0] - base_pos_w[:, 0]
        x_distance = torch.abs(x_delta)
        available_x = x_distance - base_x_clearance - object_x_clearance
        if torch.any(available_x <= 0.0):
            failed_ids = env_ids[available_x <= 0.0].tolist()
            raise RuntimeError(
                "No collision-safe EEF x interval exists between base and "
                f"target for environments {failed_ids[:8]}."
            )

        desired_eef_pos = torch.zeros(
            (num_resets, 3), dtype=torch.float32, device=env.device
        )
        env_origins = env.scene.env_origins[env_ids]

        heading_axis = torch.as_tensor(
            tool_heading_axis, dtype=torch.float32, device=env.device
        )
        if torch.linalg.norm(heading_axis) <= 1.0e-6:
            raise ValueError("tool_heading_axis must be non-zero.")
        heading_axis = torch.nn.functional.normalize(heading_axis, dim=0)
        heading_axis_batch = heading_axis.unsqueeze(0).expand(num_resets, -1)
        reference_quat = self.robot.data.body_quat_w[
            env_ids, self.eef_body_id
        ]
        reference_heading = math_utils.quat_apply(
            reference_quat, heading_axis_batch
        )
        desired_heading = torch.zeros_like(reference_heading)

        if len(joint_seed_offsets) == 0:
            raise ValueError("joint_seed_offsets must contain at least one seed.")
        seed_offsets = torch.as_tensor(
            joint_seed_offsets, dtype=torch.float32, device=env.device
        )
        if seed_offsets.ndim != 2 or seed_offsets.shape[1] != len(
            self.arm_joint_ids
        ):
            raise ValueError(
                "Each seed must contain one offset per arm joint: "
                f"expected {len(self.arm_joint_ids)}, got {tuple(seed_offsets.shape)}."
            )

        offset = torch.as_tensor(
            eef_offset, dtype=torch.float32, device=env.device
        ).unsqueeze(0).expand(num_resets, -1)
        success = torch.zeros(num_resets, dtype=torch.bool, device=env.device)
        used_pose_bank = torch.zeros_like(success)
        pose_attempt_count = torch.zeros(
            num_resets, dtype=torch.long, device=env.device
        )
        final_position_error = torch.full(
            (num_resets,), torch.inf, dtype=torch.float32, device=env.device
        )
        final_heading_error = torch.full_like(final_position_error, torch.inf)
        identity = torch.eye(
            len(self.solve_joint_ids), device=env.device
        ).unsqueeze(0)

        def evaluate(rows: torch.Tensor) -> dict[str, torch.Tensor]:
            """Evaluate target accuracy and every hard reset constraint."""

            row_env_ids = env_ids[rows]
            body_pos = self.robot.data.body_pos_w[row_env_ids, self.eef_body_id]
            body_quat = self.robot.data.body_quat_w[row_env_ids, self.eef_body_id]
            current_eef_pos = body_pos + math_utils.quat_apply(
                body_quat, offset[rows]
            )
            current_heading = math_utils.quat_apply(
                body_quat, heading_axis_batch[rows]
            )
            position_error = torch.linalg.norm(
                desired_eef_pos[rows] - current_eef_pos, dim=-1
            )
            heading_error = torch.acos(
                torch.clamp(
                    torch.sum(current_heading * desired_heading[rows], dim=-1),
                    -1.0,
                    1.0,
                )
            )
            reference_heading_error = torch.acos(
                torch.clamp(
                    torch.sum(current_heading * reference_heading[rows], dim=-1),
                    -1.0,
                    1.0,
                )
            )
            local_pos = current_eef_pos - env_origins[rows]
            base_distance = (
                current_eef_pos[:, 0] - base_pos_w[rows, 0]
            ) * torch.sign(x_delta[rows])
            safe_base_x = base_distance >= base_x_clearance
            safe_object_x = self._is_object_clearance_safe(
                env=env,
                env_ids=env_ids,
                rows=rows,
                current_eef_pos=current_eef_pos,
                target_pos_w=target_pos_w,
                base_distance=base_distance,
                x_distance=x_distance,
                x_delta=x_delta,
                object_x_clearance=object_x_clearance,
            )
            safe_y = (local_pos[:, 1] >= eef_y_range[0]) & (
                local_pos[:, 1] <= eef_y_range[1]
            )
            safe_z = (local_pos[:, 2] >= eef_z_range[0]) & (
                local_pos[:, 2] <= eef_z_range[1]
            )
            wrist_error = torch.abs(
                self.robot.data.joint_pos[row_env_ids, self.locked_joint_id]
                - self.robot.data.default_joint_pos[
                    row_env_ids, self.locked_joint_id
                ]
            )
            finite = (
                torch.isfinite(position_error)
                & torch.isfinite(heading_error)
                & torch.isfinite(reference_heading_error)
                & torch.all(torch.isfinite(current_eef_pos), dim=-1)
            )
            hard_valid = (
                finite
                & safe_base_x
                & safe_object_x
                & safe_y
                & safe_z
                & (reference_heading_error < heading_limit)
                & (wrist_error < 1.0e-6)
            )
            return {
                "current_eef_pos": current_eef_pos,
                "current_heading": current_heading,
                "local_pos": local_pos,
                "base_distance": base_distance,
                "position_error": position_error,
                "heading_error": heading_error,
                "reference_heading_error": reference_heading_error,
                "finite": finite,
                "safe_base_x": safe_base_x,
                "safe_object_x": safe_object_x,
                "safe_y": safe_y,
                "safe_z": safe_z,
                "safe_wrist": wrist_error < 1.0e-6,
                "hard_valid": hard_valid,
            }

        # A pose attempt samples from the complete requested workspace.  Only
        # failed environments receive a new pose; the object is never moved.
        for _ in range(max_pose_attempts):
            pose_rows = torch.where(~success)[0]
            if len(pose_rows) == 0:
                break
            pose_attempt_count[pose_rows] += 1

            self._sample_desired_eef_positions(
                env=env,
                env_ids=env_ids,
                pose_rows=pose_rows,
                desired_eef_pos=desired_eef_pos,
                base_pos_w=base_pos_w,
                target_pos_w=target_pos_w,
                x_delta=x_delta,
                available_x=available_x,
                env_origins=env_origins,
                base_x_clearance=base_x_clearance,
                eef_y_range=eef_y_range,
                eef_z_range=eef_z_range,
            )
            desired_heading[pose_rows] = _sample_heading_in_cone(
                reference_heading[pose_rows], max_heading_angle
            )
            env.random_init_target_eef_pos_w[env_ids[pose_rows]] = (
                desired_eef_pos[pose_rows]
            )
            env.random_init_target_heading_w[env_ids[pose_rows]] = (
                desired_heading[pose_rows]
            )

            for seed_offset in seed_offsets:
                seed_rows = torch.where(~success)[0]
                if len(seed_rows) == 0:
                    break
                seed_env_ids = env_ids[seed_rows]

                joint_pos = self.robot.data.default_joint_pos[
                    seed_env_ids
                ].clone()
                joint_pos[:, self.arm_joint_ids] += seed_offset.unsqueeze(0)
                joint_pos[:, self.locked_joint_id] = (
                    self.robot.data.default_joint_pos[
                        seed_env_ids, self.locked_joint_id
                    ]
                )
                limits = self.robot.data.soft_joint_pos_limits[
                    seed_env_ids[:, None], self.arm_joint_ids
                ]
                joint_pos[:, self.arm_joint_ids] = torch.clamp(
                    joint_pos[:, self.arm_joint_ids], limits[..., 0], limits[..., 1]
                )
                joint_vel = torch.zeros_like(
                    self.robot.data.joint_vel[seed_env_ids]
                )
                self.robot.write_joint_state_to_sim(
                    joint_pos, joint_vel, env_ids=seed_env_ids
                )

                for _ in range(max_iterations):
                    active_rows = seed_rows[~success[seed_rows]]
                    if len(active_rows) == 0:
                        break
                    active_env_ids = env_ids[active_rows]
                    state = evaluate(active_rows)
                    converged = (
                        state["position_error"] < position_tolerance
                    ) & (
                        state["heading_error"] < heading_tolerance
                    ) & state["hard_valid"]
                    success[active_rows[converged]] = True
                    solve_rows = active_rows[~converged]
                    if len(solve_rows) == 0:
                        break

                    solve_env_ids = env_ids[solve_rows]
                    solve_state = {
                        key: value[~converged] for key, value in state.items()
                    }
                    position_error = (
                        desired_eef_pos[solve_rows]
                        - solve_state["current_eef_pos"]
                    )
                    heading_cross_error = torch.linalg.cross(
                        solve_state["current_heading"],
                        desired_heading[solve_rows],
                        dim=-1,
                    )
                    body_quat = self.robot.data.body_quat_w[
                        solve_env_ids, self.eef_body_id
                    ]
                    world_offset = math_utils.quat_apply(
                        body_quat, offset[solve_rows]
                    )
                    jacobian = self.robot.root_physx_view.get_jacobians()[
                        solve_env_ids, self.jacobian_body_id, :, :
                    ][..., self.jacobian_joint_ids]
                    angular_jacobian = jacobian[:, 3:, :]
                    offset_jacobian = torch.linalg.cross(
                        angular_jacobian.transpose(1, 2),
                        world_offset.unsqueeze(1),
                        dim=-1,
                    ).transpose(1, 2)
                    task_jacobian = torch.cat(
                        (jacobian[:, :3, :] + offset_jacobian, angular_jacobian),
                        dim=1,
                    )
                    error = torch.cat(
                        (position_error, heading_cross_error), dim=-1
                    )
                    jacobian_transpose = task_jacobian.transpose(1, 2)
                    delta_joint = torch.linalg.solve(
                        jacobian_transpose @ task_jacobian
                        + damping**2 * identity,
                        jacobian_transpose @ error.unsqueeze(-1),
                    ).squeeze(-1)
                    delta_joint = torch.clamp(
                        step_size * delta_joint, -0.2, 0.2
                    )

                    joint_pos = self.robot.data.joint_pos[solve_env_ids].clone()
                    joint_pos[:, self.solve_joint_ids] += delta_joint
                    joint_pos[:, self.locked_joint_id] = (
                        self.robot.data.default_joint_pos[
                            solve_env_ids, self.locked_joint_id
                        ]
                    )
                    limits = self.robot.data.soft_joint_pos_limits[
                        solve_env_ids[:, None], self.arm_joint_ids
                    ]
                    joint_pos[:, self.arm_joint_ids] = torch.clamp(
                        joint_pos[:, self.arm_joint_ids],
                        limits[..., 0],
                        limits[..., 1],
                    )
                    joint_vel = torch.zeros_like(
                        self.robot.data.joint_vel[solve_env_ids]
                    )
                    self.robot.write_joint_state_to_sim(
                        joint_pos, joint_vel, env_ids=solve_env_ids
                    )

                # Re-evaluate after the final simulator write.  An accurate
                # target match outside a hard boundary is never convergence.
                seed_state = evaluate(seed_rows)
                seed_success = (
                    seed_state["hard_valid"]
                    & (
                        seed_state["position_error"]
                        < acceptance_position_tolerance
                    )
                    & (
                        seed_state["heading_error"]
                        < acceptance_heading_tolerance
                    )
                )
                success[seed_rows] = seed_success
                final_position_error[seed_rows] = seed_state["position_error"]
                final_heading_error[seed_rows] = seed_state["heading_error"]

        # Populate a stratified cache from genuine randomized successes before
        # handling rare environments that exhausted online pose resampling.
        successful_rows = torch.where(success)[0]
        if len(successful_rows) > 0:
            successful_state = evaluate(successful_rows)
            x_fraction = torch.clamp(
                (
                    successful_state["base_distance"] - base_x_clearance
                ) / available_x[successful_rows],
                0.0,
                1.0,
            )
            y_fraction = torch.clamp(
                (
                    successful_state["local_pos"][:, 1] - eef_y_range[0]
                ) / (eef_y_range[1] - eef_y_range[0]),
                0.0,
                1.0,
            )
            z_fraction = torch.clamp(
                (
                    successful_state["local_pos"][:, 2] - eef_z_range[0]
                ) / (eef_z_range[1] - eef_z_range[0]),
                0.0,
                1.0,
            )
            heading_fraction = torch.clamp(
                successful_state["reference_heading_error"] / heading_limit,
                0.0,
                1.0,
            )
            cell_ids = (
                (x_fraction >= 0.5).long()
                + 2 * (y_fraction >= 0.5).long()
                + 4 * (z_fraction >= 0.5).long()
                + 8 * (heading_fraction >= 0.5).long()
            )
            successful_joints = self.robot.data.joint_pos[
                env_ids[successful_rows]
            ][:, self.arm_joint_ids].clone()
            for cell_id in range(16):
                cell_joints = successful_joints[cell_ids == cell_id]
                if len(cell_joints) == 0:
                    continue
                previous = self._pose_bank.get(cell_id)
                if previous is not None:
                    cell_joints = torch.cat((previous, cell_joints), dim=0)
                if len(cell_joints) > pose_bank_capacity_per_cell:
                    keep = torch.randperm(
                        len(cell_joints), device=env.device
                    )[:pose_bank_capacity_per_cell]
                    cell_joints = cell_joints[keep]
                self._pose_bank[cell_id] = cell_joints

        # No fixed home pose is used.  Cache fallback samples workspace cells
        # uniformly, then validates the borrowed joint state against this
        # environment's current object and hard constraints.
        occupied_cells = tuple(self._pose_bank)
        for _ in range(pose_bank_fallback_attempts):
            bank_rows = torch.where(~success)[0]
            if len(bank_rows) == 0 or len(occupied_cells) == 0:
                break
            bank_env_ids = env_ids[bank_rows]
            selected_cells = torch.randint(
                len(occupied_cells), (len(bank_rows),), device=env.device
            )
            bank_joint_pos = self.robot.data.default_joint_pos[
                bank_env_ids
            ].clone()
            for choice_id, cell_id in enumerate(occupied_cells):
                choice_rows = torch.where(selected_cells == choice_id)[0]
                if len(choice_rows) == 0:
                    continue
                candidates = self._pose_bank[cell_id]
                candidate_ids = torch.randint(
                    len(candidates), (len(choice_rows),), device=env.device
                )
                bank_joint_pos[
                    choice_rows[:, None], self.arm_joint_ids
                ] = candidates[candidate_ids]
            bank_joint_pos[:, self.locked_joint_id] = (
                self.robot.data.default_joint_pos[
                    bank_env_ids, self.locked_joint_id
                ]
            )
            bank_joint_vel = torch.zeros_like(
                self.robot.data.joint_vel[bank_env_ids]
            )
            self.robot.write_joint_state_to_sim(
                bank_joint_pos, bank_joint_vel, env_ids=bank_env_ids
            )
            bank_state = evaluate(bank_rows)
            final_position_error[bank_rows] = bank_state["position_error"]
            final_heading_error[bank_rows] = bank_state["heading_error"]
            bank_success = bank_state["hard_valid"]
            accepted_rows = bank_rows[bank_success]
            success[accepted_rows] = True
            used_pose_bank[accepted_rows] = True
            desired_eef_pos[accepted_rows] = bank_state["current_eef_pos"][
                bank_success
            ]
            desired_heading[accepted_rows] = bank_state["current_heading"][
                bank_success
            ]
            final_position_error[accepted_rows] = 0.0
            final_heading_error[accepted_rows] = 0.0
            env.random_init_target_eef_pos_w[env_ids[accepted_rows]] = (
                desired_eef_pos[accepted_rows]
            )
            env.random_init_target_heading_w[env_ids[accepted_rows]] = (
                desired_heading[accepted_rows]
            )

        final_joint_pos = self.robot.data.joint_pos[env_ids].clone()
        final_joint_pos[:, self.locked_joint_id] = self.robot.data.default_joint_pos[
            env_ids, self.locked_joint_id
        ]
        zero_joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])
        self.robot.write_joint_state_to_sim(
            final_joint_pos, zero_joint_vel, env_ids=env_ids
        )
        self.robot.set_joint_position_target(final_joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(zero_joint_vel, env_ids=env_ids)
        env.random_init_ik_success[env_ids] = success
        env.random_init_pose_attempt_count[env_ids] = pose_attempt_count
        env.random_init_used_pose_bank[env_ids] = used_pose_bank

        if not torch.all(success):
            failed_rows = torch.where(~success)[0]
            failed_state = evaluate(failed_rows)
            details = [
                {
                    "env_id": int(env_ids[row].item()),
                    "target_pos_w": target_pos_w[row].tolist(),
                    "desired_eef_pos_w": desired_eef_pos[row].tolist(),
                    "position_error": float(final_position_error[row].item()),
                    "heading_error_rad": float(final_heading_error[row].item()),
                    "failed_constraints": [
                        name
                        for name, passed in (
                            ("finite", failed_state["finite"][index]),
                            (
                                "target_position_accuracy",
                                failed_state["position_error"][index]
                                < acceptance_position_tolerance,
                            ),
                            (
                                "target_heading_accuracy",
                                failed_state["heading_error"][index]
                                < acceptance_heading_tolerance,
                            ),
                            ("base_clearance", failed_state["safe_base_x"][index]),
                            ("object_clearance", failed_state["safe_object_x"][index]),
                            ("eef_y_range", failed_state["safe_y"][index]),
                            ("eef_z_range", failed_state["safe_z"][index]),
                            (
                                "heading_limit",
                                failed_state["reference_heading_error"][index]
                                < heading_limit,
                            ),
                            ("wrist_3", failed_state["safe_wrist"][index]),
                        )
                        if not bool(passed.item())
                    ],
                }
                for index, row in enumerate(failed_rows[:8])
            ]
            raise RuntimeError(
                "Random shelf-pose constrained reset failed for "
                f"{len(failed_rows)}/{num_resets} environments. "
                f"Tried {max_pose_attempts} independently sampled poses, "
                f"{len(seed_offsets)} joint seeds per pose, and "
                f"{pose_bank_fallback_attempts} stratified pose-bank draws. "
                f"First failures: {details}"
            )


class initialize_robot_in_random_shelf_pose_with_target_offset(
    initialize_robot_in_random_shelf_pose
):
    """Extend shelf-pose sampling with a collision-safe pre-push pose.

    The inherited sampler covers the reachable free space from the robot base
    through the shelf interior in front of the object.  This subclass also
    samples beside the target, opposite the commanded sweep direction.  IK and
    all inherited hard constraints still decide whether a sampled pose is
    accepted.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        target_side_probability: float,
        target_side_offset_range: tuple[float, float],
        target_side_clearance: float,
        target_side_x_tolerance: float,
        **kwargs,
    ) -> None:
        if not 0.0 <= target_side_probability <= 1.0:
            raise ValueError("target_side_probability must be in [0, 1].")
        if (
            target_side_offset_range[0] <= 0.0
            or target_side_offset_range[0] >= target_side_offset_range[1]
        ):
            raise ValueError(
                "target_side_offset_range must contain increasing positive bounds."
            )
        if target_side_clearance <= 0.0:
            raise ValueError("target_side_clearance must be positive.")
        if target_side_x_tolerance <= 0.0:
            raise ValueError("target_side_x_tolerance must be positive.")

        self._target_side_probability = target_side_probability
        self._target_side_offset_range = target_side_offset_range
        self._target_side_clearance = target_side_clearance
        self._target_side_x_tolerance = target_side_x_tolerance
        super().__call__(env=env, env_ids=env_ids, **kwargs)

    def _sample_desired_eef_positions(
        self,
        *,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        pose_rows: torch.Tensor,
        desired_eef_pos: torch.Tensor,
        base_pos_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        x_delta: torch.Tensor,
        available_x: torch.Tensor,
        env_origins: torch.Tensor,
        base_x_clearance: float,
        eef_y_range: tuple[float, float],
        eef_z_range: tuple[float, float],
    ) -> None:
        # First sample the full inherited region.  Rows without enough safe
        # lateral shelf room remain valid inherited samples.
        super()._sample_desired_eef_positions(
            env=env,
            env_ids=env_ids,
            pose_rows=pose_rows,
            desired_eef_pos=desired_eef_pos,
            base_pos_w=base_pos_w,
            target_pos_w=target_pos_w,
            x_delta=x_delta,
            available_x=available_x,
            env_origins=env_origins,
            base_x_clearance=base_x_clearance,
            eef_y_range=eef_y_range,
            eef_z_range=eef_z_range,
        )

        row_env_ids = env_ids[pose_rows]
        sweep_sign = torch.sign(env.sweep_dir[row_env_ids, 1])
        target_width = env.target_width[row_env_ids, 0]
        minimum_offset = torch.maximum(
            torch.full_like(target_width, self._target_side_offset_range[0]),
            0.5 * target_width + self._target_side_clearance,
        )
        local_target_y = target_pos_w[pose_rows, 1] - env_origins[pose_rows, 1]
        room_opposite_sweep = torch.where(
            sweep_sign > 0.0,
            local_target_y - eef_y_range[0],
            eef_y_range[1] - local_target_y,
        )
        maximum_offset = torch.minimum(
            torch.full_like(target_width, self._target_side_offset_range[1]),
            room_opposite_sweep,
        )
        choose_side = (
            torch.rand(len(pose_rows), device=env.device)
            < self._target_side_probability
        ) & (sweep_sign != 0.0) & (maximum_offset >= minimum_offset)
        side_rows = pose_rows[choose_side]
        if len(side_rows) == 0:
            return

        side_offset = minimum_offset[choose_side] + torch.rand(
            len(side_rows), device=env.device
        ) * (maximum_offset[choose_side] - minimum_offset[choose_side])
        # Match the target's shelf depth and move laterally away from the
        # commanded push direction.  The clearance validator below permits
        # this x alignment only while the lateral object clearance is valid.
        desired_eef_pos[side_rows, 0] = target_pos_w[side_rows, 0]
        desired_eef_pos[side_rows, 1] = (
            target_pos_w[side_rows, 1]
            - sweep_sign[choose_side] * side_offset
        )

    def _is_object_clearance_safe(
        self,
        *,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        rows: torch.Tensor,
        current_eef_pos: torch.Tensor,
        target_pos_w: torch.Tensor,
        base_distance: torch.Tensor,
        x_distance: torch.Tensor,
        x_delta: torch.Tensor,
        object_x_clearance: float,
    ) -> torch.Tensor:
        safe_in_front = super()._is_object_clearance_safe(
            env=env,
            env_ids=env_ids,
            rows=rows,
            current_eef_pos=current_eef_pos,
            target_pos_w=target_pos_w,
            base_distance=base_distance,
            x_distance=x_distance,
            x_delta=x_delta,
            object_x_clearance=object_x_clearance,
        )
        row_env_ids = env_ids[rows]
        sweep_sign = torch.sign(env.sweep_dir[row_env_ids, 1])
        opposite_distance = (
            target_pos_w[rows, 1] - current_eef_pos[:, 1]
        ) * sweep_sign
        minimum_distance = (
            0.5 * env.target_width[row_env_ids, 0]
            + self._target_side_clearance
        )
        beside_target_x = (
            torch.abs(current_eef_pos[:, 0] - target_pos_w[rows, 0])
            <= self._target_side_x_tolerance
        )
        safe_beside_target = (
            (sweep_sign != 0.0)
            & beside_target_x
            & (opposite_distance >= minimum_distance)
            & (
                opposite_distance
                <= self._target_side_offset_range[1]
                + self._target_side_x_tolerance
            )
        )
        return safe_in_front | safe_beside_target


class initialize_robot_at_reaching_pose(ManagerTermBase):
    """Initialize the arm with its TCP beside the reset-time target object.

    The target position and width are intentionally read from environment
    state only inside this reset event.  They are privileged initialization
    information and do not need to be exposed as policy observations.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["robot_cfg"].name]
        self.shelf: RigidObject = env.scene[params["shelf_cfg"].name]
        self.object_collection: RigidObjectCollection = env.scene[
            params["object_collection_cfg"].name
        ]

        self.arm_joint_ids, _ = self.robot.find_joints(
            params["arm_joint_names"], preserve_order=True
        )
        body_ids, body_names = self.robot.find_bodies(params["eef_body_name"])
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one EEF body named {params['eef_body_name']!r}; "
                f"found {body_names}."
            )
        self.eef_body_id = int(body_ids[0])
        self.jacobian_body_id = (
            self.eef_body_id - 1 if self.robot.is_fixed_base else self.eef_body_id
        )
        self.jacobian_joint_ids = (
            self.arm_joint_ids
            if self.robot.is_fixed_base
            else [joint_id + 6 for joint_id in self.arm_joint_ids]
        )

        # Diagnostics remain environment-internal and are never policy terms.
        if not hasattr(env, "target_init_pos_w"):
            env.target_init_pos_w = torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            )
        if not hasattr(env, "desired_reaching_pose_w"):
            env.desired_reaching_pose_w = torch.zeros(
                (env.num_envs, 7), dtype=torch.float32, device=env.device
            )
        if not hasattr(env, "reaching_ik_success"):
            env.reaching_ik_success = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
        if not hasattr(env, "reaching_ik_attempt_count"):
            env.reaching_ik_attempt_count = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg,
        shelf_cfg: SceneEntityCfg,
        object_collection_cfg: SceneEntityCfg,
        arm_joint_names: tuple[str, ...],
        eef_body_name: str,
        eef_offset: tuple[float, float, float],
        reaching_x_offset: float,
        reaching_z_offset: float,
        position_noise: float | tuple[float, float, float],
        max_iterations: int,
        damping: float,
        step_size: float,
        position_tolerance: float,
        orientation_tolerance: float,
        joint_seed_offsets: tuple[tuple[float, ...], ...],
        eef_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        use_column_axes: bool = False,
    ) -> None:
        del (
            robot_cfg,
            shelf_cfg,
            object_collection_cfg,
            arm_joint_names,
            eef_body_name,
        )
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        target_ids = env.target_id[env_ids, 0].long()
        target_init_pos_w = self.object_collection.data.object_link_state_w[
            env_ids, target_ids, :3
        ].clone()
        env.target_init_pos_w[env_ids] = target_init_pos_w

        # Place the TCP behind the target relative to the sampled sweep
        # direction, matching the offset used by reward_for_hand_reaching.
        direction = torch.sign(env.sweep_dir[env_ids, 1])
        desired_eef_pos = target_init_pos_w.clone()
        desired_eef_pos[:, 0] += reaching_x_offset
        desired_eef_pos[:, 1] -= env.target_width[env_ids, 0] * direction
        desired_eef_pos[:, 2] += reaching_z_offset
        noise = torch.as_tensor(position_noise, device=env.device, dtype=torch.float32)
        if noise.numel() not in (1, 3) or not torch.isfinite(noise).all() or (noise < 0).any():
            raise ValueError("position_noise must be a nonnegative scalar or XYZ tuple.")
        desired_eef_pos += math_utils.sample_uniform(
            -noise,
            noise,
            (len(env_ids), 3),
            device=env.device,
        )

        # Construct the same reward-aligned orientation as the reference
        # event.  The EEF y-axis follows the shelf z-axis while the nominal
        # EEF x-axis is preserved as far as possible.
        shelf_rotation = math_utils.matrix_from_quat(
            self.shelf.data.default_root_state[env_ids, 3:7]
        )
        frame_rotation = torch.tensor(eef_rotation, device=env.device).repeat(len(env_ids), 1)
        nominal_eef_rotation = math_utils.matrix_from_quat(math_utils.quat_mul(
            self.robot.data.body_quat_w[env_ids, self.eef_body_id], frame_rotation,
        ))
        # Keep legacy event callers unchanged; corrected mass tasks opt in.
        desired_y_axis = shelf_rotation[:, :, 2] if use_column_axes else shelf_rotation[:, 2]
        desired_x_axis = nominal_eef_rotation[:, :, 0] if use_column_axes else nominal_eef_rotation[:, 0]
        desired_x_axis = desired_x_axis - torch.sum(
            desired_x_axis * desired_y_axis, dim=-1, keepdim=True
        ) * desired_y_axis
        x_axis_norm = torch.linalg.norm(desired_x_axis, dim=-1, keepdim=True)
        desired_x_axis = torch.where(
            x_axis_norm > 1.0e-6,
            desired_x_axis / torch.clamp_min(x_axis_norm, 1.0e-6),
            shelf_rotation[:, :, 0] if use_column_axes else shelf_rotation[:, 0],
        )
        desired_z_axis = torch.linalg.cross(
            desired_x_axis, desired_y_axis, dim=-1
        )
        desired_z_axis = torch.nn.functional.normalize(desired_z_axis, dim=-1)
        desired_x_axis = torch.linalg.cross(
            desired_y_axis, desired_z_axis, dim=-1
        )
        desired_eef_rotation = torch.stack(
            (desired_x_axis, desired_y_axis, desired_z_axis), dim=2 if use_column_axes else 1
        )
        desired_eef_quat = math_utils.quat_unique(
            math_utils.quat_from_matrix(desired_eef_rotation)
        )
        # IK solves the rigid body, whereas the reward observes its rotated TCP.
        desired_body_quat = math_utils.quat_mul(
            desired_eef_quat, math_utils.quat_inv(frame_rotation),
        )

        offset = torch.tensor(
            eef_offset, dtype=torch.float32, device=env.device
        ).repeat(len(env_ids), 1)
        desired_body_pos = desired_eef_pos - math_utils.quat_apply(
            desired_body_quat, offset
        )
        env.desired_reaching_pose_w[env_ids, :3] = desired_eef_pos
        env.desired_reaching_pose_w[env_ids, 3:7] = desired_eef_quat

        if len(joint_seed_offsets) == 0:
            raise ValueError("joint_seed_offsets must contain at least one seed.")
        seed_offsets = torch.tensor(
            joint_seed_offsets, dtype=torch.float32, device=env.device
        )
        if seed_offsets.ndim != 2 or seed_offsets.shape[1] != len(
            self.arm_joint_ids
        ):
            raise ValueError(
                "Each joint seed offset must contain one value per arm joint: "
                f"expected {len(self.arm_joint_ids)}, "
                f"got shape {tuple(seed_offsets.shape)}."
            )

        # The object pose and desired TCP pose stay fixed across attempts;
        # failed environments retry only from bounded arm joint seeds.
        success = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        attempt_count = torch.zeros(
            len(env_ids), dtype=torch.long, device=env.device
        )
        final_pos_error = torch.full(
            (len(env_ids),),
            torch.inf,
            dtype=torch.float32,
            device=env.device,
        )
        final_rot_error_norm = torch.full_like(final_pos_error, torch.inf)
        zero_joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])
        identity = torch.eye(6, device=env.device).unsqueeze(0)

        for seed_offset in seed_offsets:
            pending_rows = torch.where(~success)[0]
            if len(pending_rows) == 0:
                break
            pending_env_ids = env_ids[pending_rows]
            attempt_count[pending_rows] += 1

            seed_joint_pos = self.robot.data.default_joint_pos[
                pending_env_ids
            ].clone()
            seed_joint_pos[:, self.arm_joint_ids] += seed_offset.unsqueeze(0)
            seed_limits = self.robot.data.soft_joint_pos_limits[
                pending_env_ids[:, None], self.arm_joint_ids
            ]
            seed_joint_pos[:, self.arm_joint_ids] = torch.clamp(
                seed_joint_pos[:, self.arm_joint_ids],
                seed_limits[..., 0],
                seed_limits[..., 1],
            )
            seed_joint_vel = torch.zeros_like(
                self.robot.data.joint_vel[pending_env_ids]
            )
            self.robot.set_joint_position_target(
                seed_joint_pos, env_ids=pending_env_ids
            )
            self.robot.set_joint_velocity_target(
                seed_joint_vel, env_ids=pending_env_ids
            )
            self.robot.write_joint_state_to_sim(
                seed_joint_pos, seed_joint_vel, env_ids=pending_env_ids
            )

            for _ in range(max_iterations):
                active_rows = pending_rows[~success[pending_rows]]
                if len(active_rows) == 0:
                    break
                active_env_ids = env_ids[active_rows]

                current_pos = self.robot.data.body_pos_w[
                    active_env_ids, self.eef_body_id
                ]
                current_quat = self.robot.data.body_quat_w[
                    active_env_ids, self.eef_body_id
                ]
                body_pos_error, rot_error = math_utils.compute_pose_error(
                    current_pos,
                    current_quat,
                    desired_body_pos[active_rows],
                    desired_body_quat[active_rows],
                    rot_error_type="axis_angle",
                )
                current_eef_pos = current_pos + math_utils.quat_apply(
                    current_quat, offset[active_rows]
                )
                tcp_pos_error = (
                    desired_eef_pos[active_rows] - current_eef_pos
                )
                converged = (
                    torch.linalg.norm(tcp_pos_error, dim=-1)
                    < position_tolerance
                ) & (
                    torch.linalg.norm(rot_error, dim=-1)
                    < orientation_tolerance
                )
                success[active_rows[converged]] = True
                solve_rows = active_rows[~converged]
                if len(solve_rows) == 0:
                    break

                solve_env_ids = env_ids[solve_rows]
                error = torch.cat(
                    (
                        body_pos_error[~converged],
                        rot_error[~converged],
                    ),
                    dim=-1,
                )
                jacobian = self.robot.root_physx_view.get_jacobians()[
                    solve_env_ids, self.jacobian_body_id, :, :
                ]
                jacobian = jacobian[..., self.jacobian_joint_ids]
                dls_inverse = torch.linalg.solve(
                    jacobian @ jacobian.transpose(1, 2)
                    + damping**2 * identity,
                    error.unsqueeze(-1),
                )
                delta_joint = (
                    jacobian.transpose(1, 2) @ dls_inverse
                ).squeeze(-1)
                delta_joint = torch.clamp(
                    step_size * delta_joint, -0.2, 0.2
                )

                joint_pos = self.robot.data.joint_pos[solve_env_ids].clone()
                joint_pos[:, self.arm_joint_ids] += delta_joint
                limits = self.robot.data.soft_joint_pos_limits[
                    solve_env_ids[:, None], self.arm_joint_ids
                ]
                joint_pos[:, self.arm_joint_ids] = torch.clamp(
                    joint_pos[:, self.arm_joint_ids],
                    limits[..., 0],
                    limits[..., 1],
                )
                joint_vel = torch.zeros_like(
                    self.robot.data.joint_vel[solve_env_ids]
                )
                self.robot.set_joint_position_target(
                    joint_pos, env_ids=solve_env_ids
                )
                self.robot.set_joint_velocity_target(
                    joint_vel, env_ids=solve_env_ids
                )
                self.robot.write_joint_state_to_sim(
                    joint_pos, joint_vel, env_ids=solve_env_ids
                )

            # Re-evaluate the result of this seed, including the final update.
            current_pos = self.robot.data.body_pos_w[
                pending_env_ids, self.eef_body_id
            ]
            current_quat = self.robot.data.body_quat_w[
                pending_env_ids, self.eef_body_id
            ]
            current_eef_pos = current_pos + math_utils.quat_apply(
                current_quat, offset[pending_rows]
            )
            pos_error_norm = torch.linalg.norm(
                desired_eef_pos[pending_rows] - current_eef_pos, dim=-1
            )
            _, rot_error = math_utils.compute_pose_error(
                current_pos,
                current_quat,
                current_pos,
                desired_body_quat[pending_rows],
                rot_error_type="axis_angle",
            )
            rot_error_norm = torch.linalg.norm(rot_error, dim=-1)
            finite = torch.isfinite(pos_error_norm) & torch.isfinite(
                rot_error_norm
            )
            seed_success = (
                finite
                & (pos_error_norm < position_tolerance)
                & (rot_error_norm < orientation_tolerance)
            )
            success[pending_rows] = seed_success
            final_pos_error[pending_rows] = pos_error_norm
            final_rot_error_norm[pending_rows] = rot_error_norm

        # Keep the solved joint state as the drive target at the first step.
        final_joint_pos = self.robot.data.joint_pos[env_ids].clone()
        self.robot.set_joint_position_target(final_joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(zero_joint_vel, env_ids=env_ids)

        env.reaching_ik_success[env_ids] = success
        env.reaching_ik_attempt_count[env_ids] = attempt_count
        if not torch.all(success):
            failed_rows = torch.where(~success)[0]
            preview_rows = failed_rows[:8]
            details = [
                {
                    "env_id": int(env_ids[row].item()),
                    "target_pos_w": target_init_pos_w[row].tolist(),
                    "sweep_dir": env.sweep_dir[env_ids[row]].tolist(),
                    "desired_eef_pos_w": desired_eef_pos[row].tolist(),
                    "position_error": float(final_pos_error[row].item()),
                    "orientation_error": float(
                        final_rot_error_norm[row].item()
                    ),
                }
                for row in preview_rows
            ]
            raise RuntimeError(
                "Reaching-pose IK reset validation failed for "
                f"{len(failed_rows)}/{len(env_ids)} environments after "
                f"{len(seed_offsets)} joint seeds. No episode was started. "
                f"First failures: {details}"
            )


def sweeping_right_mode(target_index: int, rows: int, cols: int, device):
    """
    Identify objects in front, right, and diagonal positions of the target.
    If the target is in the last row, include all objects in the front rows.
    Returns results as a GPU Tensor.
    """
    target_row = target_index // cols
    target_col = target_index % cols

    # (1) 앞쪽 찾기 (모든 앞쪽 행 포함)
    if target_row > 0:
        front_rows = torch.arange(target_row - 1, -1, -1, device=device)
        front_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col)), dim=1)
    else:
        front_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (2) 오른쪽 찾기
    if target_col < cols - 1:
        right_index = torch.tensor([[target_row, target_col + 1]], device=device)
    else:
        right_index = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (3) 우측 대각선 찾기
    if target_row > 0 and target_col < cols - 1:
        diagonal_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col + 1)), dim=1)
    else:
        diagonal_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (4) 모든 결과를 GPU Tensor로 결합
    adjacent_array = torch.cat((front_indices, right_index, diagonal_indices), dim=0)

    return adjacent_array

def sweeping_left_mode(target_index: int, rows: int, cols: int, device):
    """
    Identify objects in front, left, and diagonal positions of the target.
    If the target is in the last row, include all objects in the front rows.
    Returns results as a GPU Tensor.
    """
    target_row = target_index // cols
    target_col = target_index % cols

    # (1) 앞쪽 찾기 (모든 앞쪽 행 포함)
    if target_row > 0:
        front_rows = torch.arange(target_row - 1, -1, -1, device=device)
        front_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col)), dim=1)
    else:
        front_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (2) 왼쪽 찾기
    if target_col > 0:
        left_index = torch.tensor([[target_row, target_col - 1]], device=device)
    else:
        left_index = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (3) 좌측 대각선 찾기
    if target_row > 0 and target_col > 0:
        diagonal_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col - 1)), dim=1)
    else:
        diagonal_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (4) 모든 결과를 GPU Tensor로 결합
    adjacent_array = torch.cat((front_indices, left_index, diagonal_indices), dim=0)

    return adjacent_array


def grasping_mode(target_index: int, rows: int, cols: int, device):
    """
    Identify objects in front, right, and diagonal positions of the target.
    If the target is in the last row, include all objects in the front rows.
    Returns results as a GPU Tensor.
    """
    target_row = target_index // cols
    target_col = target_index % cols

    # (1) 앞쪽 찾기 (모든 앞쪽 행 포함)
    if target_row > 0:
        front_rows = torch.arange(target_row - 1, -1, -1, device=device)
        front_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col)), dim=1)
    else:
        front_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    return front_indices
