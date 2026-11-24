"""Check physical axis alignment and bottom-origin reaching height."""
from types import SimpleNamespace as NS
import torch

from test_inspire_sensor_observations import _functions, math_utils


def test_correct_alignment_rejects_row_based_false_positive_and_tracks_shelf_rotation():
    functions = _functions("sweeping_policy/mdp/inspire_mass_rewards.py", {
        "torch": torch, "matrix_from_quat": math_utils.matrix_from_quat,
    })
    # Cyclic tool rotation: row Y is up, but physical column Y is horizontal.
    false_positive = torch.tensor([.5, -.5, -.5, -.5])
    # +90 about X: physical tool Y really is up.
    aligned = torch.tensor([2**-.5, 2**-.5, 0., 0.])
    upside_down = torch.tensor([2**-.5, -2**-.5, 0., 0.])
    frames = torch.stack((false_positive, aligned, upside_down))
    shelf = torch.tensor([1., 0., 0., 0.]).expand(3, 4)
    env = NS(scene={"ee_frame": NS(data=NS(target_quat_w=frames[:, None])),
                    "shelf": NS(data=NS(default_root_state=torch.cat((torch.zeros(3, 3), shelf), dim=-1)))})
    reward = functions["align_eef_y_with_shelf_z"]
    torch.testing.assert_close(reward(env), torch.tensor([0., 1., -1.]), atol=1e-6, rtol=1e-6)
    # Rotate the complete scene: true axis alignment must be unchanged.
    global_q = math_utils.quat_from_euler_xyz(torch.tensor([.4]), torch.tensor([.2]), torch.tensor([-.6])).expand(3, 4)
    env.scene["ee_frame"].data.target_quat_w = math_utils.quat_mul(global_q, frames)[:, None]
    env.scene["shelf"].data.default_root_state[:, 3:7] = global_q
    torch.testing.assert_close(reward(env), torch.tensor([0., 1., -1.]), atol=1e-6, rtol=1e-6)


def test_reach_is_maximal_at_four_centimeters_above_bottom_in_both_sweep_directions():
    functions = _functions("sweeping_policy/mdp/inspire_mass_rewards.py", {"torch": torch})
    bottom = torch.tensor([[[-.75, 0., 1.05]], [[-.75, .2, 1.05]]])
    reach = torch.tensor([[[-.77, -.04, 1.09]], [[-.77, .24, 1.09]]])
    env = NS(num_envs=2, device="cpu", target_id=torch.zeros(2, 1, dtype=torch.long),
             target_width=torch.full((2, 1), .04), sweep_dir=torch.tensor([[0., 1., 0.], [0., -1., 0.]]),
             scene={"object_collection": NS(data=NS(object_pos_w=bottom)), "ee_frame": NS(data=NS(target_pos_w=reach))})
    reward = functions["reaching_at_object_height"]
    torch.testing.assert_close(reward(env), torch.ones(2), atol=1e-6, rtol=1e-6)
    env.scene["ee_frame"].data.target_pos_w[:, 0, 2] += .05
    torch.testing.assert_close(reward(env), torch.full((2,), torch.exp(torch.tensor(-.5))), atol=1e-6, rtol=1e-6)
