# Contact-Rich Omnidirectional Sweeping

An Isaac Lab research workspace for a UR5e and an Inspire Hand to sweep a partially visible object toward a goal on a cluttered shelf. The policy combines an initial estimate of the visible geometry with fingertip contact and wrist force/torque feedback. The experiments run in simulation.

## What the policy observes

At reset, the reference task `Inspire-Sweep-Prof-Revision-v0_1` estimates which parts of the target and obstacles are visible from the front of the shelf. It fits an oriented bounding box to the largest visible target fragment and stores up to 16 visible obstacle fragments. Visibility is computed from scene geometry. The boxes stay fixed in world coordinates throughout the episode and are expressed in the current end-effector frame at each policy step.

During the sweep, 17 tactile channels and a six-axis wrist wrench provide new contact information. PPO commands six arm motion components and six hand actuators. The target is a cylinder, and the goal is sampled 6–12 cm away in the shelf plane. At the end of the 10-second episode, success requires the tracked target fragment to be within 3 cm of the goal without a simultaneous physical failure.

## Repository layout

| Path | Purpose |
| --- | --- |
| [src/inspire_sweep_baseline](src/inspire_sweep_baseline) | Registered Gymnasium tasks, scene and robot definitions, observations, rewards, PPO models, training and playback |
| [src/inspire_tactile_validation](src/inspire_tactile_validation) | Probe tests for the hand’s tactile pads |
| [src/axia80_feasibility](src/axia80_feasibility) | Wrist force/torque checks with known payloads and lever arms |
| [src/observed_obb](src/observed_obb) · [src/scene_randomization](src/scene_randomization) | Visible-object geometry and randomized shelf layouts |
| [IsaacLab](IsaacLab) · [example](example) | Bundled Isaac Lab 2.3.2 source, task specifications and the earlier sweeping project |
| [src/trash](src/trash) | Archived experiments, including the bundled Universal Robots descriptions |

## Set up locally

1. Clone the repository and enter the workspace. The dependency directories are included in the checkout.

   ```bash
   git clone https://github.com/JeonHaneul/Contact-Rich-Omnidirectional-Sweeping.git
   cd Contact-Rich-Omnidirectional-Sweeping
   ```

2. On a machine with a supported NVIDIA GPU, set up Isaac Sim 5.1 and activate its Python environment. For a binary installation, follow the bundled instructions for linking `IsaacLab/_isaac_sim` to the installation. Install the Isaac Lab extensions and RSL-RL from the workspace root. [Simulator setup](IsaacLab/docs/source/setup/installation/index.rst)

   ```bash
   ./IsaacLab/isaaclab.sh --install rsl_rl
   ./IsaacLab/isaaclab.sh -p -m pip install -e src/inspire_sweep_baseline \
     -e src/inspire_tactile_validation -e src/axia80_feasibility
   ```

3. Supply the original shelf USD together with its referenced files. Place the asset tree under `assets/shelf/Arena/Collected_speedrack_shape/` or set the variable below to an existing local file. The robot and hand models are bundled. The shelf is external, and the task also resolves YCB obstacle models through the Isaac Sim asset service or an accessible asset cache.

   ```bash
   export INSPIRE_SWEEP_SHELF_USD_PATH=/absolute/path/to/speedrack_shape.usd
   ```

## Train and replay

Run from the workspace root. Start with a small environment count to check scene loading before scaling training on your GPU.

```bash
TASK=Inspire-Sweep-Prof-Revision-v0_1
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task "$TASK" --num_envs 4 --headless
```

To replay a policy, point to a checkpoint trained with the same task and observation layout.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task "$TASK" --num_envs 1 --checkpoint /absolute/path/to/model_ITER.pt
```

## Check the sensors

The tactile probe exercises 17 pads at five positions each. The wrist test compares simulated measurements against known loads.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_tactile_validation/scripts/run_probe_test.py \
  --num-envs 85 --repeats 3 --headless
./IsaacLab/isaaclab.sh -p src/axia80_feasibility/scripts/run_payload_test.py \
  --headless --device cuda:0 --strict
```

The [reference task specification](example/Contact-Rich-Manipulation-Context/02_implementation/environments/PROF_REVISION_V0.md) describes observation coordinates, control, rewards and termination. [Source revisions and licenses](SOURCE.md)
