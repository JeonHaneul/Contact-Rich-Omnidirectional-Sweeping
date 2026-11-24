# Reference projects

- [Contact-Rich-Manipulation-Context](Contact-Rich-Manipulation-Context/) contains the design notes, experiment records and original media. The [reference task specification](Contact-Rich-Manipulation-Context/02_implementation/environments/PROF_REVISION_V0.md) defines the observation layout and control assumptions.
- [Sweep-Policy](Sweep-Policy/) is the earlier training package pinned by this workspace. Its [task configuration](Sweep-Policy/sweeping_policy/shelf_sweep_random_env_cfg.py) and [PPO configuration](Sweep-Policy/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py) are used for baseline comparisons.

The two projects are included as ordinary directories. To train the contact-rich policy, use the [main workspace launcher](../README.md#train-and-replay). [Source revisions](../SOURCE.md)
