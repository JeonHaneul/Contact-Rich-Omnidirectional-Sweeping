# Source and maintenance

Haneul Jeon maintains this distribution for omnidirectional shelf-sweeping research. The base workspace is [7cmdehdrb/Contact-Rich-Manipulation at 8ce37bf](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/tree/8ce37bfbfc7df5e85608f7c46f5b62d032e6869a). [Upstream commit history](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/commits/8ce37bfbfc7df5e85608f7c46f5b62d032e6869a/) records the original contributors. The project includes the original Academic Free License 3.0 and component-specific notices.

## Dependency revisions

| Workspace path | Upstream | Pinned commit |
| --- | --- | --- |
| `IsaacLab` | [7cmdehdrb/IsaacLab](https://github.com/7cmdehdrb/IsaacLab/tree/929f0fe1e9c1699efebf31b2ad0ab723a7d6e7e9) | `929f0fe1e9c1699efebf31b2ad0ab723a7d6e7e9` |
| `example/Contact-Rich-Manipulation-Context` | [7cmdehdrb/Contact-Rich-Manipulation-Context](https://github.com/7cmdehdrb/Contact-Rich-Manipulation-Context/tree/cc69933d013a0a050aa360e2b984cae9ada05337) | `cc69933d013a0a050aa360e2b984cae9ada05337` |
| `example/Sweep-Policy` | [IROL-SSU/Sweeping-Policy-DRL-Sim-Train](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/tree/baa3740e38c1a693b2094dc4d141b72de0dd3142) | `baa3740e38c1a693b2094dc4d141b72de0dd3142` |
| `src/trash/Universal_Robots_ROS2_Description` | [UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/18e6f603b3ebc2ec479fecb62d6be544b15755e9) | `18e6f603b3ebc2ec479fecb62d6be544b15755e9` |

All four components are included as regular Git directories. Local asset defaults and machine-specific command examples have been adapted for this checkout. In the original Sweep-Policy package, 103 of the 108 files are byte-identical to the pinned revision. Five files adapt the shelf path, its documentation and the associated tests. The reference test also checks the scalar IK action scale of 0.5 implemented at the pinned revision. The presentation PDF and tactile recording in the context project are stored as full files and match their upstream Git LFS SHA-256 checksums.

Each component retains its upstream licenses and attribution. The UR description includes license information alongside its meshes.
