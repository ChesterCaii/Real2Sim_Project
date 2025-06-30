# Real-to-Simulation Pipeline 🤖🐰

A complete pipeline that converts real-world 3D point clouds into interactive MuJoCo physics simulations with robotic manipulation.

## 🎯 Project Overview

This project demonstrates a **Real-to-Simulation (Real2Sim)** pipeline that:

1. **Reconstructs 3D meshes** from point cloud data using Poisson surface reconstruction
2. **Combines reconstructed objects** with robotic models in physics simulation
3. **Enables interactive simulation** where robots can interact with real-world reconstructed objects

## 🎬 Demo

The pipeline successfully combines:
- **Franka Emika Panda** 7-DOF robotic arm (from MuJoCo Menagerie)
- **Stanford Bunny** reconstructed from point cloud data
- **Real-time physics simulation** in MuJoCo

## 🛠️ Pipeline Components

### Phase 1: 3D Reconstruction ✅
- **Input**: Point cloud data (`bunny.pcd`)
- **Process**: Poisson surface reconstruction using Open3D
- **Output**: High-quality 3D mesh (`bunny_final.stl`)

### Phase 2: Simulation Integration ✅
- **Challenge**: Combine robot models with reconstructed objects
- **Solution**: Custom XML scene combining robot and object models
- **Result**: Interactive physics simulation with full robot control

### Phase 3: Future Extensions 🚀
- Object segmentation from RGB-D scenes using SAM
- Physics parameter estimation (mass, friction, etc.)
- Automated robot manipulation and grasping

## 🚀 Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install mujoco open3d numpy

# Clone MuJoCo Menagerie (robot models) - REQUIRED
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

**Important**: The `mujoco_menagerie` directory is not included in this repository due to its large size. You must clone it separately as shown above.

### Running the Pipeline

1. **Reconstruct 3D mesh from point cloud:**
```bash
python reconstruct_mesh.py
```

2. **Launch the combined simulation:**
```bash
mjpython run_real2sim.py
```

## 📁 Key Files

- `reconstruct_mesh.py` - 3D reconstruction from point cloud to mesh
- `run_real2sim.py` - Main simulation launcher (clean interface)
- `fixed_simulation.py` - Detailed debugging version
- `bunny.pcd` - Input point cloud data
- `bunny_final.stl` - Reconstructed 3D mesh
- `mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml` - Combined simulation scene

## 🎮 Simulation Controls

- **Mouse**: Rotate and zoom camera
- **Right-click + drag**: Pan camera
- **Scroll**: Zoom in/out
- **Space**: Pause/unpause physics
- **Ctrl+R**: Reset simulation

## 🔧 Technical Achievements

### Problem Solved: Model Merging
- **Initial approach**: Attempted to use non-existent `mj_mergeModels()` function
- **Solution**: Created unified XML scene combining robot and object definitions
- **Result**: Single simulation with 13 bodies, 16 DOF, 68 meshes, 8 actuators

### Keyframe Compatibility
- **Challenge**: Robot keyframes incompatible with additional object DOF
- **Solution**: Removed conflicting keyframes while preserving robot functionality
- **Result**: Stable simulation with proper physics integration

### Asset Path Management
- **Challenge**: Complex mesh asset path resolution in MuJoCo
- **Solution**: Proper meshdir configuration and asset organization
- **Result**: All 68+ robot and object meshes load correctly

## 📊 Pipeline Statistics

- **Bodies**: 13 (12 robot + 1 reconstructed object)
- **Degrees of Freedom**: 16 (7 arm + 2 gripper + 6 object free joint + 1 world)
- **Meshes**: 68 (visual and collision meshes for detailed robot)
- **Actuators**: 8 (7 joint actuators + 1 gripper)

## 🔬 Research Applications

This pipeline enables research in:
- **Sim-to-Real Transfer**: Training policies in simulation with real object geometry
- **Digital Twins**: Creating digital replicas of real-world manipulation scenarios
- **Robotic Grasping**: Testing grasp strategies on reconstructed real objects
- **Scene Understanding**: Combining perception and interaction in robotics

## 🎓 Educational Value

Perfect for learning:
- 3D reconstruction techniques
- MuJoCo physics simulation
- XML scene modeling
- Real-to-simulation pipelines
- Robotic manipulation setup

## 🚀 Future Extensions

### Phase 3A: Advanced Segmentation
- Integrate SAM (Segment Anything Model) for object segmentation
- Process real RGB-D camera feeds
- Extract individual objects from cluttered scenes

### Phase 3B: Physics Parameter Estimation
- Estimate object mass, friction, and inertial properties
- System identification from real-world observations
- Improve simulation fidelity

### Phase 3C: Automated Manipulation
- Implement robot control for object interaction
- Grasp planning and execution
- Task and motion planning integration

## 🙏 Acknowledgments

- **MuJoCo**: Physics simulation framework
- **Open3D**: 3D reconstruction and processing
- **MuJoCo Menagerie**: High-quality robot models
- **Stanford Computer Graphics Laboratory**: Bunny dataset

## 📜 License

This project is open source. Please see individual component licenses:
- MuJoCo Menagerie: Apache 2.0
- Open3D: MIT License

---

**🎉 Status: Pipeline Successfully Implemented and Tested on macOS**

Built with ❤️ for the robotics and computer vision community. 