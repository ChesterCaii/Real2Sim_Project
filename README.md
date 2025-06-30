# Real-to-Simulation Pipeline

Transform real-world point clouds into interactive MuJoCo robot simulations. This project demonstrates a complete pipeline from 3D reconstruction to physics simulation.

## Overview

**Input**: Point cloud file (`bunny.pcd`)  
**Output**: Interactive MuJoCo simulation with Franka robot + reconstructed object

### Pipeline Stages:
1. **3D Reconstruction** - Convert point cloud to mesh using Poisson surface reconstruction
2. **Model Integration** - Combine reconstructed object with robot model  
3. **Physics Simulation** - Launch interactive MuJoCo viewer

## Current Status

**WORKING**: End-to-end pipeline successfully tested and verified
- 3D reconstruction: `bunny.pcd` → `bunny_final.stl` (469KB mesh)
- Scene loading: 13 bodies, 16 DOF, 68 meshes, 8 actuators  
- Interactive simulation: MuJoCo viewer with robot + reconstructed bunny
- **Phase 3A**: Robot control and manipulation of reconstructed objects
- **Phase 3B**: Object segmentation and multi-object reconstruction

## Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/ChesterCaii/Real2Sim_Project.git
cd Real2Sim_Project

# Activate virtual environment (recommended)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

**Step 1: 3D Reconstruction**
```bash
python reconstruct_mesh.py
```
Converts `bunny.pcd` → `bunny_final.stl`

**Step 2: Launch Simulation**
```bash
# On macOS (recommended):
mjpython run_real2sim.py

# On Linux/Windows:
python run_real2sim.py
```

**Expected Result**: Interactive MuJoCo viewer with Franka robot and reconstructed bunny

**Step 3: Robot Control (NEW)**
```bash
# Autonomous manipulation demonstration:
mjpython robot_control_demo.py

# Interactive joint control:
mjpython interactive_robot_control.py
```

**Expected Result**: Robot arm moves through manipulation sequences, interacting with reconstructed object

**Step 4: Object Segmentation & Multi-Object Scenes (NEW - Phase 3B)**
```bash
# Object segmentation using SAM:
python object_segmentation.py

# Multi-object 3D reconstruction:
python reconstruct_multi_objects.py

# Test multi-object simulation:
mjpython test_multi_object_scene.py
```

**Expected Result**: Multiple objects automatically segmented, reconstructed in 3D, and simulated together

### Controls
- **Mouse**: Rotate camera
- **Right-click + drag**: Pan camera  
- **Scroll**: Zoom in/out
- **Space**: Pause/unpause physics
- **Ctrl+R**: Reset simulation

## File Structure

- `reconstruct_mesh.py` - 3D reconstruction from point cloud to mesh
- `run_real2sim.py` - Main simulation launcher (clean interface)
- `fixed_simulation.py` - Detailed debugging version
- **`robot_control_demo.py`** - Autonomous robot manipulation demonstration (Phase 3A)
- **`interactive_robot_control.py`** - Manual robot joint control interface (Phase 3A)
- `bunny.pcd` - Input point cloud data
- `bunny_final.stl` - Reconstructed 3D mesh
- `mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml` - Combined simulation scene

## Simulation Controls

- **Mouse**: Rotate and zoom camera
- **Right-click + drag**: Pan camera
- **Scroll**: Zoom in/out
- **Space**: Pause/unpause physics
- **Ctrl+R**: Reset simulation

## Technical Achievements

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

## Pipeline Statistics

- **Bodies**: 13 (12 robot + 1 reconstructed object)
- **Degrees of Freedom**: 16 (7 arm + 2 gripper + 6 object free joint + 1 world)
- **Meshes**: 68 (visual and collision meshes for detailed robot)
- **Actuators**: 8 (7 joint actuators + 1 gripper)

## Research Applications

This pipeline enables research in:
- **Sim-to-Real Transfer**: Training policies in simulation with real object geometry
- **Digital Twins**: Creating digital replicas of real-world manipulation scenarios
- **Robotic Grasping**: Testing grasp strategies on reconstructed real objects
- **Scene Understanding**: Combining perception and interaction in robotics

## Educational Value

Suitable for learning:
- 3D reconstruction techniques
- MuJoCo physics simulation
- XML scene modeling
- Real-to-simulation pipelines
- Robotic manipulation setup

## Future Extensions

### Phase 3A: Robot Control ✅ COMPLETED
- ✅ Basic robot arm control (7 DOF joint control)
- ✅ Gripper control and manipulation
- ✅ Smooth motion planning and execution
- ✅ Real-time feedback and monitoring
- ✅ Interactive control interfaces

### Phase 3B: Advanced Segmentation
- Integrate SAM (Segment Anything Model) for object segmentation
- Process real RGB-D camera feeds
- Extract individual objects from cluttered scenes
- Handle multi-object reconstruction scenarios

### Phase 3C: Physics Parameter Estimation
- Estimate object mass, friction, and inertial properties
- System identification from real-world observations
- Improve simulation fidelity with realistic physics
- Calibrate simulation parameters from real interactions

### Phase 3D: Advanced Manipulation
- Implement intelligent grasp planning
- Contact-rich manipulation tasks
- Task and motion planning integration
- Closed-loop control with force feedback

## Acknowledgments

- **MuJoCo**: Physics simulation framework
- **Open3D**: 3D reconstruction and processing
- **MuJoCo Menagerie**: High-quality robot models
- **Stanford Computer Graphics Laboratory**: Bunny dataset

## License

This project is open source. Please see individual component licenses:
- MuJoCo Menagerie: Apache 2.0
- Open3D: MIT License

---

**Status**: Pipeline Successfully Implemented and Tested on macOS

Built for the robotics and computer vision community. 