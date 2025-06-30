# Real-to-Simulation Pipeline

Transform real-world point clouds into interactive MuJoCo robot simulations. This project demonstrates a complete pipeline from 3D reconstruction to physics simulation.

## Overview

**Input**: Point cloud file (`data/point_clouds/bunny.pcd`) OR Live camera feed  
**Output**: Interactive MuJoCo simulation with Franka robot + reconstructed object(s)

### Pipeline Stages:
1. **3D Reconstruction** - Convert point cloud to mesh using Poisson surface reconstruction
2. **Model Integration** - Combine reconstructed object with robot model  
3. **Physics Simulation** - Launch interactive MuJoCo viewer
4. **Live Integration** - Real-time camera feeds with dynamic simulation updates

## Current Status

**WORKING**: Complete end-to-end pipeline from static files to real-time camera integration
- ✅ 3D reconstruction: `bunny.pcd` → `bunny_final.stl` (469KB mesh)
- ✅ Scene loading: 13 bodies, 16 DOF, 68 meshes, 8 actuators  
- ✅ Interactive simulation: MuJoCo viewer with robot + reconstructed bunny
- ✅ **Phase 3A**: Robot control and manipulation of reconstructed objects
- ✅ **Phase 3B**: Object segmentation and multi-object reconstruction  
- ✅ **Phase 4**: Real camera integration with live object detection

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

### Easy Pipeline Launcher 🆕
```bash
# Phase 1: 3D Reconstruction
python run_pipeline.py 1

# Phase 2: Simulation
python run_pipeline.py 2

# Phase 3A: Robot Control
python run_pipeline.py 3a

# Phase 3B: Multi-Object Reconstruction  
python run_pipeline.py 3b

# Phase 4: Camera Integration
python run_pipeline.py 4

# Phase 4: Live Simulation Bridge
python run_pipeline.py 4 --live

# Run Tests
python run_pipeline.py test
```

### Manual Commands (Alternative)

**Step 1: 3D Reconstruction (Static)**
```bash
python src/reconstruction/reconstruct_mesh.py
```
Converts `data/point_clouds/bunny.pcd` → `data/meshes/bunny_final.stl`

**Step 2: Launch Simulation (Static)**
```bash
# On macOS (recommended):
mjpython src/simulation/run_real2sim.py

# On Linux/Windows:
python src/simulation/run_real2sim.py
```

**Expected Result**: Interactive MuJoCo viewer with Franka robot and reconstructed bunny

**Step 3: Robot Control (Phase 3A)**
```bash
# Autonomous manipulation demonstration:
mjpython src/control/robot_control_demo.py

# Interactive joint control:
mjpython src/control/interactive_robot_control.py
```

**Expected Result**: Robot arm moves through manipulation sequences, interacting with reconstructed object

**Step 4: Object Segmentation & Multi-Object Scenes (Phase 3B)**
```bash
# Object segmentation using SAM:
python src/reconstruction/object_segmentation.py

# Multi-object 3D reconstruction:
python src/reconstruction/reconstruct_multi_objects.py

# Test multi-object simulation:
mjpython src/simulation/test_multi_object_scene.py
```

**Expected Result**: Multiple objects automatically segmented, reconstructed in 3D, and simulated together

**Step 5: Live Camera Integration (Phase 4) 🆕**
```bash
# Test camera setup:
python tests/test_camera_integration.py

# Live object detection only:
python src/live/camera_integration.py

# Full live simulation bridge:
python src/live/live_simulation_bridge.py
```

**Expected Result**: Real-time camera feed → object detection → 3D reconstruction → dynamic simulation updates

### Controls
- **Mouse**: Rotate camera
- **Right-click + drag**: Pan camera  
- **Scroll**: Zoom in/out
- **Space**: Pause/unpause physics
- **Ctrl+R**: Reset simulation
- **Q**: Quit (in camera windows)

## File Structure

### Core Source Code (`src/`)
- **`reconstruction/`** - Phase 1 & 3B: 3D reconstruction and object segmentation
  - `reconstruct_mesh.py` - Main 3D reconstruction from point cloud to mesh
  - `object_segmentation.py` - SAM-based automatic object detection  
  - `reconstruct_multi_objects.py` - Multi-object 3D reconstruction
  - `simplify_meshes.py` - Mesh optimization for MuJoCo compatibility
- **`simulation/`** - Phase 2: MuJoCo simulation integration
  - `run_real2sim.py` - Main simulation launcher (clean interface)
  - `fixed_simulation.py` - Detailed debugging version
  - `test_multi_object_scene.py` - Multi-object simulation testing
- **`control/`** - Phase 3A: Robot control and manipulation
  - `robot_control_demo.py` - Autonomous robot manipulation demonstration
  - `interactive_robot_control.py` - Manual robot joint control interface
- **`live/`** - Phase 4: Real-time camera integration 🆕
  - `camera_integration.py` - Real-time RGB-D camera processing with live object detection
  - `live_simulation_bridge.py` - Bridge between live camera and dynamic MuJoCo simulation

### Data Directories
- **`data/`** - Input data and assets
  - `point_clouds/` - Point cloud files (`.pcd`, `.ply`)
  - `meshes/` - 3D mesh files (`.stl`)
  - `scenes/` - MuJoCo scene XML files
- **`models/`** - Machine learning models
  - `sam_vit_b.pth` - Segment Anything Model checkpoint
- **`outputs/`** - Generated outputs
  - `segmented_objects/` - Object segmentation results
  - `reconstructed_objects/` - 3D reconstruction outputs

### Testing & Examples
- **`tests/`** - Test suites and validation
  - `test_camera_integration.py` - Camera setup and component testing
- **`examples/`** - Example scripts and legacy code
- **`docs/`** - Documentation and images

### External Dependencies
- `mujoco_menagerie/` - High-quality robot models from Google DeepMind

## Phase 4: Live Camera Integration Features

### Supported Cameras
- **Intel RealSense D400 series** (preferred - true RGB-D)
- **Standard webcams** (with synthetic depth generation)
- Auto-detection and fallback

### Real-Time Capabilities
- **Live Object Detection**: SAM or OpenCV background subtraction
- **Streaming 3D Reconstruction**: Point cloud → mesh conversion
- **Dynamic Simulation Updates**: Objects appear/disappear in simulation
- **Multi-threaded Processing**: Separate camera, detection, and simulation threads

### Performance
- **Real-time FPS**: 15-30 FPS camera feed
- **Detection Rate**: 1-5 Hz object detection updates  
- **Simulation Updates**: 1-2 Hz dynamic scene reloading
- **Optimized for Speed**: Reduced SAM parameters, mesh simplification

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

### Real-Time Performance Optimization
- **Challenge**: SAM too slow for real-time use (5-10 seconds per frame)
- **Solution**: Optimized parameters, background subtraction fallback, multi-threading
- **Result**: Real-time object detection at 15+ FPS

## Pipeline Statistics

- **Bodies**: 13+ (12 robot + 1+ reconstructed objects)
- **Degrees of Freedom**: 16+ (7 arm + 2 gripper + 6 object free joint + 1 world)
- **Meshes**: 68+ (visual and collision meshes for detailed robot + dynamic objects)
- **Actuators**: 8 (7 joint actuators + 1 gripper)
- **Live Objects**: Unlimited (limited by performance)

## Research Applications

This pipeline enables research in:
- **Sim-to-Real Transfer**: Training policies in simulation with real object geometry
- **Digital Twins**: Creating digital replicas of real-world manipulation scenarios
- **Robotic Grasping**: Testing grasp strategies on reconstructed real objects
- **Scene Understanding**: Combining perception and interaction in robotics
- **Live Adaptation**: Real-time simulation updates based on changing environments
- **Human-Robot Interaction**: Dynamic object manipulation in shared workspaces

## Educational Value

Suitable for learning:
- 3D reconstruction techniques
- MuJoCo physics simulation
- XML scene modeling
- Real-to-simulation pipelines
- Robotic manipulation setup
- Computer vision and object detection
- Real-time system integration
- Multi-threaded programming

## Current Phases Status

### ✅ Phase 1: Basic Pipeline (COMPLETE)
- Single object reconstruction (bunny.pcd → bunny_final.stl)
- Static MuJoCo simulation integration
- Interactive viewer with robot + object

### ✅ Phase 2: Scene Integration (COMPLETE)  
- Robot + object combined simulation
- Physics interaction and collision detection
- Stable multi-body dynamics

### ✅ Phase 3A: Robot Control (COMPLETE)
- 7 DOF arm control and gripper manipulation
- Smooth motion planning and execution
- Real-time feedback and monitoring
- Interactive control interfaces

### ✅ Phase 3B: Multi-Object Reconstruction (COMPLETE)
- SAM-based object segmentation from images
- Individual 3D reconstruction of multiple objects
- Multi-object physics simulation (6 objects, 36 DOF)
- Mesh optimization for MuJoCo compatibility

### ✅ Phase 4: Real Camera Integration (COMPLETE) 🆕
- **Live RGB-D camera feeds** (Intel RealSense + webcam support)
- **Real-time object detection** (SAM + OpenCV background subtraction)
- **Streaming 3D reconstruction** (point cloud → mesh → simulation)
- **Dynamic simulation updates** (objects appear/disappear in real-time)
- **Multi-threaded architecture** (camera, detection, simulation threads)
- **Performance optimization** (15+ FPS camera, 1-2 Hz simulation updates)

## Future Extensions

### Phase 5: Advanced Perception
- **YOLO/DINO object classification** (recognize specific object types)
- **6D pose estimation** (precise object orientation)
- **Semantic segmentation** (understand object properties)
- **Temporal consistency** (object tracking across frames)

### Phase 6: Intelligent Manipulation
- **Grasp planning** based on reconstructed geometry
- **Contact-rich manipulation** with force feedback
- **Task and motion planning** integration
- **Closed-loop visual servoing**

### Phase 7: Multi-Agent Systems
- **Multiple robots** in shared workspace
- **Human-robot collaboration** with live human detection
- **Distributed simulation** across multiple computers
- **Cloud-based processing** for complex scenes

## Camera Setup Recommendations

### Intel RealSense D435i (Recommended)
```bash
# Install Intel RealSense SDK
# Ubuntu/Debian:
sudo apt-get install librealsense2-dev

# macOS (Homebrew):
brew install librealsense

# Windows: Download from Intel website
```

### Alternative Cameras
- **Azure Kinect**: Excellent depth quality, requires separate SDK
- **Webcam + depth estimation**: Works but limited accuracy
- **Stereo cameras**: Good for custom setups

### SAM Model Setup
```bash
# Download SAM checkpoint (358MB for vit_b) to models/ directory
cd models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth sam_vit_b.pth
cd ..
```

## Troubleshooting

### Common Issues
1. **Camera not detected**: Check USB connection, install drivers
2. **SAM too slow**: Use OpenCV fallback or smaller model
3. **Simulation crashes**: Check mesh complexity, use simplify_meshes.py
4. **Memory issues**: Reduce update rate, limit number of objects

### Performance Tips
1. **Use Intel RealSense** for best RGB-D quality
2. **Enable GPU acceleration** for SAM (CUDA)
3. **Adjust update rates** based on system performance
4. **Close unnecessary applications** during live demos

## Acknowledgments

- **MuJoCo**: Physics simulation framework
- **Open3D**: 3D reconstruction and processing
- **MuJoCo Menagerie**: High-quality robot models
- **Stanford Computer Graphics Laboratory**: Bunny dataset
- **Meta Research**: Segment Anything Model (SAM)
- **Intel**: RealSense SDK and hardware support

## License

This project is open source. Please see individual component licenses:
- MuJoCo Menagerie: Apache 2.0
- Open3D: MIT License
- SAM: Apache 2.0

---

**Status**: Complete Real-to-Simulation Pipeline with Live Camera Integration

Built for the robotics and computer vision community. 
Ready for research, education, and real-world applications! 🚀 