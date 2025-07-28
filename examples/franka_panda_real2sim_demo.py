#!/usr/bin/env python3
"""
REAL2SIM DEMO: YCB MUSTARD BOTTLE
    Real 3D scan data → MuJoCo simulation → Robot manipulation
    Using Yale-CMU-Berkeley Object Dataset with Franka Panda Robot
"""

import mujoco
import mujoco.viewer
import numpy as np
import os
import time
from pathlib import Path

def create_real2sim_scene():
    """Create MuJoCo XML scene with real scanned object and Franka Panda robot"""
    
    # Path to the real scan
    mesh_path = "data/meshes/mustard_bottle.stl"
    
    if not os.path.exists(mesh_path):
        print(f"ERROR: Mesh file not found: {mesh_path}")
        print("TIP: Run: python examples/tool_reconstruct.py first")
        return None
    
    # Use the standalone Panda file we prepared
    scene_file = "examples/real2sim_standalone_panda.xml"
    
    if not os.path.exists(scene_file):
        print(f"ERROR: Standalone Panda file not found: {scene_file}")
        return None
    
    return scene_file

class FrankaPandaController:
    """Controller for the actual Franka Panda robot from mujoco_menagerie"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Get the actual joint and actuator names from the Panda model
        print(f"Model info:")
        print(f"   Bodies: {model.nbody}")
        print(f"   Joints: {model.njnt}")
        print(f"   Actuators: {model.nu}")
        
        # Print joint names for debugging
        joint_names = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                joint_names.append(name)
        print(f"   Joint names: {joint_names}")
        
        # Print actuator names for debugging  
        actuator_names = []
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                actuator_names.append(name)
        print(f"   Actuator names: {actuator_names}")
        
        # Define joint and actuator names (based on actual Panda model)
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7', 'actuator8']
        
        # Get indices
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names]
        
        print(f"Franka Panda Controller initialized!")
        print(f"   Found joints: {[name for name, id in zip(self.joint_names, self.joint_ids) if id != -1]}")
        print(f"   Found actuators: {[name for name, id in zip(self.actuator_names, self.actuator_ids) if id != -1]}")
        
        # Create manipulation sequences
        self.sequences = self.create_manipulation_sequences()
        print(f"Loaded {len(self.sequences)} manipulation sequences")
    
    def create_manipulation_sequences(self):
        """Create manipulation sequences for picking up the mustard bottle"""
        return [
            # Home position (matching Panda's home keyframe)
            {'name': 'Home', 'joints': [0, 0, 0, -1.57079, 0, 1.57079, -0.7853], 'gripper': 255},
            # Approach bottle
            {'name': 'Approach', 'joints': [0.5, -0.2, 0.1, -1.8, 0.1, 1.6, 0.8], 'gripper': 255},
            # Pre-grasp position
            {'name': 'Pre-grasp', 'joints': [0.6, -0.1, 0.2, -1.6, 0.2, 1.5, 0.9], 'gripper': 255},
            # Grasp position (close gripper)
            {'name': 'Grasp', 'joints': [0.6, -0.1, 0.2, -1.6, 0.2, 1.5, 0.9], 'gripper': 50},
            # Lift bottle
            {'name': 'Lift', 'joints': [0.4, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7], 'gripper': 50},
            # Move to new location
            {'name': 'Move', 'joints': [-0.2, -0.3, -0.1, -1.8, -0.1, 1.5, 0.5], 'gripper': 50},
            # Place down
            {'name': 'Place', 'joints': [-0.3, -0.1, 0.1, -1.6, 0.1, 1.4, 0.6], 'gripper': 50},
            # Release gripper
            {'name': 'Release', 'joints': [-0.3, -0.1, 0.1, -1.6, 0.1, 1.4, 0.6], 'gripper': 255},
            # Retract
            {'name': 'Retract', 'joints': [-0.1, -0.5, -0.2, -2.0, -0.2, 1.2, 0.4], 'gripper': 255},
            # Return home
            {'name': 'Return Home', 'joints': [0, 0, 0, -1.57079, 0, 1.57079, -0.7853], 'gripper': 255},
        ]
    
    def execute_sequence(self, viewer, sequence_step):
        """Execute one step of the manipulation sequence"""
        sequence = self.sequences[sequence_step]
        print(f"Executing: {sequence['name']}")
        
        target_joints = sequence['joints']
        target_gripper = sequence['gripper']
        
        # Execute for 2 seconds (1000 steps at 0.002s timestep)
        for step in range(1000):
            # Set joint targets
            for i, (actuator_id, target) in enumerate(zip(self.actuator_ids[:7], target_joints)):
                if actuator_id != -1:
                    self.data.ctrl[actuator_id] = target
            
            # Set gripper target (actuator8 controls both fingers via tendon)
            if len(self.actuator_ids) > 7 and self.actuator_ids[7] != -1:
                self.data.ctrl[self.actuator_ids[7]] = target_gripper
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            time.sleep(0.002)
        
        # Show bottle position for debugging
        if hasattr(self.data, 'body'):
            try:
                bottle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mustard_bottle')
                if bottle_id != -1:
                    bottle_pos = self.data.xpos[bottle_id]
                    print(f"   Bottle position: [{bottle_pos[0]:.3f}, {bottle_pos[1]:.3f}, {bottle_pos[2]:.3f}]")
            except:
                pass

def run_demo():
    """Run the Real2Sim demo with actual Franka Panda robot"""
    
    print("=" * 80)
    print("REAL2SIM DEMO: YCB MUSTARD BOTTLE + FRANKA PANDA")
    print("    Real 3D scan data → MuJoCo simulation → Franka Panda manipulation")
    print("    Using Yale-CMU-Berkeley Object Dataset")
    print("=" * 80)
    print()
    
    # Create scene
    print("Creating simulation scene with Franka Panda...")
    scene_file = create_real2sim_scene()
    if not scene_file:
        return
    
    print(f"Scene saved: {scene_file}")
    
    # Check file size
    mesh_path = "data/meshes/mustard_bottle.stl"
    if os.path.exists(mesh_path):
        size = os.path.getsize(mesh_path)
        print(f"Using real scan: {mesh_path} ({size} bytes)")
    
    print()
    print("Loading MuJoCo simulation...")
    
    try:
        # Load model
        model = mujoco.MjModel.from_xml_path(scene_file)
        data = mujoco.MjData(model)
        
        print("Simulation loaded successfully!")
        
        # Initialize controller
        controller = FrankaPandaController(model, data)
        
        print()
        print("Demo Options:")
        print("1. Interactive viewer (recommended)")
        print("2. Automatic manipulation sequence")
        choice = input("Choice (1/2): ")
        
        if choice == "2":
            print()
            print("Running automatic Franka Panda manipulation...")
            
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Set initial pose using the corrected home keyframe
                if model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                
                for i, sequence in enumerate(controller.sequences):
                    print(f"   Step {i+1}/{len(controller.sequences)}: {sequence['name']}")
                    controller.execute_sequence(viewer, i)
                    time.sleep(0.5)  # Brief pause between actions
            
            print("Automatic sequence completed!")
        
        else:
            print()
            print("Starting interactive viewer...")
            print("Controls:")
            print("  - Mouse: Rotate camera")
            print("  - Right-click + drag: Pan")
            print("  - Scroll: Zoom")
            print("  - Space: Pause/unpause")
            print("  - Ctrl+R: Reset")
            
            # Set initial pose using the corrected home keyframe
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            
            print("Interactive simulation running!")
            print("The yellow object is a REAL 3D scan of a mustard bottle!")
            print("The robot is a Franka Emika Panda - industry standard for research!")
            print("Press Ctrl+C to exit...")
            
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.002)
    
    except Exception as e:
        print(f"ERROR: Error loading simulation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Demo ended successfully!")

if __name__ == "__main__":
    run_demo() 