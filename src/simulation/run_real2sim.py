#!/usr/bin/env python3
"""
Real-to-Simulation Pipeline - Final Demo
=========================================

This script loads the combined Franka Panda robot and reconstructed bunny mesh
into a single MuJoCo physics simulation. It includes an option to record a
video of the simulation.

Usage:
    mjpython run_real2sim.py

Requirements:
    - MuJoCo Python bindings
    - mujoco_menagerie (Franka robot models)
    - bunny_final.stl (reconstructed mesh)
    - imageio (for video recording)

Author: Real2Sim Project
"""

import mujoco
import mujoco.viewer
import sys
import os
import time
import imageio

def main():
    print("=" * 60)
    print(" REAL-TO-SIMULATION PIPELINE DEMO ")
    print("=" * 60)
    print()
    
    # Check if running with mjpython (required on macOS)
    if sys.platform == "darwin" and "mjpython" not in sys.executable:
        print(" Error: On macOS, this script should be run with 'mjpython'")
        print("   Try: mjpython src/simulation/run_real2sim.py")
        print()
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Path to our combined scene
    scene_path = os.path.join(project_root, "mujoco_menagerie", "franka_emika_panda", "robot_bunny_scene.xml")
    
    if not os.path.exists(scene_path):
        print(" Error: Combined scene file not found!")
        print(f"   Expected: {scene_path}")
        print("   Make sure you've run the setup correctly.")
        return
    
    try:
        print(" Loading combined robot and bunny scene...")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print(f" Successfully loaded scene!")
        print(f"    Bodies: {model.nbody}")
        print(f"    DOFs: {model.nq}")
        print(f"    Meshes: {model.nmesh}")
        print(f"    Actuators: {model.nu}")
        print()
        
        # --- Video Recording Logic ---
        record_video_choice = input("Record a 10-second video? (y/n): ").lower().strip()
        
        if record_video_choice == 'y':
            duration = 10
            framerate = 60
            frames = []
            
            print(f"\\n🎥 Recording a {duration}-second video at {framerate} FPS...")
            
            # Reset simulation for recording
            mujoco.mj_resetData(model, data)
            
            # Simulate and capture frames
            renderer = mujoco.Renderer(model)
            while data.time < duration:
                mujoco.mj_step(model, data)
                if len(frames) < data.time * framerate:
                    renderer.update_scene(data)
                    pixels = renderer.render()
                    frames.append(pixels)
            
            # Save the video
            video_path = os.path.join(project_root, "simulation_video.mp4")
            with imageio.get_writer(video_path, fps=framerate) as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            print(f"Video saved successfully to: {video_path}")
            return # Exit after recording

        # --- Interactive Viewer Logic ---
        print("\\nLaunching interactive viewer...")
        mujoco.viewer.launch(model, data)
        
    except Exception as e:
        print(f"\\nAn error occurred: {e}")
        print("\\nTroubleshooting:")
        print("  1. Make sure 'imageio' is installed: pip install imageio")
        print("  2. If on macOS, run with 'mjpython'")
        return

if __name__ == "__main__":
    main()