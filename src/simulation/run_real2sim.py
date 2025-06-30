#!/usr/bin/env python3
"""
Real-to-Simulation Pipeline - Final Demo
=========================================

This script loads the combined Franka Panda robot and reconstructed bunny mesh
into a single MuJoCo physics simulation.

Usage:
    mjpython run_real2sim.py

Requirements:
    - MuJoCo Python bindings
    - mujoco_menagerie (Franka robot models)
    - bunny_final.stl (reconstructed mesh)

Author: Real2Sim Project
"""

import mujoco
import mujoco.viewer
import sys
import os
import time

def main():
    print("=" * 60)
    print("🤖 REAL-TO-SIMULATION PIPELINE DEMO 🐰")
    print("=" * 60)
    print()
    
    # Check if running with mjpython (required on macOS)
    if sys.platform == "darwin" and "mjpython" not in sys.executable:
        print("⚠️  On macOS, this script should be run with 'mjpython'")
        print("   Try: mjpython run_real2sim.py")
        print()
    
    # Path to our combined scene
    scene_path = "robot_bunny_scene.xml"
    
    if not os.path.exists(scene_path):
        print("❌ Error: Combined scene file not found!")
        print(f"   Expected: {scene_path}")
        print("   Make sure you've run the setup correctly.")
        return
    
    try:
        print("🔄 Loading combined robot and bunny scene...")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print(f"✅ Successfully loaded scene!")
        print(f"   📊 Bodies: {model.nbody}")
        print(f"   🔧 DOFs: {model.nq}")
        print(f"   🎨 Meshes: {model.nmesh}")
        print(f"   🤖 Actuators: {model.nu}")
        print()
        
        print("🎯 Scene Contents:")
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name != "world":
                icon = "🤖" if "link" in body_name or "hand" in body_name or "finger" in body_name else "🐰"
                print(f"   {icon} {body_name}")
        print()
        
        print("🚀 Launching interactive viewer...")
        print("📖 Controls:")
        print("   • Mouse: Rotate camera")
        print("   • Right-click + drag: Pan camera") 
        print("   • Scroll: Zoom in/out")
        print("   • Space: Pause/unpause physics")
        print("   • Ctrl+R: Reset simulation")
        print()
        print("🎉 SUCCESS! Your Real-to-Simulation pipeline is working!")
        print("   You should see the Franka robot and reconstructed bunny in the viewer.")
        print()
        
        # Try different viewer launch methods
        try:
            print("🔄 Attempting to launch viewer...")
            # Use passive launch with a proper wait loop
            with mujoco.viewer.launch_passive(model, data) as viewer:
                print("✅ Viewer launched! Press Ctrl+C to exit or close the viewer window.")
                # Keep the script running while viewer is open
                while viewer.is_running():
                    time.sleep(0.1)
        except Exception as viewer_error:
            print(f"⚠️  Viewer launch failed: {viewer_error}")
            print("🔄 Trying alternative method...")
            
            # Alternative: try the basic launch
            try:
                mujoco.viewer.launch(model, data)
            except Exception as alt_error:
                print(f"❌ Alternative viewer also failed: {alt_error}")
                print("\n💡 Alternative: Save the scene and view it externally")
                print("   You can:")
                print("   1. Use MuJoCo's standalone viewer: mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml")
                print("   2. Or run: mjpython -m mujoco.viewer mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml")
                return
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Make sure you're using 'mjpython' instead of 'python'")
        print("   2. Check that all mesh files are in the assets directory")
        print("   3. Verify that bunny_final.stl was created successfully")
        return

if __name__ == "__main__":
    main() 