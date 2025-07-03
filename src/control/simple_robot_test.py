#!/usr/bin/env python3
"""
Simple Robot Movement Test
Very obvious movements to verify the robot actually responds to commands
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math

def main():
    print("🤖 Simple Robot Movement Test")
    print("=" * 40)
    
    # Load the MuJoCo model
    model_path = "mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get actuator IDs
    actuator_names = [
        'actuator1', 'actuator2', 'actuator3', 'actuator4',
        'actuator5', 'actuator6', 'actuator7', 'actuator8'
    ]
    
    actuator_ids = []
    for name in actuator_names:
        try:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            actuator_ids.append(actuator_id)
            print(f"Found actuator: {name} (ID: {actuator_id})")
        except:
            print(f"Warning: Actuator {name} not found")
    
    print(f"\n🎮 CONTROLS:")
    print(f"   - SPACEBAR: Start robot movement (simulation must be UNPAUSED)")
    print(f"   - ESC: Exit")
    print(f"   - If robot doesn't move, press SPACEBAR in the viewer!")
    print(f"\n🔥 Robot will make BIG obvious movements every 3 seconds")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        movement_timer = 0
        movement_phase = 0
        
        print(f"\n🚀 Robot should start moving NOW...")
        print(f"If you don't see movement, the simulation might be PAUSED")
        print(f"Press SPACEBAR in the MuJoCo viewer window!")
        
        while viewer.is_running():
            step_start = time.time()
            current_time = time.time() - start_time
            
            # Physics simulation step
            mujoco.mj_step(model, data)
            
            # Very obvious robot movements - BIG changes every 3 seconds
            movement_timer += 1.0/60.0  # Assuming 60 FPS
            
            if movement_timer > 3.0:  # Change every 3 seconds
                movement_timer = 0
                movement_phase = (movement_phase + 1) % 4
                
                if movement_phase == 0:
                    # HOME position
                    targets = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.79, 0]
                    print(f"🏠 HOME position - robot should move to upright")
                elif movement_phase == 1:
                    # BIG swing to the RIGHT
                    targets = [1.5, 0.5, 0.0, -1.0, 0.0, 2.0, 0.0, 0]
                    print(f"➡️  SWING RIGHT - robot should move dramatically to right")
                elif movement_phase == 2:
                    # BIG swing to the LEFT  
                    targets = [-1.5, -0.5, 0.0, -2.0, 0.0, 1.0, 1.57, 100]
                    print(f"⬅️  SWING LEFT - robot should swing to left + close gripper")
                else:
                    # UP high position
                    targets = [0.0, -0.8, 0.0, -0.5, 0.0, 0.8, 0.0, 0]
                    print(f"⬆️  REACH UP - robot should reach up high")
                
                # Apply commands to actuators
                for i, actuator_id in enumerate(actuator_ids):
                    if i < len(targets):
                        data.ctrl[actuator_id] = targets[i]
                        
                print(f"   Set actuator {i}: {targets[i]:.2f}")
            
            # Update viewer
            viewer.sync()
            
            # Control simulation speed (60 FPS)
            elapsed = time.time() - step_start
            sleep_time = max(0, 1.0/60.0 - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main() 