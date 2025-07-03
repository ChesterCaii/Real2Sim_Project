#!/usr/bin/env python3
"""
Robot Control Demo - Phase 3A
Demonstrates basic robot arm control and manipulation of reconstructed objects
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import math

class IntelligentRobotControl:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.phase = 0
        self.phase_timer = 0
        self.behavior_mode = 0  # 0=exploration, 1=inspection, 2=manipulation, 3=demonstration
        self.mode_timer = 0
        self.target_pos = None
        
        # Robot joint names for the Panda arm (corrected names)
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]
        
        # Robot actuator names for control
        self.actuator_names = [
            'actuator1', 'actuator2', 'actuator3', 'actuator4',
            'actuator5', 'actuator6', 'actuator7', 'actuator8'  # actuator8 controls gripper
        ]
        
        # Get joint IDs for monitoring
        self.joint_ids = []
        for name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_ids.append(joint_id)
            except:
                print(f"Warning: Joint {name} not found")
        
        # Get actuator IDs for control
        self.actuator_ids = []
        for name in self.actuator_names:
            try:
                actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.actuator_ids.append(actuator_id)
            except:
                print(f"Warning: Actuator {name} not found")
        
        print(f"🤖 Intelligent Robot Control Initialized")
        print(f"📡 Found {len(self.joint_ids)} joints")
        print(f"🎮 Found {len(self.actuator_ids)} actuators")
        print(f"🎯 Behavior Modes: Exploration → Inspection → Manipulation → Demo")
        
    def get_bunny_position(self):
        """Get the bunny object position"""
        try:
            bunny_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bunny')
            return self.data.xpos[bunny_id].copy()
        except:
            return np.array([0.5, 0.0, 0.1])  # Default position
    
    def smooth_interpolate(self, start, end, t):
        """Smooth interpolation using cosine"""
        smooth_t = 0.5 * (1 - np.cos(np.pi * t))
        return start + (end - start) * smooth_t
    
    def exploration_behavior(self):
        """Robot explores the workspace intelligently"""
        cycle_time = 8.0  # 8 seconds per exploration cycle
        t = (self.phase_timer % cycle_time) / cycle_time
        
        if t < 0.25:  # Look around phase
            target = np.array([0.3, 0.2, 0.4, -1.5, 0.0, 1.8, 0.8])
        elif t < 0.5:  # Reach toward bunny
            target = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.8])
        elif t < 0.75:  # Side inspection
            target = np.array([-0.3, 0.1, 0.3, -1.8, 0.0, 1.9, 0.4])
        else:  # Return to observation
            target = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.79])
            
        # Add open gripper
        return np.append(target, 0)
    
    def inspection_behavior(self):
        """Robot carefully inspects the bunny from different angles"""
        cycle_time = 10.0
        t = (self.phase_timer % cycle_time) / cycle_time
        
        bunny_pos = self.get_bunny_position()
        
        if t < 0.2:  # Approach from front
            target = np.array([0.2, -0.4, 0.1, -2.0, 0.1, 1.6, 0.8])
        elif t < 0.4:  # Close inspection
            target = np.array([0.1, -0.6, 0.0, -2.3, 0.0, 1.7, 0.8])
        elif t < 0.6:  # Side view
            target = np.array([-0.4, -0.2, 0.2, -1.8, 0.3, 1.5, 0.6])
        elif t < 0.8:  # Top view
            target = np.array([0.0, 0.0, 0.4, -1.2, 0.0, 1.2, 0.8])
        else:  # Retreat and observe
            target = np.array([0.3, 0.3, 0.2, -1.4, 0.0, 1.7, 0.8])
            
        # Add open gripper
        return np.append(target, 0)
    
    def manipulation_behavior(self):
        """Robot demonstrates manipulation movements with actual grasping"""
        cycle_time = 12.0
        t = (self.phase_timer % cycle_time) / cycle_time
        
        if t < 0.15:  # Home position - gripper open
            arm_target = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.79])
            gripper_target = 0  # Open gripper
        elif t < 0.3:  # Pre-grasp approach - gripper open
            arm_target = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
            gripper_target = 0  # Open gripper
        elif t < 0.45:  # Grasp position - gripper open
            arm_target = np.array([0.0, -0.7, -0.1, -2.3, 0.0, 1.6, 0.8])
            gripper_target = 0  # Open gripper
        elif t < 0.5:  # Close gripper - GRASP!
            arm_target = np.array([0.0, -0.7, -0.1, -2.3, 0.0, 1.6, 0.8])
            gripper_target = 200  # Close gripper
        elif t < 0.6:  # Lift with object
            arm_target = np.array([0.0, -0.5, 0.1, -1.8, 0.0, 1.3, 0.8])
            gripper_target = 200  # Keep gripper closed
        elif t < 0.75:  # Transport with object
            arm_target = np.array([0.4, -0.2, 0.2, -1.5, 0.0, 1.3, 0.8])
            gripper_target = 200  # Keep gripper closed
        elif t < 0.85:  # Place position
            arm_target = np.array([0.4, -0.4, 0.0, -1.8, 0.0, 1.4, 0.8])
            gripper_target = 200  # Still closed
        elif t < 0.9:  # Release object
            arm_target = np.array([0.4, -0.4, 0.0, -1.8, 0.0, 1.4, 0.8])
            gripper_target = 0  # Open gripper - RELEASE!
        else:  # Return home
            arm_target = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.79])
            gripper_target = 0  # Open gripper
            
        # Combine arm and gripper commands
        return np.append(arm_target, gripper_target)
    
    def demonstration_behavior(self):
        """Smooth, presentation-ready movements"""
        cycle_time = 15.0
        t = (self.phase_timer % cycle_time) / cycle_time
        
        # Smooth sinusoidal movements
        base_pos = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.79])
        
        # Add gentle oscillations to different joints
        variation = np.array([
            0.3 * np.sin(2 * np.pi * t),           # Joint 1
            -0.2 * np.cos(2 * np.pi * t * 1.3),    # Joint 2
            0.15 * np.sin(2 * np.pi * t * 0.7),    # Joint 3
            -0.3 * np.cos(2 * np.pi * t * 0.9),    # Joint 4
            0.2 * np.sin(2 * np.pi * t * 1.1),     # Joint 5
            0.25 * np.cos(2 * np.pi * t * 0.8),    # Joint 6
            0.1 * np.sin(2 * np.pi * t * 1.5)      # Joint 7
        ])
        
        arm_target = base_pos + variation
        # Add open gripper
        return np.append(arm_target, 0)
    
    def update_behavior_mode(self):
        """Switch between different behavior modes"""
        mode_duration = 20.0  # 20 seconds per mode
        
        if self.mode_timer > mode_duration:
            self.behavior_mode = (self.behavior_mode + 1) % 4
            self.mode_timer = 0
            self.phase_timer = 0
            
            mode_names = ["🔍 Exploration", "🔬 Inspection", "🤖 Manipulation", "✨ Demonstration"]
            print(f"\n🎯 Switching to: {mode_names[self.behavior_mode]} Mode")
    
    def control_step(self, dt):
        """Main control loop - called every simulation step"""
        self.phase_timer += dt
        self.mode_timer += dt
        
        # Update behavior mode
        self.update_behavior_mode()
        
        # Get target joint angles based on current behavior
        if self.behavior_mode == 0:
            target_angles = self.exploration_behavior()
        elif self.behavior_mode == 1:
            target_angles = self.inspection_behavior()
        elif self.behavior_mode == 2:
            target_angles = self.manipulation_behavior()
        else:  # self.behavior_mode == 3
            target_angles = self.demonstration_behavior()
        
        # Apply target angles to actuators (corrected control method)
        for i, actuator_id in enumerate(self.actuator_ids):
            if i < len(target_angles):
                self.data.ctrl[actuator_id] = target_angles[i]

def main():
    print("🚀 Starting Real2Sim Robot Demonstration")
    print("=" * 50)
    
    # Load the MuJoCo model
    model_path = "mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize robot controller
    controller = IntelligentRobotControl(model, data)
    
    print(f"\n🎮 Controls:")
    print(f"   - Space: Pause/Resume")
    print(f"   - ESC: Exit")
    print(f"   - Mouse: Rotate/Pan view")
    print(f"\n🤖 The robot will cycle through 4 behavior modes:")
    print(f"   1. Exploration (20s) - Workspace exploration")
    print(f"   2. Inspection (20s) - Detailed object examination") 
    print(f"   3. Manipulation (20s) - Pick and place demonstration")
    print(f"   4. Presentation (20s) - Smooth exhibition movements")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # Physics simulation step
            mujoco.mj_step(model, data)
            
            # Robot control update
            dt = time.time() - step_start
            controller.control_step(dt)
            
            # Update viewer
            viewer.sync()
            
            # Control simulation speed (60 FPS)
            elapsed = time.time() - step_start
            sleep_time = max(0, 1.0/60.0 - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main() 