#!/usr/bin/env python3
"""
Robot Control Demo - Phase 3A
Demonstrates basic robot arm control and manipulation of reconstructed objects
"""

import mujoco
import numpy as np
import time
import math

class RobotController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Get actuator and joint indices
        self.arm_actuators = [f"actuator{i}" for i in range(1, 8)]  # actuator1-7
        self.gripper_actuator = "actuator8"
        
        # Get joint indices for easier access
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.finger_joints = ["finger_joint1", "finger_joint2"]
        
        # Control parameters
        self.home_position = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])  # Safe home pose
        self.reach_position = np.array([0.5, -0.5, 0.3, -1.8, 0.1, 1.2, 0])     # Reach toward bunny
        
        print("🤖 Robot Controller Initialized")
        print(f"   • Arm actuators: {len(self.arm_actuators)}")
        print(f"   • Gripper actuator: {self.gripper_actuator}")
        
    def get_joint_positions(self):
        """Get current joint positions for the 7 DOF arm"""
        joint_positions = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                joint_positions.append(self.data.qpos[joint_id])
        return np.array(joint_positions)
    
    def set_arm_position(self, target_positions):
        """Set target positions for the 7 DOF arm"""
        for i, actuator_name in enumerate(self.arm_actuators):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0 and i < len(target_positions):
                self.data.ctrl[actuator_id] = target_positions[i]
    
    def set_gripper_position(self, grip_value):
        """
        Control gripper opening/closing
        grip_value: 0 = fully open, 255 = fully closed
        """
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_actuator)
        if actuator_id >= 0:
            self.data.ctrl[actuator_id] = np.clip(grip_value, 0, 255)
    
    def get_end_effector_position(self):
        """Get the current position of the end effector (hand)"""
        hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if hand_body_id >= 0:
            return self.data.xpos[hand_body_id].copy()
        return np.array([0, 0, 0])
    
    def get_bunny_position(self):
        """Get the current position of the bunny"""
        bunny_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bunny")
        if bunny_body_id >= 0:
            return self.data.xpos[bunny_body_id].copy()
        return np.array([0.5, 0.4, 0.12])  # Default bunny position
    
    def smooth_motion(self, start_pos, end_pos, duration_steps, current_step):
        """Generate smooth interpolated motion between two positions"""
        t = min(current_step / duration_steps, 1.0)
        # Use smooth sigmoid-like interpolation
        smooth_t = 3*t**2 - 2*t**3  # Smooth step function
        return start_pos + smooth_t * (end_pos - start_pos)
    
    def demonstrate_movements(self, viewer):
        """Run a demonstration of robot movements"""
        print("\n🎬 Starting Robot Control Demonstration")
        
        step_count = 0
        demo_phase = 0
        phase_start_step = 0
        
        # Demo phases
        phases = [
            {"name": "Home Position", "duration": 100, "target": self.home_position, "gripper": 100},
            {"name": "Open Gripper", "duration": 50, "target": self.home_position, "gripper": 0},
            {"name": "Reach Toward Bunny", "duration": 200, "target": self.reach_position, "gripper": 0},
            {"name": "Close Gripper", "duration": 50, "target": self.reach_position, "gripper": 200},
            {"name": "Lift Object", "duration": 100, "target": self.reach_position + np.array([0,0,0,0,0,0,0.5]), "gripper": 200},
            {"name": "Return Home", "duration": 200, "target": self.home_position, "gripper": 200},
            {"name": "Release Object", "duration": 50, "target": self.home_position, "gripper": 0},
        ]
        
        current_arm_pos = self.get_joint_positions()
        target_arm_pos = current_arm_pos
        
        while viewer.is_running():
            step_start_time = time.time()
            
            # Check if we need to move to next phase
            phase_step = step_count - phase_start_step
            if demo_phase < len(phases) and phase_step >= phases[demo_phase]["duration"]:
                demo_phase += 1
                phase_start_step = step_count
                phase_step = 0
                if demo_phase < len(phases):
                    print(f"📍 Phase {demo_phase + 1}: {phases[demo_phase]['name']}")
                    current_arm_pos = self.get_joint_positions()
            
            # Reset demonstration after all phases complete
            if demo_phase >= len(phases):
                demo_phase = 0
                phase_start_step = step_count
                phase_step = 0
                current_arm_pos = self.get_joint_positions()
                print("\n🔄 Restarting demonstration...")
            
            # Execute current phase
            if demo_phase < len(phases):
                current_phase = phases[demo_phase]
                
                # Smooth arm movement
                target_arm_pos = self.smooth_motion(
                    current_arm_pos, 
                    current_phase["target"], 
                    current_phase["duration"], 
                    phase_step
                )
                
                # Set arm position
                self.set_arm_position(target_arm_pos)
                
                # Set gripper position
                self.set_gripper_position(current_phase["gripper"])
            
            # Step the simulation
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            
            # Display status every 100 steps
            if step_count % 100 == 0:
                ee_pos = self.get_end_effector_position()
                bunny_pos = self.get_bunny_position()
                distance = np.linalg.norm(ee_pos - bunny_pos)
                
                current_joints = self.get_joint_positions()
                print(f"Step {step_count:4d} | Phase: {phases[demo_phase]['name']:20s} | "
                      f"EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}] | "
                      f"Bunny Distance: {distance:.3f}m")
            
            # Maintain real-time
            elapsed = time.time() - step_start_time
            if elapsed < 0.002:  # 500 Hz
                time.sleep(0.002 - elapsed)
            
            step_count += 1

def main():
    """Main robot control demonstration"""
    print("============================================================")
    print("🤖 ROBOT CONTROL DEMO - PHASE 3A")
    print("============================================================")
    
    try:
        # Load the model
        print("🔄 Loading robot and bunny scene...")
        model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml")
        data = mujoco.MjData(model)
        
        print(f"✅ Scene loaded successfully!")
        print(f"   📊 Bodies: {model.nbody}")
        print(f"   🔧 DOFs: {model.nv}")
        print(f"   🤖 Actuators: {model.nu}")
        
        # Create robot controller
        controller = RobotController(model, data)
        
        # Launch viewer with control loop
        print("\n🚀 Launching interactive control demo...")
        print("🎮 Controls:")
        print("   • Space: Pause/unpause physics")
        print("   • Mouse: Rotate camera")
        print("   • Ctrl+C: Exit demo")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Initialize robot position
            controller.set_arm_position(controller.home_position)
            controller.set_gripper_position(100)  # Partially closed
            
            # Start the demonstration
            controller.demonstrate_movements(viewer)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run with: mjpython robot_control_demo.py")

if __name__ == "__main__":
    main() 