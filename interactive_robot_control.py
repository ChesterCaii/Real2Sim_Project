#!/usr/bin/env python3
"""
Interactive Robot Control - Phase 3A
Manual control of robot joints and gripper for testing and experimentation
"""

import mujoco
import numpy as np
import time

class InteractiveRobotController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Joint control parameters
        self.current_joint_targets = np.zeros(7)  # 7 DOF arm
        self.gripper_target = 100  # Gripper position (0-255)
        
        # Control increments
        self.joint_increment = 0.1  # radians
        self.gripper_increment = 20  # gripper units
        
        # Safe joint limits (approximate)
        self.joint_limits = [
            (-2.8, 2.8),   # joint1
            (-1.7, 1.7),   # joint2  
            (-2.8, 2.8),   # joint3
            (-3.0, -0.1),  # joint4
            (-2.8, 2.8),   # joint5
            (-0.0, 3.7),   # joint6
            (-2.8, 2.8),   # joint7
        ]
        
        print("🎮 Interactive Robot Controller Ready")
        print("Controls:")
        print("  Joints 1-7: q/a, w/s, e/d, r/f, t/g, y/h, u/j")
        print("  Gripper: o (open), p (close)")
        print("  Special: HOME (reset to home), SPACE (info), ESC (quit)")
        
    def apply_controls(self):
        """Apply current target positions to the robot"""
        # Apply joint positions
        for i in range(7):
            actuator_name = f"actuator{i+1}"
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = self.current_joint_targets[i]
        
        # Apply gripper position
        gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if gripper_actuator_id >= 0:
            self.data.ctrl[gripper_actuator_id] = self.gripper_target
    
    def adjust_joint(self, joint_index, delta):
        """Safely adjust a joint position"""
        if 0 <= joint_index < 7:
            new_value = self.current_joint_targets[joint_index] + delta
            min_limit, max_limit = self.joint_limits[joint_index]
            self.current_joint_targets[joint_index] = np.clip(new_value, min_limit, max_limit)
            print(f"Joint {joint_index+1}: {self.current_joint_targets[joint_index]:.3f} rad")
    
    def adjust_gripper(self, delta):
        """Adjust gripper position"""
        self.gripper_target = np.clip(self.gripper_target + delta, 0, 255)
        print(f"Gripper: {self.gripper_target}")
    
    def reset_to_home(self):
        """Reset robot to home position"""
        self.current_joint_targets = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        self.gripper_target = 100
        print("🏠 Reset to home position")
    
    def get_robot_info(self):
        """Display current robot state"""
        # Get end effector position
        hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if hand_body_id >= 0:
            ee_pos = self.data.xpos[hand_body_id]
        else:
            ee_pos = [0, 0, 0]
        
        # Get bunny position
        bunny_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bunny")
        if bunny_body_id >= 0:
            bunny_pos = self.data.xpos[bunny_body_id]
            distance = np.linalg.norm(ee_pos - bunny_pos)
        else:
            bunny_pos = [0.5, 0.4, 0.12]
            distance = 0
        
        print(f"\n📊 Robot Status:")
        print(f"   End Effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"   Bunny: [{bunny_pos[0]:.3f}, {bunny_pos[1]:.3f}, {bunny_pos[2]:.3f}]")
        print(f"   Distance to Bunny: {distance:.3f}m")
        print(f"   Joint Targets: {[f'{x:.3f}' for x in self.current_joint_targets]}")
        print(f"   Gripper: {self.gripper_target}\n")

def run_interactive_control():
    """Main interactive control loop"""
    print("============================================================")
    print("🎮 INTERACTIVE ROBOT CONTROL - PHASE 3A")
    print("============================================================")
    
    try:
        # Load model
        print("🔄 Loading robot and bunny scene...")
        model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml")
        data = mujoco.MjData(model)
        
        print(f"✅ Scene loaded!")
        
        # Create controller
        controller = InteractiveRobotController(model, data)
        controller.reset_to_home()
        
        print("\n🚀 Starting interactive control...")
        print("Note: This demo runs autonomously. For keyboard control, modify the script.")
        
        # Create a simple demo sequence instead of keyboard control
        demo_commands = [
            # Command format: (joint_index, delta) or ('gripper', delta) or ('info',) or ('home',)
            ('home',),
            ('info',),
            (0, 0.5),  # Move joint 1
            (1, -0.3), # Move joint 2
            (2, 0.2),  # Move joint 3
            ('gripper', -50),  # Open gripper
            (3, 0.5),  # Move joint 4
            (4, 0.3),  # Move joint 5
            (5, -0.5), # Move joint 6
            (6, 0.4),  # Move joint 7
            ('gripper', 100), # Close gripper
            ('info',),
            ('home',),
        ]
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step_count = 0
            command_index = 0
            steps_per_command = 100  # Hold each command for 100 steps
            
            while viewer.is_running() and command_index < len(demo_commands):
                if step_count % steps_per_command == 0:
                    # Execute next command
                    if command_index < len(demo_commands):
                        cmd = demo_commands[command_index]
                        
                        if cmd[0] == 'home':
                            controller.reset_to_home()
                        elif cmd[0] == 'info':
                            controller.get_robot_info()
                        elif cmd[0] == 'gripper':
                            controller.adjust_gripper(cmd[1])
                        elif isinstance(cmd[0], int):
                            controller.adjust_joint(cmd[0], cmd[1])
                        
                        command_index += 1
                
                # Apply controls and step simulation
                controller.apply_controls()
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # Small delay for real-time viewing
                time.sleep(0.002)
                step_count += 1
            
            # Keep running viewer after demo
            print("\n✅ Demo complete. Viewer will stay open for observation.")
            while viewer.is_running():
                controller.apply_controls()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run with: mjpython interactive_robot_control.py")

if __name__ == "__main__":
    run_interactive_control() 