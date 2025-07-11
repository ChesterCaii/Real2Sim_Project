#!/usr/bin/env python3
"""
Real2Sim Pipeline Launcher
Easy access to all pipeline phases
"""

import os
import sys
import subprocess

def print_banner():
    """Print the pipeline banner"""
    print("Real2Sim Pipeline - Complete System")
    print("=" * 50)
    print("Transform real-world point clouds into interactive robot simulations")
    print()

def show_phases():
    """Show available pipeline phases"""
    phases = {
        "1": {
            "name": "3D Reconstruction",
            "description": "Point cloud → 3D mesh conversion",
            "file": "src/reconstruction/reconstruct_mesh.py",
            "status": " Core Feature"
        },
        "2": {
            "name": "MuJoCo Simulation",
            "description": "3D mesh → Robot simulation environment",
            "file": "src/simulation/run_real2sim.py", 
            "status": " Core Feature"
        },
        "3a": {
            "name": "Robot Control",
            "description": "Autonomous robot manipulation demo",
            "file": "src/control/robot_control_demo.py",
            "status": " Core Feature"
        },
        "3b": {
            "name": "Multi-Object Reconstruction",
            "description": "Multiple objects → Complex scenes",
            "file": "src/reconstruction/reconstruct_multi_objects.py",
            "status": " Advanced Feature"
        },
        "4": {
            "name": "Live Camera Integration", 
            "description": "Real-time camera → Dynamic simulation",
            "file": "src/live/camera_integration.py",
            "status": " Advanced Feature"
        },
        "5": {
            "name": "Advanced Integration",
            "description": "YOLO detection + Intelligent grasp planning",
            "file": "src/live/phase5_integration_demo.py",
            "status": " Latest Feature"
        },
        "6": {
            "name": "RL Training Setup",
            "description": "Setup MuJoCo Playground for RL training",
            "file": "setup_mujoco_playground.py",
            "status": " RL Extension"
        },
        "7": {
            "name": "RL Training", 
            "description": "Train pick-and-place policies with parallel environments",
            "file": "src/rl_training/train_real2sim_pickup.py",
            "status": " RL Extension"
        }
    }
    
    print("📋 Available Phases:")
    print()
    for phase_id, phase_info in phases.items():
        print(f"Phase {phase_id}: {phase_info['name']}")
        print(f"   {phase_info['description']}")
                  print(f"   File: {phase_info['file']}")
        print(f"   {phase_info['status']}")
        print()

def run_phase(phase_id: str):
    """Run a specific pipeline phase"""
    phase_commands = {
        "1": ("python", "src/reconstruction/reconstruct_mesh.py"),
        "2": ("mjpython", "src/simulation/run_real2sim.py"),
        "3a": ("mjpython", "src/control/robot_control_demo.py"), 
        "3b": ("python", "src/reconstruction/reconstruct_multi_objects.py"),
        "4": ("python", "src/live/camera_integration.py"),
        "5": ("python", "src/live/phase5_integration_demo.py"),
        "6": ("python", "setup_mujoco_playground.py"),
        "7": ("python", "src/rl_training/train_real2sim_pickup.py")
    }
    
    if phase_id not in phase_commands:
        print(f" Unknown phase: {phase_id}")
        print("Available phases: " + ", ".join(phase_commands.keys()))
        return False
    
    python_cmd, script_path = phase_commands[phase_id]
    
    if not os.path.exists(script_path):
        print(f" Script not found: {script_path}")
        return False
    
    print(f" Running Phase {phase_id}...")
    print(f" Script: {script_path}")
    print(f" Command: {python_cmd}")
    print("=" * 50)
    
    try:
        # Run the script with the appropriate Python command
        result = subprocess.run([python_cmd, script_path], 
                              cwd=os.getcwd(),
                              capture_output=False)
        
        if result.returncode == 0:
            print(f" Phase {phase_id} completed successfully!")
        else:
            print(f" Phase {phase_id} failed with exit code {result.returncode}")
            
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print(f"\nPhase {phase_id} interrupted by user")
        return False
    except Exception as e:
        print(f" Error running Phase {phase_id}: {e}")
        return False

def show_system_info():
    """Show system information and requirements"""
    print("💻 System Information:")
    print("=" * 30)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Check key dependencies with correct import names
    dependencies = [
        ("numpy", "Core computation", "numpy"),
        ("open3d", "3D processing", "open3d"), 
        ("opencv-python", "Computer vision", "cv2"),
        ("mujoco", "Physics simulation", "mujoco"),
        ("ultralytics", "YOLO object detection", "ultralytics"),
        ("scipy", "Scientific computing", "scipy")
    ]
    
    print("📦 Key Dependencies:")
    for package, description, import_name in dependencies:
        try:
            __import__(import_name)
            status = " Installed"
        except ImportError:
            status = " Missing"
        print(f"   {package}: {description} - {status}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check if phase specified as command line argument
    if len(sys.argv) > 1:
        phase_id = sys.argv[1].lower()
        
        # Handle special commands
        if phase_id in ['help', '-h', '--help']:
            show_phases()
            return
        elif phase_id in ['info', 'system']:
            show_system_info()
            return
        
        # Run specific phase
        success = run_phase(phase_id)
        sys.exit(0 if success else 1)
    
    # Interactive mode
    while True:
        print("\nInteractive Mode:")
        print("Enter phase number (1, 2, 3a, 3b, 4, 5)")
        print("Or type: 'list' (show phases), 'info' (system info), 'quit' (exit)")
        
        choice = input("\n👉 Your choice: ").strip().lower()
        
        if choice in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif choice in ['list', 'phases', 'l']:
            show_phases()
        elif choice in ['info', 'system', 'i']:
            show_system_info()
        elif choice in ['1', '2', '3a', '3b', '4', '5']:
            success = run_phase(choice)
            if success:
                print(f"\n Phase {choice} completed! Ready for next phase.")
            else:
                print(f"\n  Phase {choice} had issues. Check output above.")
        else:
            print(f" Unknown command: {choice}")
            print("Available: 1, 2, 3a, 3b, 4, 5, list, info, quit")

if __name__ == "__main__":
    main() 