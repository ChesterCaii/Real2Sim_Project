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
    print("🚀 Real2Sim Pipeline - Complete System")
    print("=" * 50)
    print("Transform real-world point clouds into interactive robot simulations")
    print()

def show_phases():
    """Show available pipeline phases"""
    phases = {
        "1": {
            "name": "3D Reconstruction",
            "description": "Point cloud → 3D mesh conversion",
            "file": "src/reconstruction/point_cloud_processor.py",
            "status": "✅ Core Feature"
        },
        "2": {
            "name": "MuJoCo Simulation",
            "description": "3D mesh → Robot simulation environment",
            "file": "src/simulation/mujoco_scene_generator.py", 
            "status": "✅ Core Feature"
        },
        "3a": {
            "name": "Robot Control",
            "description": "Autonomous robot manipulation demo",
            "file": "src/control/robot_controller.py",
            "status": "✅ Core Feature"
        },
        "3b": {
            "name": "Multi-Object Reconstruction",
            "description": "Multiple objects → Complex scenes",
            "file": "src/reconstruction/multi_object_reconstructor.py",
            "status": "✅ Advanced Feature"
        },
        "4": {
            "name": "Live Camera Integration", 
            "description": "Real-time camera → Dynamic simulation",
            "file": "src/live/camera_integration.py",
            "status": "✅ Advanced Feature"
        },
        "5": {
            "name": "Advanced Integration",
            "description": "YOLO detection + Intelligent grasp planning",
            "file": "src/live/phase5_integration_demo.py",
            "status": "🚀 Latest Feature"
        }
    }
    
    print("📋 Available Phases:")
    print()
    for phase_id, phase_info in phases.items():
        print(f"Phase {phase_id}: {phase_info['name']}")
        print(f"   📝 {phase_info['description']}")
        print(f"   📁 {phase_info['file']}")
        print(f"   {phase_info['status']}")
        print()

def run_phase(phase_id: str):
    """Run a specific pipeline phase"""
    phase_files = {
        "1": "src/reconstruction/point_cloud_processor.py",
        "2": "src/simulation/mujoco_scene_generator.py",
        "3a": "src/control/robot_controller.py", 
        "3b": "src/reconstruction/multi_object_reconstructor.py",
        "4": "src/live/camera_integration.py",
        "5": "src/live/phase5_integration_demo.py"
    }
    
    if phase_id not in phase_files:
        print(f"❌ Unknown phase: {phase_id}")
        print("Available phases: " + ", ".join(phase_files.keys()))
        return False
    
    script_path = phase_files[phase_id]
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"🚀 Running Phase {phase_id}...")
    print(f"📁 Script: {script_path}")
    print("=" * 50)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.getcwd(),
                              capture_output=False)
        
        if result.returncode == 0:
            print(f"✅ Phase {phase_id} completed successfully!")
        else:
            print(f"❌ Phase {phase_id} failed with exit code {result.returncode}")
            
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Phase {phase_id} interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running Phase {phase_id}: {e}")
        return False

def show_system_info():
    """Show system information and requirements"""
    print("💻 System Information:")
    print("=" * 30)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Check key dependencies
    dependencies = [
        ("numpy", "Core computation"),
        ("open3d", "3D processing"), 
        ("opencv-python", "Computer vision"),
        ("mujoco", "Physics simulation"),
        ("ultralytics", "YOLO object detection"),
        ("scipy", "Scientific computing")
    ]
    
    print("📦 Key Dependencies:")
    for package, description in dependencies:
        try:
            __import__(package.replace('-', '_'))
            status = "✅ Installed"
        except ImportError:
            status = "❌ Missing"
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
        print("\n🎮 Interactive Mode:")
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
                print(f"\n🎉 Phase {choice} completed! Ready for next phase.")
            else:
                print(f"\n⚠️  Phase {choice} had issues. Check output above.")
        else:
            print(f"❌ Unknown command: {choice}")
            print("Available: 1, 2, 3a, 3b, 4, 5, list, info, quit")

if __name__ == "__main__":
    main() 