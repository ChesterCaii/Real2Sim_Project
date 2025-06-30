#!/usr/bin/env python3
"""
Real2Sim Pipeline Entry Point
Simple launcher for all pipeline phases
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Real-to-Simulation Pipeline")
    parser.add_argument('phase', choices=['1', '2', '3a', '3b', '4', 'test'], 
                       help='Pipeline phase to run')
    parser.add_argument('--live', action='store_true', 
                       help='Use live camera for Phase 4')
    
    args = parser.parse_args()
    
    # Add src to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print("🚀 Real-to-Simulation Pipeline")
    print("="*50)
    
    if args.phase == '1':
        print("Phase 1: 3D Reconstruction")
        from src.reconstruction.reconstruct_mesh import main as phase1_main
        phase1_main()
        
    elif args.phase == '2':
        print("Phase 2: MuJoCo Simulation")
        from src.simulation.run_real2sim import main as phase2_main
        phase2_main()
        
    elif args.phase == '3a':
        print("Phase 3A: Robot Control Demo")
        from src.control.robot_control_demo import main as phase3a_main
        phase3a_main()
        
    elif args.phase == '3b':
        print("Phase 3B: Multi-Object Reconstruction")
        from src.reconstruction.reconstruct_multi_objects import main as phase3b_main
        phase3b_main()
        
    elif args.phase == '4':
        if args.live:
            print("Phase 4: Live Simulation Bridge")
            from src.live.live_simulation_bridge import main as phase4_main
            phase4_main()
        else:
            print("Phase 4: Camera Integration")
            from src.live.camera_integration import main as phase4_main
            phase4_main()
            
    elif args.phase == 'test':
        print("Running Tests")
        from tests.test_camera_integration import main as test_main
        test_main()

if __name__ == "__main__":
    main() 