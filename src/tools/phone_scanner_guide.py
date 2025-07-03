#!/usr/bin/env python3
"""
Phone Scanner Integration Guide
Step-by-step guide for scanning objects with phones and importing to Real2Sim
"""

import os
import subprocess
import shutil

class PhoneScannerGuide:
    def __init__(self):
        self.scan_dir = "phone_scans"
        self.assets_dir = "mujoco_menagerie/franka_emika_panda/assets"
        os.makedirs(self.scan_dir, exist_ok=True)
        
    def show_scanning_options(self):
        """Show available phone scanning methods"""
        print("Phone 3D Scanning Options")
        print("=" * 40)
        print()
        
        print("iPhone (with LiDAR - iPhone 12 Pro and newer):")
        print("  - 3D Scanner App (free)")
        print("  - Scaniverse (free)")
        print("  - Polycam (free tier)")
        print()
        
        print("iPhone (without LiDAR - any iPhone):")
        print("  - Photogrammetry apps:")
        print("    * RealityCapture")
        print("    * Trnio")
        print("    * WIDAR")
        print()
        
        print("Android:")
        print("  - 3D Scanner App")
        print("  - Qlone")
        print("  - Sony 3D Creator")
        print("  - Canvas: Pocket 3D Room Scanner")
        print()
        
        print("Web-based (any phone with camera):")
        print("  - Meshroom (free)")
        print("  - AliceVision")
        print("  - COLMAP")
        
    def scanning_tips(self):
        """Provide scanning tips for best results"""
        print("\nScanning Tips for Best Results:")
        print("-" * 40)
        print("LIGHTING:")
        print("  ✓ Use bright, even lighting")
        print("  ✓ Avoid shadows and reflections")
        print("  ✓ Natural daylight works best")
        print()
        
        print("OBJECT SETUP:")
        print("  ✓ Place object on contrasting background")
        print("  ✓ Matte surfaces scan better than shiny")
        print("  ✓ Add texture with temporary markers if needed")
        print()
        
        print("SCANNING TECHNIQUE:")
        print("  ✓ Move phone slowly around object")
        print("  ✓ Keep object in center of frame")
        print("  ✓ Capture from multiple heights/angles")
        print("  ✓ Overlap each shot by 70-80%")
        print()
        
        print("EXPORT SETTINGS:")
        print("  ✓ Export as OBJ format (preferred)")
        print("  ✓ Include texture files if available")
        print("  ✓ Use medium resolution (not too high/low)")
        
    def process_phone_scan(self, scan_file_path, object_name):
        """Process a scanned file from phone"""
        print(f"\nProcessing phone scan: {object_name}")
        print("-" * 40)
        
        if not os.path.exists(scan_file_path):
            print(f"Error: File not found: {scan_file_path}")
            return False
        
        # Copy to assets directory
        file_ext = os.path.splitext(scan_file_path)[1].lower()
        target_name = f"{object_name}_scanned{file_ext}"
        target_path = os.path.join(self.assets_dir, target_name)
        
        try:
            shutil.copy2(scan_file_path, target_path)
            print(f"✓ Copied scan to: {target_path}")
            
            # Get file info
            file_size = os.path.getsize(target_path)
            print(f"✓ File size: {file_size} bytes")
            
            if file_ext == '.obj':
                # Count vertices (basic analysis)
                with open(target_path, 'r') as f:
                    lines = f.readlines()
                    vertices = len([l for l in lines if l.startswith('v ')])
                    faces = len([l for l in lines if l.startswith('f ')])
                print(f"✓ Mesh quality: {vertices} vertices, {faces} faces")
            
            return target_name
            
        except Exception as e:
            print(f"Error processing scan: {e}")
            return False
    
    def create_demo_scene(self, scanned_object):
        """Create a demo scene with the scanned object"""
        print(f"\nCreating demo scene with {scanned_object}")
        
        from src.tools.object_swapper import ObjectSwapper
        swapper = ObjectSwapper()
        
        # Backup and swap
        swapper.swap_object(
            object_name="phone_scan",
            mesh_file=scanned_object,
            material="bunny_material",
            position=[0.5, 0.4, 0.1]
        )
        
        print("✓ Demo scene created!")
        print("✓ Ready to test with robot")
        
    def demo_instructions(self):
        """Instructions for recording the demo"""
        print("\nDemo Recording Instructions")
        print("=" * 40)
        print()
        
        print("BEFORE RECORDING:")
        print("1. Test your scanned object first")
        print("2. Make sure robot movement looks good")
        print("3. Have good lighting for video")
        print("4. Close unnecessary applications")
        print()
        
        print("DEMO SEQUENCE (suggested 2-3 minute video):")
        print("1. Show original object being scanned (15s)")
        print("2. Show point cloud/mesh processing (15s)")
        print("3. Show robot simulation with your object (60s)")
        print("4. Show different robot behaviors (60s)")
        print("5. Conclusion/summary (30s)")
        print()
        
        print("RECORDING TOOLS:")
        print("- macOS: QuickTime Player (built-in screen recording)")
        print("- iPhone: Built-in screen recording")
        print("- Professional: OBS Studio (free)")
        print()
        
        print("WHAT TO HIGHLIGHT:")
        print("✓ Real object → 3D scan → Robot simulation")
        print("✓ Intelligent robot behaviors")
        print("✓ Physics-based interaction")
        print("✓ Multiple manipulation strategies")

def main():
    guide = PhoneScannerGuide()
    
    print("Real2Sim Phone Scanner Integration")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Show phone scanning apps/methods")
        print("2. Show scanning tips for best results")
        print("3. Process a phone scan (have OBJ file ready)")
        print("4. Show demo recording instructions")
        print("5. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            guide.show_scanning_options()
        elif choice == "2":
            guide.scanning_tips()
        elif choice == "3":
            scan_path = input("Enter path to your scanned OBJ file: ").strip()
            obj_name = input("Enter name for this object: ").strip()
            
            result = guide.process_phone_scan(scan_path, obj_name)
            if result:
                print(f"\n✓ Success! Object ready for Real2Sim")
                print(f"✓ To test: python run_pipeline.py 3a")
                
                use_it = input("Create demo scene now? (y/n): ").strip().lower()
                if use_it == 'y':
                    guide.create_demo_scene(result)
        elif choice == "4":
            guide.demo_instructions()
        elif choice == "5":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 