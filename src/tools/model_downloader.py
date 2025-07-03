#!/usr/bin/env python3
"""
Model Downloader for Real2Sim Project
Download popular 3D models for robotics simulation
"""

import os
import requests
import zipfile
import tempfile
from pathlib import Path

class ModelDownloader:
    def __init__(self):
        self.assets_dir = "mujoco_menagerie/franka_emika_panda/assets"
        self.download_dir = os.path.join(self.assets_dir, "downloaded")
        
        # Create download directory
        os.makedirs(self.download_dir, exist_ok=True)
        
    def download_stanford_models(self):
        """Download Stanford 3D Repository models"""
        models = {
            "teapot": "https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj",
            "dragon": "https://graphics.stanford.edu/data/3Dscanrep/dragon/dragon_vrip.ply"
        }
        
        print("Downloading Stanford models...")
        for name, url in models.items():
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    ext = url.split('.')[-1]
                    filepath = os.path.join(self.download_dir, f"{name}.{ext}")
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"  Downloaded: {name}.{ext}")
                else:
                    print(f"  Failed: {name} (HTTP {response.status_code})")
            except Exception as e:
                print(f"  Error downloading {name}: {e}")
    
    def create_common_objects(self):
        """Create common robotics manipulation objects"""
        print("Creating common manipulation objects...")
        
        # Bottle
        bottle_obj = """# Bottle
v 0.0 0.0 0.0
v 0.03 0.0 0.0
v 0.03 0.03 0.0
v 0.0 0.03 0.0
v 0.0 0.0 0.12
v 0.03 0.0 0.12
v 0.03 0.03 0.12
v 0.0 0.03 0.12
v 0.015 0.015 0.15

f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
f 5 6 9
f 6 7 9
f 7 8 9
f 8 5 9
"""
        
        # Can
        can_obj = """# Can (cylinder)
v 0.0325 0.0 0.0
v 0.023 0.023 0.0
v 0.0 0.0325 0.0
v -0.023 0.023 0.0
v -0.0325 0.0 0.0
v -0.023 -0.023 0.0
v 0.0 -0.0325 0.0
v 0.023 -0.023 0.0
v 0.0325 0.0 0.11
v 0.023 0.023 0.11
v 0.0 0.0325 0.11
v -0.023 0.023 0.11
v -0.0325 0.0 0.11
v -0.023 -0.023 0.11
v 0.0 -0.0325 0.11
v 0.023 -0.023 0.11

# Bottom face
f 1 8 7 6 5 4 3 2
# Top face  
f 9 10 11 12 13 14 15 16
# Side faces
f 1 2 10 9
f 2 3 11 10
f 3 4 12 11
f 4 5 13 12
f 5 6 14 13
f 6 7 15 14
f 7 8 16 15
f 8 1 9 16
"""
        
        # Tool (screwdriver-like)
        tool_obj = """# Tool
v 0.0 0.0 0.0
v 0.005 0.0 0.0
v 0.005 0.005 0.0
v 0.0 0.005 0.0
v 0.0 0.0 0.08
v 0.005 0.0 0.08
v 0.005 0.005 0.08
v 0.0 0.005 0.08
v 0.0 0.0 0.15
v 0.001 0.0 0.15
v 0.001 0.001 0.15
v 0.0 0.001 0.15

# Handle
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
# Tip
f 5 6 10 9
f 6 7 11 10
f 7 8 12 11
f 8 5 9 12
f 9 10 11 12
"""
        
        # Write objects
        objects = {
            "bottle.obj": bottle_obj,
            "can.obj": can_obj,
            "tool.obj": tool_obj
        }
        
        for filename, content in objects.items():
            filepath = os.path.join(self.assets_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  Created: {filename}")
    
    def download_sample_models(self):
        """Download sample models from various sources"""
        # Note: In practice, you'd implement specific downloaders for each source
        # For now, we'll create instructions
        
        sources = {
            "Free Sources": [
                "OpenGameArt.org - CC-BY 3.0 licensed models",
                "C4DDownload.com - Free OBJ/STL models", 
                "TurboSquid Free - Basic models",
                "BlenderKit - Some free models"
            ],
            "Model Repositories": [
                "YCB Object Set - Standard manipulation objects",
                "ModelNet - Large-scale 3D dataset",
                "ShapeNet - Research 3D models"
            ],
            "Direct Downloads": [
                "https://www.prinmath.com/csci5229/OBJ/ - Simple OBJ models",
                "http://graphics.cs.williams.edu/data/meshes.xml - Academic models"
            ]
        }
        
        print("\nBest 3D Model Sources:")
        print("=" * 40)
        for category, links in sources.items():
            print(f"\n{category}:")
            for link in links:
                print(f"  - {link}")
    
    def convert_ply_to_obj(self, ply_file, obj_file):
        """Convert PLY to OBJ format (basic implementation)"""
        try:
            # This would need a proper PLY parser
            print(f"Converting {ply_file} to {obj_file}")
            print("Note: PLY conversion requires specialized library")
            print("Recommended: Use Blender or MeshLab for conversion")
        except Exception as e:
            print(f"Conversion failed: {e}")
    
    def list_downloaded_models(self):
        """List all available models"""
        print(f"\nAvailable models in {self.assets_dir}:")
        print("-" * 40)
        
        for file in os.listdir(self.assets_dir):
            if file.endswith(('.obj', '.stl', '.ply')):
                filepath = os.path.join(self.assets_dir, file)
                size = os.path.getsize(filepath)
                print(f"  {file} ({size} bytes)")

def main():
    downloader = ModelDownloader()
    
    print("Real2Sim Model Downloader")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Create common manipulation objects")
        print("2. Show model sources & download links")
        print("3. List available models")
        print("4. Download Stanford models (if available)")
        print("5. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            downloader.create_common_objects()
        elif choice == "2":
            downloader.download_sample_models()
        elif choice == "3":
            downloader.list_downloaded_models()
        elif choice == "4":
            downloader.download_stanford_models()
        elif choice == "5":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 