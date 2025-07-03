#!/usr/bin/env python3
"""
Object Swapper Tool
Easily replace objects in the Real2Sim simulation
"""

import os
import shutil
import xml.etree.ElementTree as ET

class ObjectSwapper:
    def __init__(self, scene_path="mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml"):
        self.scene_path = scene_path
        self.assets_dir = "mujoco_menagerie/franka_emika_panda/assets"
        
    def list_available_objects(self):
        """List objects that can be swapped in"""
        objects = {
            "bunny": "bunny_final.obj",
            "cube": "cube.obj", 
            "sphere": "sphere.obj",
            "cylinder": "cylinder.obj",
            "bottle": "bottle.obj"
        }
        
        print("Available objects:")
        for name, file in objects.items():
            path = os.path.join(self.assets_dir, file)
            status = "EXISTS" if os.path.exists(path) else "MISSING"
            print(f"  {name}: {file} [{status}]")
        
        return objects
    
    def create_basic_objects(self):
        """Create basic geometric objects for testing"""
        print("Creating basic test objects...")
        
        # Simple cube OBJ
        cube_obj = """# Cube
v -0.1 -0.1 -0.1
v  0.1 -0.1 -0.1
v  0.1  0.1 -0.1
v -0.1  0.1 -0.1
v -0.1 -0.1  0.1
v  0.1 -0.1  0.1
v  0.1  0.1  0.1
v -0.1  0.1  0.1

f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""
        
        # Simple sphere OBJ (icosphere approximation)
        sphere_obj = """# Sphere (simplified)
v 0.0 0.0 0.1
v 0.0894 0.0 0.0447
v 0.0276 0.085 0.0447
v -0.0724 0.0526 0.0447
v -0.0724 -0.0526 0.0447
v 0.0276 -0.085 0.0447
v 0.0724 0.0526 -0.0447
v -0.0276 0.085 -0.0447
v -0.0894 0.0 -0.0447
v -0.0276 -0.085 -0.0447
v 0.0724 -0.0526 -0.0447
v 0.0 0.0 -0.1

f 1 2 3
f 1 3 4
f 1 4 5
f 1 5 6
f 1 6 2
f 2 6 11
f 6 5 10
f 5 4 9
f 4 3 8
f 3 2 7
f 7 2 11
f 8 3 7
f 9 4 8
f 10 5 9
f 11 6 10
f 12 7 11
f 12 8 7
f 12 9 8
f 12 10 9
f 12 11 10
"""
        
        # Write objects
        with open(os.path.join(self.assets_dir, "cube.obj"), "w") as f:
            f.write(cube_obj)
        
        with open(os.path.join(self.assets_dir, "sphere.obj"), "w") as f:
            f.write(sphere_obj)
        
        print("Created cube.obj and sphere.obj")
    
    def swap_object(self, object_name, mesh_file, material="bunny_material", 
                   position=[0.5, 0.4, 0.15], size=[1.0, 1.0, 1.0]):
        """Swap the current object with a new one"""
        
        # Backup original
        backup_path = self.scene_path + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(self.scene_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Parse XML
        tree = ET.parse(self.scene_path)
        root = tree.getroot()
        
        # Update mesh in assets
        for mesh in root.findall(".//mesh[@name='bunny_mesh']"):
            mesh.set('file', mesh_file)
            print(f"Updated mesh file to: {mesh_file}")
        
        # Update object body
        for body in root.findall(".//body[@name='bunny']"):
            body.set('pos', f"{position[0]} {position[1]} {position[2]}")
            
            # Update geom
            for geom in body.findall("geom"):
                geom.set('material', material)
                if size != [1.0, 1.0, 1.0]:
                    geom.set('size', f"{size[0]} {size[1]} {size[2]}")
                print(f"Updated object: {object_name}")
        
        # Save
        tree.write(self.scene_path, xml_declaration=True, encoding='utf-8')
        print(f"Scene updated! New object: {object_name}")
        
    def restore_original(self):
        """Restore from backup"""
        backup_path = self.scene_path + ".backup"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, self.scene_path)
            print("Restored original scene")
        else:
            print("No backup found")

def main():
    swapper = ObjectSwapper()
    
    print("Real2Sim Object Swapper")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. List available objects")
        print("2. Create basic test objects") 
        print("3. Swap to cube")
        print("4. Swap to sphere")
        print("5. Swap to bunny (original)")
        print("6. Restore original scene")
        print("7. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            swapper.list_available_objects()
        elif choice == "2":
            swapper.create_basic_objects()
        elif choice == "3":
            swapper.swap_object("cube", "cube.obj", position=[0.5, 0.4, 0.1], size=[0.05, 0.05, 0.05])
        elif choice == "4":
            swapper.swap_object("sphere", "sphere.obj", position=[0.5, 0.4, 0.1], size=[0.05, 0.05, 0.05])
        elif choice == "5":
            swapper.swap_object("bunny", "bunny_final.obj")
        elif choice == "6":
            swapper.restore_original()
        elif choice == "7":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 