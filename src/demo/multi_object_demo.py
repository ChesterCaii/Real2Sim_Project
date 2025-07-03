#!/usr/bin/env python3
"""
Multi-Object Demo Scene
Shows robot interacting with multiple objects
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import xml.etree.ElementTree as ET
import os

def create_multi_object_scene():
    """Create a scene with multiple objects"""
    
    # Load the base scene
    scene_path = "mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml"
    
    # Parse XML
    tree = ET.parse(scene_path)
    root = tree.getroot()
    
    # Find worldbody
    worldbody = root.find('worldbody')
    
    # Add multiple objects
    objects = [
        {
            'name': 'cube1',
            'mesh': 'cube.obj',
            'pos': '0.4 0.2 0.05',
            'material': 'bunny_gold',
            'size': '0.03 0.03 0.03'
        },
        {
            'name': 'sphere1', 
            'mesh': 'sphere.obj',
            'pos': '0.5 0.1 0.05',
            'material': 'bunny_green',
            'size': '0.03 0.03 0.03'
        },
        {
            'name': 'bottle1',
            'mesh': 'bottle.obj', 
            'pos': '0.6 0.3 0.07',
            'material': 'bunny_material',
            'size': '1.0 1.0 1.0'
        }
    ]
    
    # Add meshes to assets
    assets = root.find('asset')
    for i, obj in enumerate(objects):
        mesh_elem = ET.SubElement(assets, 'mesh')
        mesh_elem.set('name', f"{obj['name']}_mesh")
        mesh_elem.set('file', obj['mesh'])
        mesh_elem.set('smoothnormal', 'true')
    
    # Remove original bunny
    for body in worldbody.findall(".//body[@name='bunny']"):
        worldbody.remove(body)
    
    # Add new objects
    for obj in objects:
        body = ET.SubElement(worldbody, 'body')
        body.set('name', obj['name'])
        body.set('pos', obj['pos'])
        
        joint = ET.SubElement(body, 'joint')
        joint.set('type', 'free')
        
        geom = ET.SubElement(body, 'geom')
        geom.set('type', 'mesh')
        geom.set('mesh', f"{obj['name']}_mesh")
        geom.set('material', obj['material'])
        geom.set('mass', '0.1')
        geom.set('friction', '1.0 0.1 0.01')
        if obj['size'] != '1.0 1.0 1.0':
            geom.set('size', obj['size'])
        
        inertial = ET.SubElement(body, 'inertial')
        inertial.set('pos', '0 0 0')
        inertial.set('mass', '0.1')
        inertial.set('diaginertia', '0.001 0.001 0.001')
    
    # Save modified scene in the correct directory
    output_path = "mujoco_menagerie/franka_emika_panda/multi_object_scene.xml"
    tree.write(output_path, xml_declaration=True, encoding='utf-8')
    print(f"Created multi-object scene: {output_path}")
    return output_path

class MultiObjectController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.current_target = 0
        self.phase_timer = 0
        self.objects = ['cube1', 'sphere1', 'bottle1']
        
        # Get actuator IDs
        actuator_names = [
            'actuator1', 'actuator2', 'actuator3', 'actuator4',
            'actuator5', 'actuator6', 'actuator7', 'actuator8'
        ]
        
        self.actuator_ids = []
        for name in actuator_names:
            try:
                actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.actuator_ids.append(actuator_id)
            except:
                pass
        
        print(f"Multi-object controller initialized")
        print(f"Found {len(self.actuator_ids)} actuators")
        print(f"Target objects: {self.objects}")
        
    def get_object_position(self, obj_name):
        """Get object position"""
        try:
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            return self.data.xpos[obj_id].copy()
        except:
            return np.array([0.5, 0.0, 0.1])
    
    def approach_object(self, obj_name):
        """Generate approach motion for specific object"""
        obj_pos = self.get_object_position(obj_name)
        
        # Different approach for each object type
        if 'cube' in obj_name:
            # Top-down grasp for cube
            approach = np.array([0.2, -0.5, 0.1, -2.0, 0.0, 1.5, 0.8])
        elif 'sphere' in obj_name:
            # Side grasp for sphere  
            approach = np.array([-0.1, -0.4, 0.2, -1.8, 0.3, 1.6, 0.6])
        else:  # bottle
            # Power grasp for bottle
            approach = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.8])
            
        return np.append(approach, 0)  # Open gripper
    
    def control_step(self, dt):
        """Main control loop"""
        self.phase_timer += dt
        
        cycle_time = 15.0  # 15 seconds per object
        
        # Switch target object every cycle
        if self.phase_timer > cycle_time:
            self.current_target = (self.current_target + 1) % len(self.objects)
            self.phase_timer = 0
            print(f"Switching to target: {self.objects[self.current_target]}")
        
        # Get current target
        target_obj = self.objects[self.current_target]
        
        # Generate motion
        t = self.phase_timer / cycle_time
        
        if t < 0.3:  # Approach
            target_angles = self.approach_object(target_obj)
        elif t < 0.4:  # Grasp
            target_angles = self.approach_object(target_obj)
            target_angles[-1] = 200  # Close gripper
        elif t < 0.7:  # Lift and move
            lift_pos = np.array([0.2, -0.2, 0.3, -1.5, 0.0, 1.3, 0.8])
            target_angles = np.append(lift_pos, 200)  # Keep gripper closed
        elif t < 0.8:  # Release
            target_angles = np.append(lift_pos, 0)  # Open gripper
        else:  # Return home
            home_pos = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.79])
            target_angles = np.append(home_pos, 0)  # Open gripper
        
        # Apply commands
        for i, actuator_id in enumerate(self.actuator_ids):
            if i < len(target_angles):
                self.data.ctrl[actuator_id] = target_angles[i]

def main():
    print("Multi-Object Robotics Demo")
    print("=" * 40)
    
    # Create multi-object scene
    scene_path = create_multi_object_scene()
    
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        print(f"Loaded scene with {model.nbody} bodies")
    except Exception as e:
        print(f"Error loading scene: {e}")
        return
    
    # Initialize controller
    controller = MultiObjectController(model, data)
    
    print("\nDemo Features:")
    print("- Robot targets 3 different objects")
    print("- Object-specific grasp strategies")
    print("- 15-second cycles per object")
    print("- Automatic target switching")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Physics step
            mujoco.mj_step(model, data)
            
            # Control step
            dt = time.time() - step_start
            controller.control_step(dt)
            
            # Update viewer
            viewer.sync()
            
            # Control speed
            elapsed = time.time() - step_start
            sleep_time = max(0, 1.0/60.0 - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    main() 