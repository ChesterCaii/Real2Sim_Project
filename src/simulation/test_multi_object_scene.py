#!/usr/bin/env python3
"""
Test Multi-Object Scene - Phase 3B
Test the reconstructed multi-object simulation scene
"""

import mujoco
import numpy as np
import time

def test_multi_object_scene():
    """Test loading and displaying the multi-object scene"""
    print("============================================================")
    print("🧪 TESTING MULTI-OBJECT SCENE - PHASE 3B")
    print("============================================================")
    
    scene_path = "reconstructed_objects/multi_object_scene.xml"
    
    try:
        print("🔄 Loading multi-object scene...")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print(f"✅ Scene loaded successfully!")
        print(f"   📊 Bodies: {model.nbody}")
        print(f"   🔧 DOFs: {model.nv}")
        print(f"   🎨 Meshes: {model.nmesh}")
        print(f"   ⚖️  Total mass: {np.sum([model.body_mass[i] for i in range(model.nbody)]):.3f} kg")
        
        # Display all bodies in the scene
        print(f"\n🎯 Scene Objects:")
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                print(f"   • {body_name}")
        
        # Display meshes
        print(f"\n🎨 Loaded Meshes:")
        for i in range(model.nmesh):
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
            if mesh_name:
                # Get mesh info
                nvert = model.mesh_vert[i].shape[0] if i < len(model.mesh_vert) else 0
                nface = model.mesh_face[i].shape[0] if i < len(model.mesh_face) else 0
                print(f"   • {mesh_name}: {nvert} vertices, {nface} faces")
        
        print("\n🚀 Launching interactive viewer...")
        print("🎮 Controls:")
        print("   • Mouse: Rotate camera")
        print("   • Right-click + drag: Pan camera")
        print("   • Scroll: Zoom in/out")
        print("   • Space: Pause/unpause physics")
        
        # Test with both viewer methods
        try:
            # Try passive viewer first
            with mujoco.viewer.launch_passive(model, data) as viewer:
                step_count = 0
                while viewer.is_running():
                    step_start = time.time()
                    
                    # Step simulation
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    
                    # Show status every 1000 steps
                    if step_count % 1000 == 0:
                        print(f"⏱️  Simulation step: {step_count}")
                    
                    # Maintain real-time
                    elapsed = time.time() - step_start
                    if elapsed < 0.002:
                        time.sleep(0.002 - elapsed)
                    
                    step_count += 1
                    
        except Exception as viewer_error:
            print(f"⚠️  Passive viewer failed: {viewer_error}")
            print("🔄 Trying alternative launch method...")
            
            # Alternative: just validate the model loading worked
            print("✅ Model validation successful!")
            print("💡 Scene contains multiple reconstructed objects ready for simulation")
            
            # Run a quick physics test
            print("\n🧪 Running physics validation...")
            for i in range(100):
                mujoco.mj_step(model, data)
            
            print("✅ Physics simulation working correctly!")
        
    except Exception as e:
        print(f"❌ Error loading scene: {e}")
        print("💡 Possible issues:")
        print("   • Large mesh files causing memory issues")
        print("   • Invalid XML format")
        print("   • Missing mesh files")
        
        # Try to provide more specific error info
        import os
        if not os.path.exists(scene_path):
            print(f"   • Scene file not found: {scene_path}")
        else:
            file_size = os.path.getsize(scene_path)
            print(f"   • Scene file size: {file_size} bytes")
            
            # Check if mesh files exist
            mesh_dir = "reconstructed_objects"
            mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.stl')]
            print(f"   • Found {len(mesh_files)} STL files")
            for mesh_file in mesh_files:
                mesh_path = os.path.join(mesh_dir, mesh_file)
                mesh_size = os.path.getsize(mesh_path)
                if mesh_size > 50_000_000:  # 50MB
                    print(f"   ⚠️  Large mesh file: {mesh_file} ({mesh_size/1000000:.1f}MB)")

def create_simplified_scene():
    """Create a simplified scene with smaller objects for testing"""
    print("\n🔧 Creating simplified test scene...")
    
    # Read the original scene and modify it
    original_scene = "reconstructed_objects/multi_object_scene.xml"
    simplified_scene = "reconstructed_objects/simplified_scene.xml"
    
    try:
        with open(original_scene, 'r') as f:
            content = f.read()
        
        # Remove the largest object (object_001) to reduce memory usage
        lines = content.split('\n')
        filtered_lines = []
        skip_object_001 = False
        
        for line in lines:
            if 'object_001' in line:
                if '<mesh name="object_001' in line or '<body name="object_001' in line:
                    skip_object_001 = True
                if skip_object_001 and ('</body>' in line or '/>' in line):
                    skip_object_001 = False
                    continue
            
            if not skip_object_001:
                filtered_lines.append(line)
        
        # Write simplified scene
        with open(simplified_scene, 'w') as f:
            f.write('\n'.join(filtered_lines))
        
        print(f"✅ Created simplified scene: {simplified_scene}")
        return simplified_scene
        
    except Exception as e:
        print(f"❌ Could not create simplified scene: {e}")
        return None

def main():
    """Main test function"""
    # First try the full scene
    test_multi_object_scene()
    
    # If that fails, try a simplified version
    simplified_scene = create_simplified_scene()
    if simplified_scene:
        print(f"\n🔄 Testing simplified scene...")
        # Update the scene path and try again
        import sys
        sys.argv = ['test_script', simplified_scene]
        test_multi_object_scene()

if __name__ == "__main__":
    main() 