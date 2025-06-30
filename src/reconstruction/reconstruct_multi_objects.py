#!/usr/bin/env python3
"""
Multi-Object Reconstruction - Phase 3B
Process segmented objects and create individual 3D models for each
"""

import open3d as o3d
import numpy as np
import cv2
import os
import glob
from PIL import Image

class MultiObjectReconstructor:
    def __init__(self, objects_dir="segmented_objects"):
        """Initialize multi-object reconstructor"""
        self.objects_dir = objects_dir
        self.output_dir = "reconstructed_objects"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🔧 Multi-Object Reconstructor initialized")
        print(f"   📁 Input directory: {self.objects_dir}")
        print(f"   📁 Output directory: {self.output_dir}")
    
    def image_to_point_cloud(self, image_path, depth_scale=0.1, z_offset=0.0):
        """
        Convert 2D object image to 3D point cloud
        
        For demo purposes, we'll create a simple depth map from the image.
        In real applications, this would use actual RGB-D camera data.
        """
        print(f"🔄 Processing: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Create synthetic depth map
        # For demo: use image intensity to create depth variation
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Create mask for object (non-white pixels)
        mask = np.all(image_rgb != [255, 255, 255], axis=2)
        
        if not np.any(mask):
            print(f"⚠️  No object pixels found in {image_path}")
            return None
        
        # Create depth map: darker pixels = closer (higher depth values)
        depth_map = np.zeros_like(gray, dtype=np.float32)
        depth_map[mask] = (255 - gray[mask]) * depth_scale / 255.0 + z_offset
        
        # Generate point cloud
        points = []
        colors = []
        
        # Camera intrinsics (synthetic)
        fx = fy = width  # Simple perspective
        cx, cy = width // 2, height // 2
        
        for y in range(height):
            for x in range(width):
                if mask[y, x] and depth_map[y, x] > 0:
                    # Convert pixel to 3D point
                    z = depth_map[y, x]
                    px = (x - cx) * z / fx
                    py = (y - cy) * z / fy
                    
                    points.append([px, py, z])
                    colors.append(image_rgb[y, x] / 255.0)
        
        if len(points) == 0:
            print(f"⚠️  No valid 3D points generated for {image_path}")
            return None
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        
        print(f"   ✅ Generated point cloud with {len(points)} points")
        return pcd
    
    def point_cloud_to_mesh(self, pcd, object_id, method="poisson"):
        """Convert point cloud to mesh using specified method"""
        if pcd is None or len(pcd.points) == 0:
            return None
        
        print(f"   🔧 Reconstructing mesh using {method} method...")
        
        try:
            if method == "poisson":
                # Poisson surface reconstruction
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=8, width=0, scale=1.1, linear_fit=False
                )
                
                if len(mesh.triangles) > 0:
                    # Clean up mesh
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_duplicated_vertices()
                    mesh.remove_unreferenced_vertices()
                    mesh.compute_vertex_normals()
                    
                    print(f"   ✅ Poisson reconstruction: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                    return mesh
            
            elif method == "ball_pivoting":
                # Ball pivoting algorithm
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radius = 2.0 * avg_dist
                radii = [radius, radius * 2, radius * 4]
                
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
                
                if len(mesh.triangles) > 0:
                    mesh.compute_vertex_normals()
                    print(f"   ✅ Ball pivoting reconstruction: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                    return mesh
            
        except Exception as e:
            print(f"   ❌ {method} reconstruction failed: {e}")
        
        return None
    
    def save_mesh(self, mesh, object_id):
        """Save mesh to STL file"""
        if mesh is None:
            return None
        
        output_path = os.path.join(self.output_dir, f"object_{object_id:03d}.stl")
        
        try:
            success = o3d.io.write_triangle_mesh(output_path, mesh)
            if success:
                file_size = os.path.getsize(output_path)
                print(f"   💾 Saved mesh: {output_path} ({file_size} bytes)")
                return output_path
            else:
                print(f"   ❌ Failed to save mesh: {output_path}")
        except Exception as e:
            print(f"   ❌ Error saving mesh: {e}")
        
        return None
    
    def process_all_objects(self):
        """Process all segmented objects and create 3D models"""
        print("\n🔄 Processing all segmented objects...")
        
        # Find all object images
        pattern = os.path.join(self.objects_dir, "object_*.png")
        object_files = sorted(glob.glob(pattern))
        
        if len(object_files) == 0:
            print(f"❌ No object files found in {self.objects_dir}")
            print("💡 Run object_segmentation.py first to generate segmented objects")
            return []
        
        print(f"📊 Found {len(object_files)} objects to process")
        
        reconstructed_objects = []
        
        for i, image_path in enumerate(object_files):
            print(f"\n📍 Processing object {i+1}/{len(object_files)}")
            
            # Extract object ID from filename
            basename = os.path.basename(image_path)
            object_id = int(basename.split('_')[1].split('.')[0])
            
            # Convert image to point cloud
            pcd = self.image_to_point_cloud(image_path, depth_scale=0.05, z_offset=0.02)
            
            if pcd is None:
                continue
            
            # Try Poisson reconstruction first, fall back to ball pivoting
            mesh = self.point_cloud_to_mesh(pcd, object_id, method="poisson")
            if mesh is None:
                mesh = self.point_cloud_to_mesh(pcd, object_id, method="ball_pivoting")
            
            if mesh is None:
                print(f"   ❌ Failed to reconstruct object {object_id}")
                continue
            
            # Save mesh
            output_path = self.save_mesh(mesh, object_id)
            if output_path:
                reconstructed_objects.append({
                    'id': object_id,
                    'mesh_path': output_path,
                    'vertices': len(mesh.vertices),
                    'triangles': len(mesh.triangles)
                })
        
        return reconstructed_objects
    
    def create_multi_object_scene(self, reconstructed_objects):
        """Create a simulation scene with multiple reconstructed objects"""
        if len(reconstructed_objects) == 0:
            print("❌ No objects to create scene")
            return None
        
        print(f"\n🏗️  Creating multi-object simulation scene...")
        
        scene_xml = f"""<mujoco model="multi-object scene">
  <compiler angle="radian" meshdir="{self.output_dir}" autolimits="true"/>
  <option integrator="implicitfast"/>
  
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  
  <asset>
    <!-- Ground texture -->
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    
    <!-- Object meshes -->"""
        
        for obj in reconstructed_objects:
            mesh_name = f"object_{obj['id']:03d}_mesh"
            mesh_file = f"object_{obj['id']:03d}.stl"
            scene_xml += f'\n    <mesh name="{mesh_name}" file="{mesh_file}" />'
        
        scene_xml += """
  </asset>
  
  <worldbody>
    <!-- Lighting -->
    <light name="top" pos="0 0 2" mode="trackcom"/>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    
    <!-- Ground plane -->
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    
    <!-- Reconstructed objects -->"""
        
        # Arrange objects in a grid
        cols = int(np.ceil(np.sqrt(len(reconstructed_objects))))
        spacing = 0.3
        
        for i, obj in enumerate(reconstructed_objects):
            row = i // cols
            col = i % cols
            x = (col - cols/2) * spacing
            y = (row - cols/2) * spacing
            
            # Random color for each object
            color = np.random.rand(3)
            color_str = f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} 1"
            
            scene_xml += f"""
    <body name="object_{obj['id']:03d}" pos="{x:.3f} {y:.3f} 0.15">
      <joint type="free"/>
      <geom type="mesh" 
            mesh="object_{obj['id']:03d}_mesh" 
            mass="0.3" 
            friction="0.8 0.005 0.0001"
            rgba="{color_str}" />
      <inertial pos="0 0 0" mass="0.3" diaginertia="0.005 0.005 0.005"/>
    </body>"""
        
        scene_xml += """
  </worldbody>
</mujoco>"""
        
        # Save scene file
        scene_path = os.path.join(self.output_dir, "multi_object_scene.xml")
        with open(scene_path, 'w') as f:
            f.write(scene_xml)
        
        print(f"✅ Multi-object scene created: {scene_path}")
        return scene_path

def main():
    """Main multi-object reconstruction"""
    print("============================================================")
    print("🔧 MULTI-OBJECT RECONSTRUCTION - PHASE 3B")
    print("============================================================")
    
    # Initialize reconstructor
    reconstructor = MultiObjectReconstructor()
    
    # Process all segmented objects
    reconstructed_objects = reconstructor.process_all_objects()
    
    if len(reconstructed_objects) == 0:
        print("\n❌ No objects were successfully reconstructed")
        print("💡 Make sure to run object_segmentation.py first")
        return
    
    # Create multi-object simulation scene
    scene_path = reconstructor.create_multi_object_scene(reconstructed_objects)
    
    # Summary
    print(f"\n✅ Multi-Object Reconstruction Complete!")
    print(f"   📊 Objects reconstructed: {len(reconstructed_objects)}")
    print(f"   📁 Mesh files: {reconstructor.output_dir}/")
    print(f"   🏗️  Scene file: {scene_path}")
    
    print(f"\n📊 Reconstruction Summary:")
    for obj in reconstructed_objects:
        print(f"   Object {obj['id']:03d}: {obj['vertices']} vertices, {obj['triangles']} triangles")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Test scene: mjpython -m mujoco.viewer {scene_path}")
    print(f"   2. Integrate with robot control for multi-object manipulation")
    print(f"   3. Extend to real RGB-D camera feeds")

if __name__ == "__main__":
    main() 