#!/usr/bin/env python3
"""
Mesh Simplification - Phase 3B
Reduce mesh complexity for MuJoCo compatibility (max 200,000 faces)
"""

import open3d as o3d
import os
import glob

def simplify_mesh(input_path, output_path=None, target_triangles=100000):
    """
    Simplify a mesh to reduce triangle count
    
    Args:
        input_path: Path to input STL file
        output_path: Path to output STL file (optional)
        target_triangles: Target number of triangles
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_simplified{ext}"
    
    print(f"🔄 Simplifying: {os.path.basename(input_path)}")
    
    try:
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(input_path)
        
        if len(mesh.triangles) == 0:
            print(f"❌ No triangles found in {input_path}")
            return None
        
        original_triangles = len(mesh.triangles)
        original_vertices = len(mesh.vertices)
        
        print(f"   📊 Original: {original_vertices} vertices, {original_triangles} triangles")
        
        if original_triangles <= target_triangles:
            print(f"   ✅ Already within limit ({target_triangles}), copying as-is")
            o3d.io.write_triangle_mesh(output_path, mesh)
            return output_path
        
        # Calculate reduction ratio
        reduction_ratio = target_triangles / original_triangles
        print(f"   🎯 Target: {target_triangles} triangles (reduction ratio: {reduction_ratio:.3f})")
        
        # Simplify mesh using quadric decimation
        simplified_mesh = mesh.simplify_quadric_decimation(target_triangles)
        
        # Clean up mesh first
        simplified_mesh.remove_degenerate_triangles()
        simplified_mesh.remove_duplicated_triangles()
        simplified_mesh.remove_duplicated_vertices()
        simplified_mesh.remove_unreferenced_vertices()
        
        # Compute vertex normals (required for STL output)
        simplified_mesh.compute_vertex_normals()
        
        final_triangles = len(simplified_mesh.triangles)
        final_vertices = len(simplified_mesh.vertices)
        
        print(f"   ✅ Simplified: {final_vertices} vertices, {final_triangles} triangles")
        print(f"   📉 Reduction: {(1-final_triangles/original_triangles)*100:.1f}%")
        
        # Save simplified mesh
        success = o3d.io.write_triangle_mesh(output_path, simplified_mesh)
        if success:
            file_size = os.path.getsize(output_path)
            print(f"   💾 Saved: {output_path} ({file_size} bytes)")
            return output_path
        else:
            print(f"   ❌ Failed to save simplified mesh")
            return None
            
    except Exception as e:
        print(f"   ❌ Simplification failed: {e}")
        return None

def simplify_all_meshes(input_dir="reconstructed_objects", output_dir=None, max_faces=150000):
    """
    Simplify all STL meshes in a directory
    
    Args:
        input_dir: Directory containing STL files
        output_dir: Output directory (default: same as input)
        max_faces: Maximum faces per mesh for MuJoCo compatibility
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("============================================================")
    print("🔧 MESH SIMPLIFICATION - PHASE 3B")
    print("============================================================")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Max faces limit: {max_faces}")
    
    # Find all STL files
    pattern = os.path.join(input_dir, "*.stl")
    stl_files = glob.glob(pattern)
    
    if len(stl_files) == 0:
        print(f"❌ No STL files found in {input_dir}")
        return
    
    print(f"📊 Found {len(stl_files)} STL files to process")
    
    simplified_files = []
    large_meshes = []
    
    for stl_file in sorted(stl_files):
        # Check file size first
        file_size = os.path.getsize(stl_file)
        if file_size > 10_000_000:  # 10MB
            large_meshes.append(stl_file)
        
        basename = os.path.basename(stl_file)
        output_path = os.path.join(output_dir, basename)
        
        # Skip if input and output are the same and we're not simplifying
        if stl_file == output_path and file_size < 10_000_000:
            print(f"⏭️  Skipping small file: {basename}")
            simplified_files.append(output_path)
            continue
        
        result = simplify_mesh(stl_file, output_path, target_triangles=max_faces)
        if result:
            simplified_files.append(result)
    
    print(f"\n✅ Simplification Complete!")
    print(f"   📊 Processed: {len(stl_files)} files")
    print(f"   ✅ Simplified: {len(simplified_files)} files")
    print(f"   ⚠️  Large meshes processed: {len(large_meshes)}")
    
    if large_meshes:
        print(f"\n📊 Large meshes that were simplified:")
        for mesh_file in large_meshes:
            file_size = os.path.getsize(mesh_file)
            print(f"   • {os.path.basename(mesh_file)}: {file_size/1000000:.1f}MB")
    
    return simplified_files

def update_scene_xml(scene_path):
    """Update the scene XML to use the simplified meshes"""
    print(f"\n🔄 Scene XML is already configured correctly")
    print(f"   💡 Simplified meshes have the same filenames")
    print(f"   📄 Scene file: {scene_path}")

def main():
    """Main simplification process"""
    # Simplify all meshes in the reconstructed_objects directory
    simplified_files = simplify_all_meshes(
        input_dir="reconstructed_objects",
        output_dir="reconstructed_objects",  # Overwrite originals
        max_faces=150000  # Conservative limit for MuJoCo
    )
    
    if simplified_files:
        # Update scene XML (no changes needed since we're overwriting)
        update_scene_xml("reconstructed_objects/multi_object_scene.xml")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Test scene: mjpython test_multi_object_scene.py")
        print(f"   2. Launch viewer: mjpython -m mujoco.viewer reconstructed_objects/multi_object_scene.xml")
        print(f"   3. All meshes should now be within MuJoCo's 200K face limit")

if __name__ == "__main__":
    main() 