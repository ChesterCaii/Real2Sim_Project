import open3d as o3d
import numpy as np
import os

# Get the project root directory (two levels up from src/reconstruction/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# --- Step 1: Load and prepare the point cloud ---
print("Loading point cloud...")
input_path = os.path.join(project_root, "data", "point_clouds", "bunny.pcd")
pcd = o3d.io.read_point_cloud(input_path)

print("Estimating normals for the point cloud...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
# Orient normals consistently
pcd.orient_normals_consistent_tangent_plane(20)

# --- Step 2: Enhanced Poisson Surface Reconstruction ---
print("\nAttempting ENHANCED reconstruction with Poisson algorithm...")
mesh = None
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    # Use higher depth for more detail
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10, width=0, scale=1.1, linear_fit=False)

# --- Step 3: Check if Poisson worked. If not, try Ball Pivoting ---
if not mesh or not mesh.has_triangles():
    print("\n--- POISSON FAILED ---")
    print("The Poisson algorithm did not produce a valid mesh.")
    print("Switching to the Ball Pivoting Algorithm (BPA)...")
    
    # BPA requires a radius parameter. We can estimate it from the point cloud.
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist
    
    radii = [radius, radius * 2]
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))

# --- Step 4: ENHANCED mesh processing for better visuals ---
if mesh and mesh.has_triangles():
    print(f"\n Initial reconstruction successful!")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")
    
    print("\n Enhancing mesh quality...")
    
    # 1. Clean up the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    
    # 2. Filter out low-density vertices (improves surface quality)
    if 'densities' in locals():
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)  # Remove bottom 1%
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 3. Smooth the mesh using Laplacian smoothing
    print("    Applying Laplacian smoothing...")
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=5, lambda_filter=0.5)
    
    # 4. Subdivide mesh for smoother appearance (if not too large)
    if len(mesh.triangles) < 50000:
        print("    Subdividing mesh for smoother surface...")
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        # Apply another round of smoothing after subdivision
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.3)
    
    # 5. Compute high-quality vertex normals
    mesh.compute_vertex_normals()
    
    print(f"\n Enhanced mesh quality:")
    print(f"    Final Vertices: {len(mesh.vertices)}")
    print(f"    Final Triangles: {len(mesh.triangles)}")

    # Let's visualize the enhanced mesh
    print("\n👀 Displaying the enhanced reconstructed mesh...")
    mesh.paint_uniform_color([0.8, 0.3, 0.3])  # Nice red color
    o3d.visualization.draw_geometries([mesh], 
                                    window_name="Enhanced Bunny Mesh",
                                    mesh_show_back_face=True)

    # Save the enhanced mesh
    output_path = os.path.join(project_root, "data", "meshes", "bunny_final.stl")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"\n SUCCESS! Enhanced mesh saved to '{output_path}'")
    
    # Also save as OBJ for better normals
    obj_path = os.path.join(project_root, "data", "meshes", "bunny_final.obj")
    o3d.io.write_triangle_mesh(obj_path, mesh)
    print(f" Also saved as OBJ: '{obj_path}'")

else:
    print("\n--- RECONSTRUCTION FAILED ---")
    print("Both Poisson and Ball Pivoting algorithms failed to create a mesh.")
    print("Try adjusting parameters or using a different point cloud.")

def main():
    """Main function for the reconstruction script"""
    pass

if __name__ == "__main__":
    main()
