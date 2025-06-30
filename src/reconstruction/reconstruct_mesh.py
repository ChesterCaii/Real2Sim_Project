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

# --- Step 2: Try Poisson Surface Reconstruction ---
print("\nAttempting reconstruction with Poisson algorithm...")
mesh = None # Initialize mesh to None
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)

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

# --- Step 4: Final check and save the result ---
if mesh and mesh.has_triangles():
    print("\nReconstruction successful!")
    
    # Clean up the mesh a little
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    
    # Compute vertex normals (required for STL output)
    mesh.compute_vertex_normals()

    # Let's visualize the final mesh
    print("Displaying the final reconstructed mesh.")
    mesh.paint_uniform_color([0.8, 0.2, 0.2]) # Red
    o3d.visualization.draw_geometries([mesh])

    # Save the successful mesh to the STL file
    output_path = os.path.join(project_root, "data", "meshes", "bunny_final.stl")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Success! Mesh saved to '{output_path}'")

else:
    print("\n--- RECONSTRUCTION FAILED ---")
    print("Both Poisson and Ball Pivoting algorithms failed to create a mesh.")
    print("Try adjusting parameters or using a different point cloud.")

def main():
    """Main function for the reconstruction script"""
    pass

if __name__ == "__main__":
    main()
