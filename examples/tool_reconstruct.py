import open3d as o3d
import numpy as np
import argparse # Import the argument parsing library

# --- Setup Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Reconstruct a mesh from a point cloud file.")
parser.add_argument("--input", required=True, help="Path to the input point cloud file (.pcd)")
parser.add_argument("--output", required=True, help="Path to save the output mesh file (.stl)")
parser.add_argument("--depth", type=int, default=9, help="Poisson reconstruction depth (e.g., 8, 9, 10)")
args = parser.parse_args()


# --- Main Script Logic ---
print(f"Loading point cloud from: {args.input}")
pcd = o3d.io.read_point_cloud(args.input)

if not pcd.has_points():
    print("Error: Input point cloud is empty or could not be read.")
    exit()

print("Estimating normals...")
pcd.estimate_normals()

print(f"Attempting reconstruction with Poisson (depth={args.depth})...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)

if mesh and mesh.has_triangles():
    print("Reconstruction successful!")
    
    # Clean up the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # Save the successful mesh to the specified output file
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"Success! Mesh saved to '{args.output}'")

    # Optional: Visualize the result
    print("Displaying the final mesh.")
    mesh.paint_uniform_color([0.7, 0.1, 0.1])
    o3d.visualization.draw_geometries([mesh])
else:
    print("--- RECONSTRUCTION FAILED ---")
    print("Could not create a valid mesh. Try adjusting the --depth parameter.")
