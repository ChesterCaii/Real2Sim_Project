import open3d as o3d
import numpy as np

# This is a simple script to test your Open3D installation.

print("Creating a 3D cube mesh...")

# Create a sample mesh of a cube
# The 'create_box' function is a simple way to make a test shape.
mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)

# Add color and compute normals for better visualization
mesh_box.paint_uniform_color([0.8, 0.2, 0.2]) # Set color to a reddish-gray
mesh_box.compute_vertex_normals()

print("Displaying the cube in a new window. Close the window to exit the script.")

# Visualize the mesh in an interactive window
o3d.visualization.draw_geometries([mesh_box])

print("Window closed. Script finished.")
