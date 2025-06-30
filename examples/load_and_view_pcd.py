import open3d as o3d
import os

# --- Configuration ---
# The name of the point cloud file you downloaded.
# Make sure this file is in the same directory as the script.
pcd_filename = "bunny.pcd"

# --- Main Script ---

print(f"Attempting to load point cloud file: {pcd_filename}")

# Check if the file exists before trying to load it
if not os.path.exists(pcd_filename):
    print(f"Error: The file '{pcd_filename}' was not found in this directory.")
    print("Please make sure you have downloaded it and placed it in the correct folder.")
else:
    # Read the point cloud file from disk
    pcd = o3d.io.read_point_cloud(pcd_filename)

    # Check if the point cloud was loaded successfully
    if not pcd.has_points():
        print(f"Error: Failed to read or parse the point cloud file.")
    else:
        print("Point cloud loaded successfully.")
        print("It has", len(pcd.points), "points.")
        
        # Let's add some color to make it look nicer
        pcd.paint_uniform_color([0.6, 0.6, 0.6]) # Gray

        print("Displaying the point cloud. Close the window to continue.")
        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

        print("Script finished.")
