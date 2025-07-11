import mujoco
import mujoco.viewer
# We don't need to import the submodule explicitly.

# --- Step 1: Load the Robot Model ---
print("Loading robot model...")
robot_model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")

# --- Step 2: Create a Simple Bunny Model ---
print("Creating bunny and world model...")
world_and_bunny_xml = """
<mujoco>
  <asset>
    <mesh name="bunny_mesh" file="data/meshes/bunny_final.stl" />
  </asset>
  <worldbody>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    <body name="bunny" pos="0.5 0 0.1">
      <joint type="free"/>
      <geom type="mesh" mesh="bunny_mesh" mass="0.5" rgba="0.8 0.2 0.2 1" />
    </body>
  </worldbody>
</mujoco>
"""
world_and_bunny_model = mujoco.MjModel.from_xml_string(world_and_bunny_xml)

# --- Step 3: Merge the Two Models ---
print("Merging models...")
# THIS IS THE CORRECTED FUNCTION CALL with the underscore:
merged_model = mujoco._functions.mj_mergeModels([robot_model, world_and_bunny_model])

# --- Step 4: Run the Simulation with the Merged Model ---
print("Launching viewer with the final merged model...")
data = mujoco.MjData(merged_model)
mujoco.viewer.launch_passive(merged_model, data)

print("Simulation finished.")
