import mujoco
import mujoco.viewer
import os

# --- Step 1: Load the Robot Model ---
# We load the robot from its own scene file, which is guaranteed to work.
# We get the full, absolute path to be safe.
robot_xml_path = os.path.join(os.getcwd(), 'mujoco_menagerie/franka_emika_panda/scene.xml')
print(f"Loading robot from: {robot_xml_path}")
robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)

# --- Step 2: Create a World with the Bunny ---
# We define our world in a string. The bunny's STL file is in the same
# main directory as this script, so the path is simple.
print("Creating world with bunny...")
world_xml = """
<mujoco>
  <asset>
    <mesh name="bunny_mesh" file="bunny_final.stl" />
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
world_model = mujoco.MjModel.from_xml_string(world_xml)

# --- Step 3: Merge the Models in Memory ---
# This is the correct way to call the merge function.
print("Merging models...")
model_list = [robot_model, world_model]
merged_model, _ = mujoco.mj_mergeModels(model_list)

# --- Step 4: Run the Simulation ---
print("Launching viewer...")
data = mujoco.MjData(merged_model)
mujoco.viewer.launch_passive(merged_model, data)

print("Simulation finished.")
