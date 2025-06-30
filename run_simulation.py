import mujoco
import mujoco.viewer

# Just point to our robust XML file.
xml_path = "robot_scene.xml"

print(f"Loading MuJoCo scene from: {xml_path}")

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print("Model loaded successfully. Launching viewer...")
    mujoco.viewer.launch_passive(model, data)

except Exception as e:
    print(f"An error occurred: {e}")
