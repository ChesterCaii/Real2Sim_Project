import mujoco
import mujoco.viewer
import os

print("--- Real-to-Simulation Fixed Version ---")
print("Loading combined robot and bunny scene...")

# Path to our combined scene XML (now in the robot's directory)
scene_xml_path = "mujoco_menagerie/franka_emika_panda/robot_bunny_scene.xml"

try:
    # Load the combined model from XML
    model = mujoco.MjModel.from_xml_path(scene_xml_path)
    data = mujoco.MjData(model)
    
    print(f"✅ Successfully loaded scene with {model.nbody} bodies and {model.nq} DOFs")
    print("Bodies in the scene:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            print(f"  - {body_name}")
    
    print(f"\n🎯 Scene contains {model.nmesh} meshes")
    print("Available meshes:")
    for i in range(model.nmesh):
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        if mesh_name:
            print(f"  - {mesh_name}")
    
    print(f"\n🤖 Robot has {model.nu} actuated joints")
    print("\n🐰 Bunny added as a free-floating object")
    
    print("\n🚀 Launching interactive viewer...")
    print("Controls:")
    print("  - Mouse: Rotate and zoom the camera")
    print("  - Right-click + drag: Pan the camera")
    print("  - Scroll: Zoom in/out")
    print("  - Space: Pause/unpause simulation")
    
    # Launch the viewer
    mujoco.viewer.launch(model, data)
    
except FileNotFoundError as e:
    print(f"❌ Error: Could not find required files: {e}")
    print("Make sure you have:")
    print("1. The mujoco_menagerie directory with Franka robot files")
    print("2. The bunny_final.stl file copied to the robot directory")
    
except Exception as e:
    print(f"❌ An error occurred: {e}")
    import traceback
    traceback.print_exc() 