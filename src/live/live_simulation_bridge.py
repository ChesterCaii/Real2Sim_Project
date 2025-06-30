#!/usr/bin/env python3
"""
Live Simulation Bridge - Phase 4
Bridge between live camera detection and MuJoCo simulation updates
"""

import mujoco
import numpy as np
import time
import threading
import queue
import tempfile
import os
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

try:
    from camera_integration import LiveReconstructionPipeline, DetectedObject
    import open3d as o3d
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to run camera_integration.py first")

class SimulationObjectManager:
    """Manages dynamic objects in MuJoCo simulation"""
    
    def __init__(self, base_scene_path="mujoco_menagerie/franka_emika_panda/panda.xml"):
        self.base_scene_path = base_scene_path
        self.current_objects = {}  # id -> object info
        self.temp_dir = tempfile.mkdtemp(prefix="live_sim_")
        self.scene_counter = 0
        
        print(f"🏗️  Simulation Object Manager initialized")
        print(f"   📁 Temp directory: {self.temp_dir}")
    
    def create_dynamic_scene(self, detected_objects: List[DetectedObject]) -> str:
        """Create dynamic scene XML with current detected objects"""
        # Load base scene
        try:
            tree = ET.parse(self.base_scene_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"❌ Error parsing base scene: {e}")
            return None
        
        # Find or create worldbody
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')
        
        # Find or create asset section
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')
        
        # Remove old dynamic objects
        for elem in worldbody.findall(".//body[@name]"):
            if elem.get('name', '').startswith('live_object_'):
                worldbody.remove(elem)
        
        for elem in asset.findall(".//mesh[@name]"):
            if elem.get('name', '').startswith('live_mesh_'):
                asset.remove(elem)
        
        # Add new detected objects
        for obj in detected_objects:
            if obj.point_cloud and len(obj.point_cloud.points) > 50:
                self._add_object_to_scene(root, asset, worldbody, obj)
        
        # Save updated scene
        scene_path = os.path.join(self.temp_dir, f"live_scene_{self.scene_counter}.xml")
        tree.write(scene_path)
        self.scene_counter += 1
        
        return scene_path
    
    def _add_object_to_scene(self, root, asset, worldbody, obj: DetectedObject):
        """Add a detected object to the scene"""
        obj_id = f"live_object_{obj.id}"
        mesh_id = f"live_mesh_{obj.id}"
        
        # Convert point cloud to mesh and save
        mesh_path = self._save_object_mesh(obj)
        if not mesh_path:
            return
        
        # Add mesh to assets
        mesh_elem = ET.SubElement(asset, 'mesh')
        mesh_elem.set('name', mesh_id)
        mesh_elem.set('file', mesh_path)
        
        # Calculate object position (relative to robot base)
        # Transform camera coordinates to robot coordinates
        robot_pos = self._camera_to_robot_transform(obj.center_3d)
        
        # Add object body to worldbody
        body_elem = ET.SubElement(worldbody, 'body')
        body_elem.set('name', obj_id)
        body_elem.set('pos', f"{robot_pos[0]:.3f} {robot_pos[1]:.3f} {robot_pos[2]:.3f}")
        
        # Add joint for free movement
        joint_elem = ET.SubElement(body_elem, 'joint')
        joint_elem.set('type', 'free')
        
        # Add geometry
        geom_elem = ET.SubElement(body_elem, 'geom')
        geom_elem.set('type', 'mesh')
        geom_elem.set('mesh', mesh_id)
        geom_elem.set('mass', '0.1')  # Light objects
        geom_elem.set('rgba', f'{np.random.rand():.3f} {np.random.rand():.3f} {np.random.rand():.3f} 1')
        
        # Add inertial properties
        inertial_elem = ET.SubElement(body_elem, 'inertial')
        inertial_elem.set('pos', '0 0 0')
        inertial_elem.set('mass', '0.1')
        inertial_elem.set('diaginertia', '0.001 0.001 0.001')
    
    def _save_object_mesh(self, obj: DetectedObject) -> Optional[str]:
        """Convert point cloud to mesh and save as STL"""
        try:
            # Simplify point cloud
            pcd = obj.point_cloud
            if len(pcd.points) > 1000:
                pcd = pcd.uniform_down_sample(every_k_points=max(1, len(pcd.points)//500))
            
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            
            # Create mesh using ball pivoting
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2.0 * avg_dist
            radii = [radius, radius * 2]
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            
            if len(mesh.triangles) == 0:
                # Fallback to Poisson reconstruction
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=6, width=0, scale=1.1
                )
            
            if len(mesh.triangles) == 0:
                return None
            
            # Clean mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            
            # Save mesh
            mesh_path = os.path.join(self.temp_dir, f"live_object_{obj.id}.stl")
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            
            return mesh_path
            
        except Exception as e:
            print(f"⚠️  Failed to create mesh for object {obj.id}: {e}")
            return None
    
    def _camera_to_robot_transform(self, camera_pos: np.ndarray) -> np.ndarray:
        """Transform camera coordinates to robot base coordinates"""
        # Simple transformation: assume camera is mounted above robot workspace
        # This would need calibration in a real system
        
        x_cam, y_cam, z_cam = camera_pos
        
        # Transform to robot coordinates (simplified)
        # Assume camera is 0.5m above robot base, looking down
        x_robot = x_cam * 0.8  # Scale factor
        y_robot = y_cam * 0.8
        z_robot = max(0.05, 0.5 - z_cam)  # Above table surface
        
        # Add robot base offset
        x_robot += 0.0  # Robot base X
        y_robot += 0.0  # Robot base Y
        z_robot += 0.82  # Robot base height (Franka base height)
        
        return np.array([x_robot, y_robot, z_robot])
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

class LiveSimulationBridge:
    """Bridge between live camera and MuJoCo simulation"""
    
    def __init__(self, camera_type="auto", update_rate=2.0):
        # Initialize camera pipeline
        self.camera_pipeline = LiveReconstructionPipeline(camera_type=camera_type)
        
        # Initialize simulation manager
        self.sim_manager = SimulationObjectManager()
        
        # Control parameters
        self.update_rate = update_rate  # Hz
        self.running = False
        
        # MuJoCo simulation
        self.model = None
        self.data = None
        self.viewer = None
        
        # Threading
        self.update_thread = None
        self.last_update_time = 0
        
        print(f"🌉 Live Simulation Bridge initialized")
        print(f"   🔄 Update rate: {update_rate} Hz")
    
    def start(self):
        """Start the live simulation bridge"""
        print("🚀 Starting Live Simulation Bridge...")
        
        try:
            # Start camera pipeline
            self.camera_pipeline.start()
            time.sleep(2)  # Let camera stabilize
            
            # Create initial simulation
            self._update_simulation()
            
            # Start update thread
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            # Start simulation viewer
            self._start_simulation_viewer()
            
        except Exception as e:
            print(f"❌ Failed to start bridge: {e}")
            self.stop()
    
    def _update_loop(self):
        """Main update loop for simulation synchronization"""
        while self.running:
            current_time = time.time()
            
            if current_time - self.last_update_time >= (1.0 / self.update_rate):
                try:
                    self._update_simulation()
                    self.last_update_time = current_time
                except Exception as e:
                    print(f"⚠️  Update failed: {e}")
            
            time.sleep(0.01)  # Small delay
    
    def _update_simulation(self):
        """Update simulation with latest detected objects"""
        # Get latest detected objects from camera pipeline
        detected_objects = self._get_latest_objects()
        
        if detected_objects is None:
            return
        
        print(f"🔄 Updating simulation with {len(detected_objects)} objects")
        
        # Create new scene with detected objects
        scene_path = self.sim_manager.create_dynamic_scene(detected_objects)
        if not scene_path:
            return
        
        # Reload simulation
        try:
            old_model = self.model
            self.model = mujoco.MjModel.from_xml_path(scene_path)
            self.data = mujoco.MjData(self.model)
            
            # If viewer exists, update it
            if hasattr(self, 'viewer') and self.viewer is not None:
                # Note: MuJoCo viewer doesn't support hot model reloading
                # This is a limitation we'd need to work around in production
                pass
                
            print(f"✅ Simulation updated: {self.model.nbody} bodies, {self.model.nv} DOF")
            
        except Exception as e:
            print(f"❌ Failed to reload simulation: {e}")
            # Restore old model if available
            if old_model is not None:
                self.model = old_model
    
    def _get_latest_objects(self) -> Optional[List[DetectedObject]]:
        """Get latest detected objects from camera pipeline"""
        try:
            # Get result from camera pipeline
            result = self.camera_pipeline.result_queue.get_nowait()
            return result['objects']
        except queue.Empty:
            return None
        except Exception as e:
            print(f"⚠️  Failed to get objects: {e}")
            return None
    
    def _start_simulation_viewer(self):
        """Start MuJoCo simulation viewer"""
        if self.model is None:
            print("❌ No simulation model to display")
            return
        
        try:
            print("🎮 Starting simulation viewer...")
            print("   Controls:")
            print("   • Mouse: Rotate camera")
            print("   • Right-click + drag: Pan")
            print("   • Scroll: Zoom")
            print("   • Press 'q' in camera window to quit")
            
            # Launch passive viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                
                while self.running and viewer.is_running():
                    step_start = time.time()
                    
                    # Step physics
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    
                    # Maintain real-time
                    elapsed = time.time() - step_start
                    if elapsed < 0.002:
                        time.sleep(0.002 - elapsed)
                
        except Exception as e:
            print(f"❌ Viewer error: {e}")
        finally:
            self.viewer = None
    
    def stop(self):
        """Stop the live simulation bridge"""
        print("🔴 Stopping Live Simulation Bridge...")
        
        self.running = False
        
        # Stop update thread
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        
        # Stop camera pipeline
        try:
            self.camera_pipeline.stop()
        except Exception as e:
            print(f"⚠️  Camera stop error: {e}")
        
        # Cleanup simulation manager
        try:
            self.sim_manager.cleanup()
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
        
        print("✅ Live Simulation Bridge stopped")

def main():
    """Main function for live simulation bridge demo"""
    print("============================================================")
    print("🌉 LIVE SIMULATION BRIDGE - PHASE 4")
    print("============================================================")
    print()
    print("This demo combines:")
    print("  🎥 Live camera feed (RGB-D)")
    print("  🔍 Real-time object detection")
    print("  🔄 Dynamic 3D reconstruction")
    print("  🤖 Live MuJoCo simulation updates")
    print()
    
    bridge = None
    try:
        # Create and start bridge
        bridge = LiveSimulationBridge(camera_type="auto", update_rate=1.0)
        bridge.start()
        
        # Keep running until stopped
        print("🏃 Bridge running... Press Ctrl+C to stop")
        while bridge.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bridge:
            bridge.stop()

if __name__ == "__main__":
    main() 