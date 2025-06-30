#!/usr/bin/env python3
"""
Intelligent Grasp Planning - Phase 5
Advanced grasp planning using object classification and pose estimation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo not available for IK. Planning will work without it.")

try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not available. Install with: pip install scipy")

class GraspType(Enum):
    """Types of grasps"""
    TOP_DOWN = "top_down"           # Grasp from above
    SIDE_APPROACH = "side_approach" # Grasp from the side
    PINCH_GRASP = "pinch_grasp"    # Two-finger pinch
    POWER_GRASP = "power_grasp"    # Full hand wrap

@dataclass
class GraspPose:
    """Grasp pose with robot configuration"""
    position: np.ndarray           # 3D position (x, y, z)
    orientation: np.ndarray        # Quaternion (w, x, y, z)
    approach_vector: np.ndarray    # Direction to approach from
    grasp_width: float            # Gripper opening width
    grasp_type: GraspType
    confidence: float             # Success confidence (0-1)
    pre_grasp_offset: float = 0.1  # Distance before contact

@dataclass
class ObjectGraspInfo:
    """Object-specific grasp information"""
    class_name: str
    bbox: Tuple[int, int, int, int]
    estimated_size: Tuple[float, float, float]  # width, height, depth
    center_3d: np.ndarray
    preferred_grasps: List[GraspPose]
    material_properties: Dict[str, float]  # friction, weight, etc.

class IntelligentGraspPlanner:
    """Advanced grasp planning with object-specific strategies"""
    
    def __init__(self, robot_model=None):
        """
        Initialize grasp planner
        
        Args:
            robot_model: MuJoCo model for robot kinematics
        """
        self.robot_model = robot_model
        
        # Object-specific grasp knowledge
        self.grasp_database = self._build_grasp_database()
        
        # Robot constraints
        self.workspace_bounds = {
            'x': (0.2, 0.8),    # meters
            'y': (-0.4, 0.4),   # meters  
            'z': (0.0, 0.5)     # meters
        }
        
        self.gripper_limits = {
            'max_width': 0.08,  # meters
            'min_width': 0.01   # meters
        }
    
    def _build_grasp_database(self) -> Dict[str, Dict]:
        """Build database of object-specific grasp strategies"""
        return {
            # Small objects - precision grasps
            'apple': {
                'size_estimate': (0.08, 0.08, 0.08),
                'preferred_grasps': [GraspType.TOP_DOWN, GraspType.SIDE_APPROACH],
                'grasp_width': 0.06,
                'approach_height': 0.05,
                'material': {'friction': 0.7, 'weight': 0.2}
            },
            'orange': {
                'size_estimate': (0.09, 0.09, 0.09),
                'preferred_grasps': [GraspType.TOP_DOWN, GraspType.SIDE_APPROACH],
                'grasp_width': 0.07,
                'approach_height': 0.05,
                'material': {'friction': 0.6, 'weight': 0.25}
            },
            'bottle': {
                'size_estimate': (0.06, 0.06, 0.20),
                'preferred_grasps': [GraspType.SIDE_APPROACH, GraspType.POWER_GRASP],
                'grasp_width': 0.05,
                'approach_height': 0.1,
                'material': {'friction': 0.8, 'weight': 0.5}
            },
            'cup': {
                'size_estimate': (0.08, 0.08, 0.10),
                'preferred_grasps': [GraspType.SIDE_APPROACH, GraspType.POWER_GRASP],
                'grasp_width': 0.06,
                'approach_height': 0.05,
                'material': {'friction': 0.9, 'weight': 0.3}
            },
            'book': {
                'size_estimate': (0.20, 0.25, 0.03),
                'preferred_grasps': [GraspType.TOP_DOWN, GraspType.PINCH_GRASP],
                'grasp_width': 0.02,
                'approach_height': 0.02,
                'material': {'friction': 0.8, 'weight': 0.4}
            },
            'cell phone': {
                'size_estimate': (0.07, 0.15, 0.01),
                'preferred_grasps': [GraspType.TOP_DOWN, GraspType.PINCH_GRASP],
                'grasp_width': 0.005,
                'approach_height': 0.02,
                'material': {'friction': 0.7, 'weight': 0.2}
            },
            'bowl': {
                'size_estimate': (0.15, 0.15, 0.08),
                'preferred_grasps': [GraspType.SIDE_APPROACH, GraspType.POWER_GRASP],
                'grasp_width': 0.04,
                'approach_height': 0.05,
                'material': {'friction': 0.9, 'weight': 0.4}
            },
            # Default for unknown objects
            'unknown_object': {
                'size_estimate': (0.05, 0.05, 0.05),
                'preferred_grasps': [GraspType.TOP_DOWN],
                'grasp_width': 0.04,
                'approach_height': 0.03,
                'material': {'friction': 0.5, 'weight': 0.2}
            }
        }
    
    def plan_grasp_sequence(self, detected_objects: List, depth_image: Optional[np.ndarray] = None) -> List[ObjectGraspInfo]:
        """
        Plan grasp sequence for multiple objects
        
        Args:
            detected_objects: List of DetectedObject from YOLO
            depth_image: Optional depth image for 3D pose estimation
            
        Returns:
            List of objects with grasp plans
        """
        grasp_plans = []
        
        for obj in detected_objects:
            # Get object info from database
            obj_info = self.grasp_database.get(obj.class_name, 
                                             self.grasp_database['unknown_object'])
            
            # Estimate 3D pose
            center_3d = self._estimate_3d_position(obj, depth_image)
            
            # Generate grasp poses
            grasp_poses = self._generate_grasp_poses(obj, obj_info, center_3d)
            
            # Create grasp info
            grasp_info = ObjectGraspInfo(
                class_name=obj.class_name,
                bbox=obj.bbox,
                estimated_size=obj_info['size_estimate'],
                center_3d=center_3d,
                preferred_grasps=grasp_poses,
                material_properties=obj_info['material']
            )
            
            grasp_plans.append(grasp_info)
        
        # Sort by grasp difficulty (easier objects first)
        grasp_plans.sort(key=lambda x: self._calculate_grasp_difficulty(x))
        
        return grasp_plans
    
    def _estimate_3d_position(self, obj, depth_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate 3D position of object center"""
        if depth_image is not None:
            # Use depth image for accurate 3D positioning
            x, y, w, h = obj.bbox
            center_x, center_y = x + w//2, y + h//2
            
            # Get depth at object center (with some averaging for robustness)
            roi_size = min(w, h) // 4
            x1, y1 = max(0, center_x - roi_size), max(0, center_y - roi_size)
            x2, y2 = min(depth_image.shape[1], center_x + roi_size), min(depth_image.shape[0], center_y + roi_size)
            
            depth_roi = depth_image[y1:y2, x1:x2]
            valid_depths = depth_roi[depth_roi > 0]
            
            if len(valid_depths) > 0:
                avg_depth = np.median(valid_depths) / 1000.0  # Convert mm to meters
            else:
                avg_depth = 0.5  # Default depth
            
            # Convert pixel coordinates to world coordinates (simplified camera model)
            # This would normally use camera intrinsics
            world_x = 0.4 + (center_x - 320) * 0.001  # Rough conversion
            world_y = (center_y - 240) * 0.001
            world_z = avg_depth
            
            return np.array([world_x, world_y, world_z])
        else:
            # Fallback: estimate from bounding box size (realistic values)
            x, y, w, h = obj.bbox
            
            # Assume camera is at reasonable distance from objects
            estimated_depth = 0.4 + np.random.uniform(0, 0.2)  # 40-60cm depth
            
            # Convert pixels to world coordinates (realistic workspace)
            world_x = 0.3 + (x + w//2 - 320) * 0.0005  # Scale down
            world_y = (y + h//2 - 240) * 0.0005        # Scale down  
            world_z = 0.1 + (h / 480.0) * 0.2         # Height based on image position
            
            return np.array([world_x, world_y, world_z])
    
    def _generate_grasp_poses(self, obj, obj_info: Dict, center_3d: np.ndarray) -> List[GraspPose]:
        """Generate possible grasp poses for an object"""
        grasp_poses = []
        
        for grasp_type in obj_info['preferred_grasps']:
            if grasp_type == GraspType.TOP_DOWN:
                pose = self._generate_top_down_grasp(center_3d, obj_info)
            elif grasp_type == GraspType.SIDE_APPROACH:
                pose = self._generate_side_approach_grasp(center_3d, obj_info)
            elif grasp_type == GraspType.PINCH_GRASP:
                pose = self._generate_pinch_grasp(center_3d, obj_info)
            elif grasp_type == GraspType.POWER_GRASP:
                pose = self._generate_power_grasp(center_3d, obj_info)
            else:
                continue
            
            if pose and self._validate_grasp_pose(pose):
                grasp_poses.append(pose)
        
        return grasp_poses
    
    def _generate_top_down_grasp(self, center_3d: np.ndarray, obj_info: Dict) -> Optional[GraspPose]:
        """Generate top-down grasp pose"""
        # Position above object
        position = center_3d.copy()
        position[2] += obj_info['approach_height']
        
        # Orientation: gripper pointing down
        if SCIPY_AVAILABLE:
            # Z-axis down, Y-axis forward
            orientation_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
            orientation = Rotation.from_matrix(orientation_matrix).as_quat()
        else:
            # Simplified quaternion for downward orientation
            orientation = np.array([0.707, 0.707, 0, 0])  # 90-degree rotation around X
        
        # Approach from above
        approach_vector = np.array([0, 0, -1])
        
        return GraspPose(
            position=position,
            orientation=orientation,
            approach_vector=approach_vector,
            grasp_width=obj_info['grasp_width'],
            grasp_type=GraspType.TOP_DOWN,
            confidence=0.8
        )
    
    def _generate_side_approach_grasp(self, center_3d: np.ndarray, obj_info: Dict) -> Optional[GraspPose]:
        """Generate side approach grasp pose"""
        # Position to the side of object
        position = center_3d.copy()
        position[1] -= 0.1  # Approach from negative Y
        
        # Orientation: gripper pointing toward object
        if SCIPY_AVAILABLE:
            # X-axis toward object, Z-axis up
            orientation_matrix = np.array([
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, -1]
            ])
            orientation = Rotation.from_matrix(orientation_matrix).as_quat()
        else:
            # Simplified quaternion for side approach
            orientation = np.array([0.707, 0, 0, 0.707])  # 90-degree rotation around Z
        
        # Approach from side
        approach_vector = np.array([0, 1, 0])
        
        return GraspPose(
            position=position,
            orientation=orientation,
            approach_vector=approach_vector,
            grasp_width=obj_info['grasp_width'],
            grasp_type=GraspType.SIDE_APPROACH,
            confidence=0.7
        )
    
    def _generate_pinch_grasp(self, center_3d: np.ndarray, obj_info: Dict) -> Optional[GraspPose]:
        """Generate precision pinch grasp"""
        # Similar to top-down but with smaller opening
        position = center_3d.copy()
        position[2] += obj_info['approach_height']
        
        # Narrow grip width for precision
        grasp_width = min(obj_info['grasp_width'], 0.02)
        
        orientation = np.array([0.707, 0.707, 0, 0])
        approach_vector = np.array([0, 0, -1])
        
        return GraspPose(
            position=position,
            orientation=orientation,
            approach_vector=approach_vector,
            grasp_width=grasp_width,
            grasp_type=GraspType.PINCH_GRASP,
            confidence=0.6
        )
    
    def _generate_power_grasp(self, center_3d: np.ndarray, obj_info: Dict) -> Optional[GraspPose]:
        """Generate power grasp (full hand wrap)"""
        # Approach from side with wider grip
        position = center_3d.copy()
        position[1] -= 0.08
        
        # Wider grip for power grasp
        grasp_width = min(obj_info['grasp_width'] * 1.2, self.gripper_limits['max_width'])
        
        orientation = np.array([0.707, 0, 0, 0.707])
        approach_vector = np.array([0, 1, 0])
        
        return GraspPose(
            position=position,
            orientation=orientation,
            approach_vector=approach_vector,
            grasp_width=grasp_width,
            grasp_type=GraspType.POWER_GRASP,
            confidence=0.9
        )
    
    def _validate_grasp_pose(self, grasp_pose: GraspPose) -> bool:
        """Validate if grasp pose is reachable and safe"""
        # Check workspace bounds
        pos = grasp_pose.position
        if not (self.workspace_bounds['x'][0] <= pos[0] <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= pos[1] <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= pos[2] <= self.workspace_bounds['z'][1]):
            return False
        
        # Check gripper limits
        if not (self.gripper_limits['min_width'] <= grasp_pose.grasp_width <= 
                self.gripper_limits['max_width']):
            return False
        
        # Additional kinematic checks would go here if robot model available
        return True
    
    def _calculate_grasp_difficulty(self, grasp_info: ObjectGraspInfo) -> float:
        """Calculate relative difficulty of grasping an object"""
        if not grasp_info.preferred_grasps:
            return 1.0  # Maximum difficulty
        
        # Factors affecting difficulty
        best_confidence = max(pose.confidence for pose in grasp_info.preferred_grasps)
        size_factor = np.prod(grasp_info.estimated_size)  # Smaller = harder
        material_friction = grasp_info.material_properties.get('friction', 0.5)
        
        # Combined difficulty score (0 = easy, 1 = hard)
        difficulty = (1 - best_confidence) * 0.5 + (1 - min(size_factor * 10, 1)) * 0.3 + (1 - material_friction) * 0.2
        
        return difficulty
    
    def execute_grasp_plan(self, grasp_info: ObjectGraspInfo, robot_data=None) -> Dict[str, any]:
        """
        Execute grasp plan (simulation)
        
        Returns:
            Execution result with success rate and performance metrics
        """
        if not grasp_info.preferred_grasps:
            return {'success': False, 'reason': 'No valid grasp poses'}
        
        # Use best grasp pose
        best_pose = max(grasp_info.preferred_grasps, key=lambda x: x.confidence)
        
        # Simulate execution phases
        phases = {
            'approach': self._simulate_approach_phase(best_pose),
            'pre_grasp': self._simulate_pre_grasp_phase(best_pose),
            'grasp': self._simulate_grasp_phase(best_pose, grasp_info),
            'lift': self._simulate_lift_phase(best_pose, grasp_info)
        }
        
        # Overall success
        overall_success = all(phase['success'] for phase in phases.values())
        
        return {
            'success': overall_success,
            'grasp_type': best_pose.grasp_type.value,
            'confidence': best_pose.confidence,
            'phases': phases,
            'execution_time': sum(phase.get('time', 0) for phase in phases.values()),
            'object_info': {
                'class': grasp_info.class_name,
                'estimated_weight': grasp_info.material_properties.get('weight', 0.2)
            }
        }
    
    def _simulate_approach_phase(self, grasp_pose: GraspPose) -> Dict:
        """Simulate robot approach to pre-grasp position"""
        # Calculate pre-grasp position
        pre_grasp_pos = grasp_pose.position + grasp_pose.approach_vector * grasp_pose.pre_grasp_offset
        
        # Simulate movement (simplified)
        approach_distance = np.linalg.norm(pre_grasp_pos - np.array([0.4, 0, 0.3]))  # From home
        approach_time = approach_distance / 0.1  # 10 cm/s movement speed
        
        return {
            'success': True,
            'time': approach_time,
            'final_position': pre_grasp_pos
        }
    
    def _simulate_pre_grasp_phase(self, grasp_pose: GraspPose) -> Dict:
        """Simulate gripper pre-positioning"""
        return {
            'success': True,
            'time': 0.5,  # Time to open gripper
            'gripper_width': grasp_pose.grasp_width + 0.01  # Slightly wider
        }
    
    def _simulate_grasp_phase(self, grasp_pose: GraspPose, grasp_info: ObjectGraspInfo) -> Dict:
        """Simulate actual grasping motion"""
        # Success depends on grasp type and object properties
        success_probability = grasp_pose.confidence * grasp_info.material_properties.get('friction', 0.5)
        
        # Simulate some randomness
        success = np.random.random() < success_probability
        
        return {
            'success': success,
            'time': 1.0,  # Time for grasp closure
            'force': grasp_info.material_properties.get('weight', 0.2) * 9.81  # Required force
        }
    
    def _simulate_lift_phase(self, grasp_pose: GraspPose, grasp_info: ObjectGraspInfo) -> Dict:
        """Simulate lifting the object"""
        object_weight = grasp_info.material_properties.get('weight', 0.2)
        
        # Success depends on grasp quality and object weight
        success = object_weight < 1.0  # Robot can lift up to 1kg
        
        return {
            'success': success,
            'time': 2.0,  # Time to lift safely
            'lift_height': 0.1  # 10cm lift
        }

def main():
    """Demo of intelligent grasp planning"""
    print("🚀 Intelligent Grasp Planning Demo")
    print("=" * 50)
    
    # Create mock detected objects
    from types import SimpleNamespace
    
    mock_objects = [
        SimpleNamespace(
            class_name='apple',
            bbox=(100, 100, 80, 80),
            confidence=0.9,
            center=(140, 140)
        ),
        SimpleNamespace(
            class_name='bottle',
            bbox=(200, 150, 60, 120),
            confidence=0.8,
            center=(230, 210)
        ),
        SimpleNamespace(
            class_name='book',
            bbox=(300, 80, 150, 200),
            confidence=0.7,
            center=(375, 180)
        )
    ]
    
    # Initialize planner
    planner = IntelligentGraspPlanner()
    
    # Plan grasps
    print("📋 Planning grasps for detected objects...")
    grasp_plans = planner.plan_grasp_sequence(mock_objects)
    
    # Display results
    for i, plan in enumerate(grasp_plans):
        print(f"\n🎯 Object {i+1}: {plan.class_name}")
        print(f"   Estimated size: {plan.estimated_size}")
        print(f"   3D position: {plan.center_3d}")
        print(f"   Available grasps: {len(plan.preferred_grasps)}")
        
        for j, grasp in enumerate(plan.preferred_grasps):
            print(f"     Grasp {j+1}: {grasp.grasp_type.value} (confidence: {grasp.confidence:.2f})")
        
        # Execute best grasp
        if plan.preferred_grasps:
            print(f"   ⚡ Executing best grasp...")
            result = planner.execute_grasp_plan(plan)
            print(f"   ✅ Success: {result['success']}")
            if result['success']:
                print(f"   ⏱️  Execution time: {result['execution_time']:.1f}s")
    
    print("\n✅ Grasp planning demo completed")

if __name__ == "__main__":
    main() 