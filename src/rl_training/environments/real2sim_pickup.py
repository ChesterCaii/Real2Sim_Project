#!/usr/bin/env python3
"""
Real2Sim PickAndPlace Environment
Custom MuJoCo Playground environment for RL training with reconstructed objects
"""

import numpy as np
import mujoco
from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random
import os
import xml.etree.ElementTree as ET

# MuJoCo Playground imports
try:
    from mujoco_playground.envs.mjx_env import MjxEnv
    from mujoco_playground.utils import mjx_scene_utils
    PLAYGROUND_AVAILABLE = True
except ImportError:
    print("⚠️  MuJoCo Playground not available")
    PLAYGROUND_AVAILABLE = False

class Real2SimPickAndPlace(MjxEnv):
    """
    Real2Sim PickAndPlace environment for MuJoCo Playground
    
    This environment:
    1. Uses Franka Panda robot from your current setup
    2. Integrates reconstructed objects from Real2Sim pipeline
    3. Provides pick-and-place task with dense/sparse rewards
    4. Supports domain randomization for sim-to-real transfer
    """
    
    def __init__(
        self,
        reconstructed_objects_dir: str = "reconstructed_objects",
        reward_type: str = "sparse",  # "sparse" or "dense"
        max_episode_steps: int = 200,
        object_randomization: bool = True,
        physics_randomization: bool = True,
        **kwargs
    ):
        """Initialize Real2Sim PickAndPlace environment"""
        
        self.reconstructed_objects_dir = reconstructed_objects_dir
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps
        self.object_randomization = object_randomization
        self.physics_randomization = physics_randomization
        
        # Load and prepare scene XML
        scene_xml = self._create_scene_xml()
        
        # Initialize parent class
        super().__init__(
            model_path=scene_xml,
            max_episode_steps=max_episode_steps,
            **kwargs
        )
        
        # Task-specific setup
        self._setup_task_parameters()
        
        print(f"✅ Real2SimPickAndPlace environment created")
        print(f"   📁 Objects dir: {reconstructed_objects_dir}")
        print(f"   🎯 Reward type: {reward_type}")
        print(f"   🔄 Max steps: {max_episode_steps}")
    
    def _create_scene_xml(self) -> str:
        """Create MJX-compatible scene XML with Franka + reconstructed objects"""
        
        # Base template for Franka Panda scene
        scene_template = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="real2sim_pickup">
    <compiler angle="radian" meshdir="." autolimits="true"/>
    
    <option timestep="0.002" solver="Newton" tolerance="1e-6"/>
    
    <asset>
        <!-- Table -->
        <mesh name="table" file="table.stl" scale="0.8 0.8 0.4"/>
        
        <!-- Load reconstructed objects dynamically -->
        {object_meshes}
    </asset>
    
    <worldbody>
        <!-- Include Franka robot -->
        <include file="mujoco_menagerie/franka_emika_panda/panda_nohand.xml"/>
        
        <!-- Add gripper -->
        <body name="gripper" pos="0 0 0">
            <include file="mujoco_menagerie/robotiq_2f85/2f85.xml"/>
        </body>
        
        <!-- Table -->
        <body name="table" pos="0.5 0 0.2">
            <geom name="table_geom" type="mesh" mesh="table" rgba="0.8 0.6 0.4 1"/>
        </body>
        
        <!-- Target object (reconstructed from Real2Sim) -->
        <body name="target_object" pos="0.6 0.1 0.45">
            <joint name="object_freejoint" type="free"/>
            <geom name="object_geom" type="mesh" mesh="reconstructed_object" 
                  rgba="0.2 0.7 0.3 1" mass="0.1"/>
        </body>
        
        <!-- Goal position marker -->
        <body name="goal" pos="0.3 0.3 0.45">
            <geom name="goal_geom" type="sphere" size="0.02" 
                  rgba="1.0 0.0 0.0 0.3" contype="0" conaffinity="0"/>
        </body>
        
        <!-- Floor -->
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
    </worldbody>
    
    <actuator>
        <!-- Franka arm actuators -->
        <motor name="panda_joint1" joint="panda_joint1" gear="100"/>
        <motor name="panda_joint2" joint="panda_joint2" gear="100"/>
        <motor name="panda_joint3" joint="panda_joint3" gear="100"/>
        <motor name="panda_joint4" joint="panda_joint4" gear="100"/>
        <motor name="panda_joint5" joint="panda_joint5" gear="100"/>
        <motor name="panda_joint6" joint="panda_joint6" gear="100"/>
        <motor name="panda_joint7" joint="panda_joint7" gear="100"/>
        
        <!-- Gripper actuators -->
        <motor name="gripper_finger1" joint="right_outer_knuckle_joint" gear="50"/>
        <motor name="gripper_finger2" joint="left_outer_knuckle_joint" gear="50"/>
    </actuator>
    
    <sensor>
        <!-- End-effector position -->
        <framepos name="ee_pos" objtype="site" objname="attachment_site"/>
        <framequat name="ee_quat" objtype="site" objname="attachment_site"/>
        
        <!-- Object position -->
        <framepos name="obj_pos" objtype="body" objname="target_object"/>
        
        <!-- Goal position -->
        <framepos name="goal_pos" objtype="body" objname="goal"/>
    </sensor>
</mujoco>
"""
        
        # Get available reconstructed objects
        object_meshes = self._get_object_meshes()
        
        # Fill in template
        scene_xml = scene_template.format(object_meshes=object_meshes)
        
        # Save to temporary file
        scene_path = "temp_real2sim_scene.xml"
        with open(scene_path, 'w') as f:
            f.write(scene_xml)
        
        return scene_path
    
    def _get_object_meshes(self) -> str:
        """Load available reconstructed object meshes"""
        mesh_xml = ""
        
        # Check for reconstructed objects
        if os.path.exists(self.reconstructed_objects_dir):
            stl_files = [f for f in os.listdir(self.reconstructed_objects_dir) 
                        if f.endswith('.stl')]
            
            if stl_files:
                # Use first available reconstructed object
                object_file = stl_files[0]
                object_path = os.path.join(self.reconstructed_objects_dir, object_file)
                mesh_xml = f'<mesh name="reconstructed_object" file="{object_path}"/>'
                print(f"📦 Using reconstructed object: {object_file}")
            else:
                print("⚠️  No reconstructed objects found, using default cube")
        
        # Fallback to simple cube if no reconstructed objects
        if not mesh_xml:
            mesh_xml = '<mesh name="reconstructed_object" vertex="0 0 0  0.05 0 0  0.05 0.05 0  0 0.05 0  0 0 0.05  0.05 0 0.05  0.05 0.05 0.05  0 0.05 0.05" face="0 1 2  0 2 3  4 5 6  4 6 7  0 1 5  0 5 4  2 3 7  2 7 6  0 3 7  0 7 4  1 2 6  1 6 5"/>'
        
        return mesh_xml
    
    def _setup_task_parameters(self):
        """Setup task-specific parameters"""
        
        # Observation and action spaces (will be set by parent class)
        self.observation_dim = 25  # Similar to FetchPickAndPlace
        self.action_dim = 9  # 7 arm joints + 2 gripper fingers
        
        # Task parameters
        self.distance_threshold = 0.05  # Success threshold (5cm)
        self.reward_scale = 1.0
        
        # Robot control parameters
        self.max_velocity = 1.0
        self.position_control = True
        
        # Domain randomization ranges
        self.object_mass_range = (0.05, 0.2)  # kg
        self.friction_range = (0.5, 1.5)
        self.object_size_range = (0.8, 1.2)  # Scale factor
    
    def reset(self, rng: jax.Array) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Reset environment for new episode"""
        
        # Reset parent environment
        obs, info = super().reset(rng)
        
        # Apply domain randomization
        if self.object_randomization or self.physics_randomization:
            self._apply_domain_randomization(rng)
        
        # Set random object and goal positions
        rng, obj_rng, goal_rng = random.split(rng, 3)
        
        # Random object position on table
        obj_pos = jnp.array([
            0.5 + random.uniform(obj_rng, minval=-0.15, maxval=0.15),
            random.uniform(obj_rng, minval=-0.15, maxval=0.15),
            0.45  # Table height + object
        ])
        
        # Random goal position on table
        goal_pos = jnp.array([
            0.5 + random.uniform(goal_rng, minval=-0.15, maxval=0.15),
            random.uniform(goal_rng, minval=-0.15, maxval=0.15),
            0.45
        ])
        
        # Update MuJoCo data with new positions
        self.data = self.data.replace(
            qpos=self.data.qpos.at[self._get_object_qpos_idx()].set(
                jnp.concatenate([obj_pos, jnp.array([1., 0., 0., 0.])])  # pos + quat
            )
        )
        
        # Store goal for reward computation
        self.goal_pos = goal_pos
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in environment"""
        
        # Clip actions to valid range
        action = jnp.clip(action, -1.0, 1.0)
        
        # Convert actions to joint targets
        if self.position_control:
            # Position control mode
            joint_targets = self._action_to_joint_targets(action)
            
            # Apply PD control
            ctrl = self._pd_control(joint_targets)
        else:
            # Direct torque control
            ctrl = action * self.max_velocity
        
        # Apply control and step physics
        self.data = self.data.replace(ctrl=ctrl)
        
        # Step parent environment
        obs, reward, done, info = super().step(action)
        
        # Compute task-specific reward
        reward = self._compute_reward()
        
        # Check success condition
        success = self._is_success()
        done = done or success
        
        # Update info
        info.update({
            'is_success': success,
            'distance_to_goal': self._get_distance_to_goal(),
            'object_pos': self._get_object_pos(),
            'goal_pos': self.goal_pos
        })
        
        return obs, reward, done, info
    
    def _get_observation(self) -> jnp.ndarray:
        """Get current observation"""
        
        # End-effector state
        ee_pos = self._get_site_pos("attachment_site")
        ee_vel = jnp.zeros(3)  # Placeholder for velocity
        
        # Object state
        obj_pos = self._get_object_pos()
        obj_vel = jnp.zeros(3)  # Placeholder for velocity
        
        # Relative object position
        rel_obj_pos = obj_pos - ee_pos
        
        # Gripper state
        gripper_pos = self._get_gripper_pos()
        
        # Goal information
        goal_pos = self.goal_pos
        achieved_goal = obj_pos
        desired_goal = goal_pos
        
        # Combine all observations
        obs = jnp.concatenate([
            ee_pos,           # 3
            obj_pos,          # 3
            rel_obj_pos,      # 3
            gripper_pos,      # 2
            obj_vel,          # 3
            ee_vel,           # 3
            achieved_goal,    # 3
            desired_goal,     # 3
            # Additional state info
            jnp.array([self._get_distance_to_goal()])  # 1
        ])
        
        return obs
    
    def _compute_reward(self) -> float:
        """Compute reward based on current state"""
        
        distance = self._get_distance_to_goal()
        
        if self.reward_type == "sparse":
            # Sparse reward: +1 for success, -1 otherwise
            return 1.0 if distance < self.distance_threshold else -1.0
        
        elif self.reward_type == "dense":
            # Dense reward: negative distance to goal
            reward = -distance
            
            # Bonus for grasping
            if self._is_grasping():
                reward += 0.1
            
            # Large bonus for success
            if distance < self.distance_threshold:
                reward += 10.0
            
            return reward
        
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _get_distance_to_goal(self) -> float:
        """Get distance from object to goal"""
        obj_pos = self._get_object_pos()
        return jnp.linalg.norm(obj_pos - self.goal_pos)
    
    def _is_success(self) -> bool:
        """Check if task is successfully completed"""
        return self._get_distance_to_goal() < self.distance_threshold
    
    def _is_grasping(self) -> bool:
        """Check if robot is grasping the object"""
        # Simple heuristic: gripper is closed and object is near end-effector
        gripper_distance = jnp.sum(self._get_gripper_pos())  # Sum of finger positions
        ee_pos = self._get_site_pos("attachment_site")
        obj_pos = self._get_object_pos()
        ee_obj_distance = jnp.linalg.norm(ee_pos - obj_pos)
        
        return (gripper_distance < 0.02) and (ee_obj_distance < 0.1)
    
    def _apply_domain_randomization(self, rng: jax.Array):
        """Apply domain randomization for sim-to-real transfer"""
        
        if self.physics_randomization:
            # Randomize object mass
            rng, mass_rng = random.split(rng)
            new_mass = random.uniform(
                mass_rng, 
                minval=self.object_mass_range[0],
                maxval=self.object_mass_range[1]
            )
            
            # Randomize friction
            rng, friction_rng = random.split(rng)
            new_friction = random.uniform(
                friction_rng,
                minval=self.friction_range[0], 
                maxval=self.friction_range[1]
            )
            
            # Note: In full implementation, these would update the MuJoCo model
    
    # Helper methods for accessing MuJoCo data
    def _get_object_pos(self) -> jnp.ndarray:
        """Get object position"""
        return self.data.xpos[self._get_body_id("target_object")]
    
    def _get_site_pos(self, site_name: str) -> jnp.ndarray:
        """Get site position"""
        site_id = self.model.site(site_name).id
        return self.data.site_xpos[site_id]
    
    def _get_body_id(self, body_name: str) -> int:
        """Get body ID by name"""
        return self.model.body(body_name).id
    
    def _get_object_qpos_idx(self) -> slice:
        """Get qpos indices for object free joint"""
        # Object has 7 DOF (3 pos + 4 quat)
        start_idx = 7  # After 7 robot joints
        return slice(start_idx, start_idx + 7)
    
    def _get_gripper_pos(self) -> jnp.ndarray:
        """Get gripper finger positions"""
        # Return positions of gripper joints
        return self.data.qpos[7:9]  # Assume gripper joints are after arm
    
    def _action_to_joint_targets(self, action: jnp.ndarray) -> jnp.ndarray:
        """Convert normalized actions to joint targets"""
        # First 7 actions control arm joints
        arm_actions = action[:7]
        # Last 2 actions control gripper
        gripper_actions = action[7:9]
        
        # Scale to joint limits (simplified)
        arm_targets = arm_actions * 1.0  # Placeholder scaling
        gripper_targets = gripper_actions * 0.04  # Gripper range
        
        return jnp.concatenate([arm_targets, gripper_targets])
    
    def _pd_control(self, targets: jnp.ndarray) -> jnp.ndarray:
        """Simple PD control for joint tracking"""
        kp = 1000.0  # Proportional gain
        kd = 100.0   # Derivative gain
        
        pos_error = targets - self.data.qpos[:9]
        vel_error = -self.data.qvel[:9]  # Target velocity is 0
        
        torques = kp * pos_error + kd * vel_error
        return jnp.clip(torques, -100.0, 100.0)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info dictionary"""
        return {
            'object_pos': self._get_object_pos(),
            'goal_pos': self.goal_pos,
            'distance_to_goal': self._get_distance_to_goal(),
            'is_grasping': self._is_grasping(),
        }


def make_real2sim_env(
    reconstructed_objects_dir: str = "reconstructed_objects",
    reward_type: str = "sparse",
    num_envs: int = 1,
    **kwargs
) -> Real2SimPickAndPlace:
    """
    Factory function to create Real2Sim PickAndPlace environment
    
    Args:
        reconstructed_objects_dir: Directory with reconstructed STL files
        reward_type: "sparse" or "dense" rewards
        num_envs: Number of parallel environments
        **kwargs: Additional environment arguments
    
    Returns:
        Real2SimPickAndPlace environment
    """
    
    if not PLAYGROUND_AVAILABLE:
        raise ImportError("MuJoCo Playground not available. Run setup_mujoco_playground.py")
    
    return Real2SimPickAndPlace(
        reconstructed_objects_dir=reconstructed_objects_dir,
        reward_type=reward_type,
        **kwargs
    )


# Test the environment
if __name__ == "__main__":
    print("🧪 Testing Real2Sim PickAndPlace Environment")
    
    try:
        env = make_real2sim_env(reward_type="dense")
        
        rng = jax.random.PRNGKey(0)
        obs, info = env.reset(rng)
        
        print(f"✅ Environment created successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_dim}")
        
        # Test a few steps
        for i in range(5):
            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(action_rng, (env.action_dim,), minval=-1, maxval=1)
            
            obs, reward, done, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.3f}, done={done}")
            
            if done:
                obs, info = env.reset(rng)
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 