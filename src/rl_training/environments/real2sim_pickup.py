#!/usr/bin/env python3
"""
Real2Sim PickAndPlace Environment

This module implements a custom MuJoCo Playground environment for training
pick-and-place policies using objects reconstructed from the Real2Sim pipeline.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import jax
    import jax.numpy as jnp
    import mujoco_playground
    # Use attribute access instead of direct import
    cartpole = mujoco_playground.dm_control_suite.cartpole
except ImportError as e:
    print(f"ERROR: MuJoCo Playground not available: {e}")
    print("Run: python setup_mujoco_playground.py")
    sys.exit(1)

class Real2SimPickAndPlace:
    """
    Real2Sim PickAndPlace Environment
    
    A custom environment that extends MuJoCo Playground for training pick-and-place
    policies on reconstructed objects from the Real2Sim pipeline.
    
    For now, this is implemented as a wrapper around the cartpole environment
    until we can properly integrate the Franka robot model.
    """
    
    def __init__(
        self,
        num_parallel_envs: int = 1024,
        max_episode_steps: int = 200,
        reward_config: Optional[Dict[str, Any]] = None,
        domain_randomization: Optional[Dict[str, Any]] = None,
        reconstructed_objects_dir: str = "outputs/reconstructed_objects"
    ):
        """Initialize the Real2Sim PickAndPlace environment"""
        
        self.num_parallel_envs = num_parallel_envs
        self.max_episode_steps = max_episode_steps
        self.reconstructed_objects_dir = Path(reconstructed_objects_dir)
        
        # For now, use cartpole as a placeholder environment
        # TODO: Integrate Franka robot model for actual pick-and-place
        self.base_env = cartpole.Balance(swing_up=False, sparse=False)
        
        # Environment properties
        self.action_size = self.base_env.action_size
        self.observation_space = type('ObsSpace', (), {
            'shape': (self.base_env.observation_size,)
        })()
        self.action_space = type('ActSpace', (), {
            'shape': (self.action_size,)
        })()
        
        # Reward configuration
        self.reward_config = reward_config or {
            'sparse_reward': False,
            'success_threshold': 0.05,
            'success_reward': 10.0,
            'distance_weight': -1.0
        }
        
        # Domain randomization (placeholder)
        self.domain_randomization = domain_randomization or {
            'enabled': False,
            'mass_range': [0.8, 1.2],
            'friction_range': [0.8, 1.2]
        }
        
        print(f"Real2Sim PickAndPlace Environment initialized")
        print(f"  Base environment: CartPole (placeholder)")
        print(f"  Parallel environments: {num_parallel_envs}")
        print(f"  Max episode steps: {max_episode_steps}")
        print(f"  Action size: {self.action_size}")
        print(f"  Observation size: {self.base_env.observation_size}")
        
        # Check for reconstructed objects
        self._check_reconstructed_objects()
    
    def _check_reconstructed_objects(self):
        """Check for available reconstructed objects"""
        if self.reconstructed_objects_dir.exists():
            stl_files = list(self.reconstructed_objects_dir.glob("*.stl"))
            print(f"  Found {len(stl_files)} reconstructed objects")
            if stl_files:
                print(f"  Objects: {[f.name for f in stl_files[:3]]}{'...' if len(stl_files) > 3 else ''}")
        else:
            print(f"  No reconstructed objects found at {self.reconstructed_objects_dir}")
            print("  Will use default simulation objects")
    
    def reset(self, key: jax.Array):
        """Reset the environment"""
        return self.base_env.reset(key)
    
    def step(self, state, action):
        """Step the environment"""
        return self.base_env.step(state, action)
    
    def _compute_reward(self, state, action, next_state) -> float:
        """Compute reward for the current step"""
        # Placeholder reward computation
        # TODO: Implement actual pick-and-place reward function
        
        if self.reward_config['sparse_reward']:
            # Sparse reward: +1 for success, -1 for failure
            success = self._check_success(next_state)
            return 1.0 if success else -1.0
        else:
            # Dense reward based on cartpole balance (placeholder)
            # In real implementation, this would be distance to target
            cart_pos = next_state.pipeline_state.qpos[0]
            pole_angle = next_state.pipeline_state.qpos[1]
            
            # Reward for keeping pole upright and cart centered
            angle_reward = np.cos(pole_angle)
            position_reward = 1.0 - abs(cart_pos) / 2.4
            
            return angle_reward + position_reward
    
    def _check_success(self, state) -> bool:
        """Check if the task was completed successfully"""
        # Placeholder success check
        # TODO: Implement actual pick-and-place success criteria
        
        # For cartpole, success is keeping the pole upright
        pole_angle = state.pipeline_state.qpos[1]
        return abs(pole_angle) < 0.2  # Within ~11 degrees
    
    def _apply_domain_randomization(self, key: jax.Array):
        """Apply domain randomization if enabled"""
        if not self.domain_randomization['enabled']:
            return
        
        # Placeholder for domain randomization
        # TODO: Implement actual physics parameter randomization
        pass
    
    def get_state_info(self, state) -> Dict[str, Any]:
        """Get information about the current state"""
        return {
            'qpos': state.pipeline_state.qpos,
            'qvel': state.pipeline_state.qvel,
            'time': state.time,
            'step': state.step
        }

def create_environment(config: Optional[Dict[str, Any]] = None) -> Real2SimPickAndPlace:
    """Create a Real2Sim PickAndPlace environment with given configuration"""
    
    config = config or {}
    
    return Real2SimPickAndPlace(
        num_parallel_envs=config.get('num_parallel_envs', 1024),
        max_episode_steps=config.get('max_episode_steps', 200),
        reward_config=config.get('reward_config', None),
        domain_randomization=config.get('domain_randomization', None),
        reconstructed_objects_dir=config.get('reconstructed_objects_dir', "outputs/reconstructed_objects")
    )

if __name__ == "__main__":
    """Test the environment"""
    print("Testing Real2Sim PickAndPlace Environment")
    print("=" * 50)
    
    # Create environment
    env = create_environment()
    
    # Test basic functionality
    key = jax.random.PRNGKey(42)
    
    print("\nTesting reset...")
    state = env.reset(key)
    print(f"Initial state type: {type(state)}")
    
    print("\nTesting step...")
    for i in range(5):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (env.action_size,))
        state = env.step(state, action)
        
        info = env.get_state_info(state)
        reward = env._compute_reward(state, action, state)
        
        print(f"Step {i+1}: reward={reward:.3f}")
    
    print("\nEnvironment test completed successfully!") 