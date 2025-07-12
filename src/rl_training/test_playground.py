#!/usr/bin/env python3
"""
Test MuJoCo Playground Installation and Performance

This script verifies that MuJoCo Playground is properly installed and 
benchmarks GPU performance for parallel training.
"""

import time
import numpy as np

def test_basic_imports():
    print("Testing basic imports...")
    
    try:
        import jax
        print(f"JAX: {jax.__version__} (backend: {jax.default_backend()})")
    except ImportError as e:
        print(f"ERROR: JAX import failed: {e}")
        return False
        
    try:
        import mujoco_playground
        print("MuJoCo Playground imported")
    except ImportError as e:
        print(f"ERROR: MuJoCo Playground import failed: {e}")
        return False
        
    return True

def test_environment_creation():
    print("\nTesting environments...")
    
    try:
        import jax
        import mujoco_playground
        
        # Test simple environment creation
        env = mujoco_playground.dm_control_suite.cartpole.Balance(swing_up=False, sparse=False)
        
        key = jax.random.PRNGKey(0)
        
        # Test reset
        state = env.reset(key)
        print("Environment creation successful")
        
        # Test a few steps
        for _ in range(5):
            key, subkey = jax.random.split(key)
            action = jax.random.uniform(subkey, (env.action_size,))
            state = env.step(state, action)
            
        return True
        
    except Exception as e:
        print(f"ERROR: Environment test failed: {e}")
        return False

def test_manipulation_environments():
    print("\nTesting manipulation environments...")
    
    import mujoco_playground
    
    # Test different environment types
    test_envs = [
        ("cartpole.Balance", lambda: mujoco_playground.dm_control_suite.cartpole.Balance(swing_up=False, sparse=False)),
        ("cartpole.Balance (swing_up)", lambda: mujoco_playground.dm_control_suite.cartpole.Balance(swing_up=True, sparse=False)),
        ("acrobot.Balance", lambda: mujoco_playground.dm_control_suite.acrobot.Balance(sparse=False)),
        ("pendulum.Balance", lambda: mujoco_playground.dm_control_suite.pendulum.Balance(sparse=False)),
    ]
    
    available_envs = []
    
    for env_name, env_constructor in test_envs:
        try:
            env = env_constructor()
            print(f"{env_name}: Available")
            available_envs.append(env_name)
        except Exception as e:
            print(f"WARNING: {env_name}: Not available ({str(e)[:50]}...)")
    
    return len(available_envs) > 0

def test_gpu_performance():
    print("\nTesting GPU performance...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Test GPU computation
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1000, 1000))
        
        start_time = time.time()
        
        # Matrix multiplication on GPU
        for _ in range(100):
            x = jnp.dot(x, x.T)
            x = x.block_until_ready()  # Ensure computation completes
            
        gpu_time = time.time() - start_time
        print(f"GPU computation test: {gpu_time:.4f}s")
        
        if gpu_time < 1.0:
            print("GPU acceleration ready for parallel training!")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"ERROR: GPU test failed: {e}")
        return False

def benchmark_parallel_environments():
    """Benchmark parallel environment performance"""
    print("\nBenchmarking parallel environments...")
    
    try:
        import jax
        import mujoco_playground
        
        batch_sizes = [1, 16, 64, 256]
        
        for batch_size in batch_sizes:
            env = mujoco_playground.dm_control_suite.cartpole.Balance(swing_up=False, sparse=False)
            
            # Vectorize environment functions
            reset_fn = jax.jit(jax.vmap(env.reset))
            step_fn = jax.jit(jax.vmap(env.step))
            
            # Generate random keys
            key = jax.random.PRNGKey(0)
            keys = jax.random.split(key, batch_size)
            
            # Benchmark reset
            start_time = time.time()
            states = reset_fn(keys)
            states = jax.tree_map(lambda x: x.block_until_ready(), states)
            reset_time = time.time() - start_time
            
            # Benchmark steps
            start_time = time.time()
            for _ in range(100):
                key, subkey = jax.random.split(key)
                actions = jax.random.uniform(subkey, (batch_size, env.action_size))
                states = step_fn(states, actions)
                states = jax.tree_map(lambda x: x.block_until_ready(), states)
            step_time = time.time() - start_time
            
            fps = (batch_size * 100) / step_time
            print(f"Batch {batch_size:4d}: {fps:8.0f} FPS")
            
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")

def main():
    print("MuJoCo Playground Installation Test")
    print("=" * 40)
    
    tests = [
        ("Import", test_basic_imports),
        ("Environment", test_environment_creation),
        ("Manipulation", test_manipulation_environments),
        ("GPU", test_gpu_performance),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n[{test_name} Test]")
            results[test_name] = test_func()
        except Exception as e:
            print(f"ERROR: {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    if all(results.values()):
        print("All tests passed! MuJoCo Playground is ready.")
        benchmark_parallel_environments()
    else:
        print("Some tests failed. Check installation.")
        print("Try running: python setup_mujoco_playground.py")

if __name__ == "__main__":
    main() 