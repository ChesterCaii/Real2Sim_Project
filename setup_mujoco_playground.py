#!/usr/bin/env python3
"""
MuJoCo Playground Setup for Real2Sim RL Training

This script installs and configures MuJoCo Playground (MJX) for the Real2Sim project.
Includes JAX CUDA support for GPU acceleration.
"""

import subprocess
import sys
import os

def check_gpu():
    """Check if GPU acceleration is available"""
    try:
        import jax
        return len(jax.devices('gpu')) > 0
    except ImportError:
        return False
    except Exception:
        return False

def install_mujoco_playground():
    print("Installing MuJoCo Playground...")
    
    # Check if we have GPU support
    has_gpu = check_gpu()
    
    # Install JAX with CUDA support if GPU available
    if has_gpu:
        print("Installing JAX with CUDA support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "jax[cuda12]", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        ], check=True)
    else:
        print("Installing JAX (CPU only)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "jax"
        ], check=True)
    
    # Install MuJoCo Playground (correct package name is 'playground')
    print("Installing MuJoCo Playground...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--upgrade", "playground"
    ], check=True)
    
    # Install additional RL dependencies
    print("Installing RL training dependencies...")
    rl_packages = [
        "stable-baselines3[extra]>=2.0.0",
        "tensorboard>=2.14.0",
        "wandb>=0.15.0",
        "gymnasium>=0.29.0",
        "dm-tree>=0.1.8",
        "optax>=0.1.7",
        "flax>=0.7.0",
        "chex>=0.1.7",
        "mediapy>=1.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0"
    ]
    
    for package in rl_packages:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

def verify_installation():
    print("\nVerifying installation...")
    
    try:
        import jax
        print(f"JAX: {jax.__version__} (backend: {jax.default_backend()})")
    except ImportError:
        print("ERROR: JAX import failed")
        return False
    
    try:
        import mujoco_playground
        print("MuJoCo Playground imported successfully")
    except ImportError:
        print("ERROR: MuJoCo Playground import failed")
        return False
    
    try:
        from mujoco_playground import envs
        print("Playground environments accessible")
    except Exception as e:
        print(f"WARNING: Playground test failed: {e}")
    
    return True

def main():
    print("Setting up MuJoCo Playground for Real2Sim RL Training")
    print("=" * 60)
    
    try:
        install_mujoco_playground()
        
        if verify_installation():
            print("\nSetup complete!")
            print("You can now run:")
            print("  python src/rl_training/test_playground.py")
            print("  python run_pipeline.py 6  # RL setup")
            print("  python run_pipeline.py 7  # RL training")
        else:
            print("\nSetup had issues. Please check error messages above.")
    
    except Exception as e:
        print(f"ERROR: Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 