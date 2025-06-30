#!/usr/bin/env python3
"""
Test Camera Integration - Phase 4
Test individual components of the camera integration system
"""

import cv2
import numpy as np
import time
import sys
import argparse

def test_camera_availability():
    """Test what cameras are available on the system"""
    print("🔍 Testing camera availability...")
    
    # Test Intel RealSense
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            pipeline.start(config)
            print("✅ Intel RealSense camera detected")
            pipeline.stop()
            realsense_available = True
        except RuntimeError:
            print("❌ Intel RealSense camera not connected")
            realsense_available = False
            
    except ImportError:
        print("❌ Intel RealSense SDK not installed")
        realsense_available = False
    
    # Test webcam
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("✅ Webcam detected")
            webcam_available = True
        else:
            print("❌ Webcam not working")
            webcam_available = False
        cap.release()
    else:
        print("❌ No webcam detected")
        webcam_available = False
    
    return realsense_available, webcam_available

def test_basic_camera_feed(camera_type="auto"):
    """Test basic camera feed display"""
    print(f"📹 Testing basic camera feed ({camera_type})...")
    
    if camera_type == "realsense":
        try:
            import pyrealsense2 as rs
            
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            pipeline.start(config)
            align = rs.align(rs.stream.color)
            
            print("✅ RealSense camera started. Press 'q' to quit.")
            
            while True:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Normalize depth for display
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                    )
                    
                    # Stack images horizontally
                    combined = np.hstack((color_image, depth_colormap))
                    cv2.imshow('RealSense RGB-D Feed', combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pipeline.stop()
            
        except Exception as e:
            print(f"❌ RealSense test failed: {e}")
            return False
    
    else:  # webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        print("✅ Webcam started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Create synthetic depth map
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            depth_synthetic = cv2.applyColorMap(255 - blurred, cv2.COLORMAP_JET)
            
            # Stack images horizontally
            combined = np.hstack((frame, depth_synthetic))
            cv2.imshow('Webcam + Synthetic Depth', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    
    cv2.destroyAllWindows()
    print("✅ Basic camera test completed")
    return True

def test_object_detection():
    """Test object detection capabilities"""
    print("🔍 Testing object detection...")
    
    # Test OpenCV background subtraction
    print("Testing OpenCV background subtraction...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera for detection test")
        return False
    
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    print("✅ Move objects in front of camera. Press 'q' to quit.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw detection results
        result_frame = frame.copy()
        object_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_frame, f"Object {object_count}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                object_count += 1
        
        # Add status text
        status_text = f"Frame: {frame_count} | Objects: {object_count}"
        cv2.putText(result_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show results
        cv2.imshow('Object Detection Test', result_frame)
        cv2.imshow('Foreground Mask', fg_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Object detection test completed")
    return True

def test_sam_integration():
    """Test SAM (Segment Anything Model) integration"""
    print("🤖 Testing SAM integration...")
    
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        import torch
        print("✅ SAM libraries available")
    except ImportError:
        print("❌ SAM not installed. Install with: pip install segment-anything")
        return False
    
    # Check for SAM checkpoint
    import os
    checkpoint_path = "sam_vit_b.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ SAM checkpoint not found: {checkpoint_path}")
        print("💡 Download from: https://github.com/facebookresearch/segment-anything")
        return False
    
    try:
        # Load SAM model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading SAM model on {device}...")
        
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device=device)
        
        mask_generator = SamAutomaticMaskGenerator(sam)
        print("✅ SAM model loaded successfully")
        
        # Test with camera feed
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera for SAM test")
            return False
        
        print("📸 SAM segmentation test. Press 's' to segment, 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('SAM Test - Press S to segment', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("🔄 Running SAM segmentation...")
                start_time = time.time()
                
                # Convert BGR to RGB for SAM
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Generate masks
                masks = mask_generator.generate(rgb_frame)
                
                elapsed = time.time() - start_time
                print(f"✅ SAM found {len(masks)} objects in {elapsed:.2f}s")
                
                # Visualize results
                result_frame = frame.copy()
                for i, mask_data in enumerate(masks[:5]):  # Show top 5
                    mask = mask_data['segmentation']
                    color = np.random.randint(0, 255, 3)
                    result_frame[mask] = result_frame[mask] * 0.7 + color * 0.3
                
                cv2.imshow('SAM Segmentation Results', result_frame)
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ SAM integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ SAM test failed: {e}")
        return False

def test_3d_reconstruction():
    """Test 3D reconstruction capabilities"""
    print("🔧 Testing 3D reconstruction...")
    
    try:
        import open3d as o3d
        print("✅ Open3D available")
    except ImportError:
        print("❌ Open3D not installed")
        return False
    
    # Create a simple test point cloud
    print("Creating test point cloud...")
    
    # Generate random points
    points = np.random.rand(1000, 3)
    points[:, 2] *= 0.5  # Flatten Z dimension
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors
    colors = np.random.rand(1000, 3)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    pcd.estimate_normals()
    
    print("✅ Point cloud created with 1000 points")
    
    # Test mesh reconstruction
    print("Testing Poisson reconstruction...")
    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        print(f"✅ Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    except Exception as e:
        print(f"❌ Poisson reconstruction failed: {e}")
        return False
    
    # Test ball pivoting
    print("Testing ball pivoting reconstruction...")
    try:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2]
        
        mesh_bp = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        print(f"✅ Ball pivoting mesh: {len(mesh_bp.vertices)} vertices, {len(mesh_bp.triangles)} triangles")
    except Exception as e:
        print(f"⚠️  Ball pivoting failed: {e}")
    
    print("✅ 3D reconstruction test completed")
    return True

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Camera Integration Components")
    parser.add_argument('--test', choices=['all', 'camera', 'feed', 'detection', 'sam', '3d'], 
                       default='all', help='Which test to run')
    parser.add_argument('--camera', choices=['auto', 'realsense', 'webcam'], 
                       default='auto', help='Camera type to test')
    
    args = parser.parse_args()
    
    print("============================================================")
    print("🧪 CAMERA INTEGRATION TESTS - PHASE 4")
    print("============================================================")
    
    success_count = 0
    total_tests = 0
    
    if args.test in ['all', 'camera']:
        print("\n📋 Test 1: Camera Availability")
        realsense_ok, webcam_ok = test_camera_availability()
        if realsense_ok or webcam_ok:
            success_count += 1
        total_tests += 1
    
    if args.test in ['all', 'feed']:
        print("\n📋 Test 2: Basic Camera Feed")
        camera_type = args.camera
        if camera_type == 'auto':
            # Auto-detect best camera
            realsense_ok, webcam_ok = test_camera_availability()
            camera_type = 'realsense' if realsense_ok else 'webcam'
        
        if test_basic_camera_feed(camera_type):
            success_count += 1
        total_tests += 1
    
    if args.test in ['all', 'detection']:
        print("\n📋 Test 3: Object Detection")
        if test_object_detection():
            success_count += 1
        total_tests += 1
    
    if args.test in ['all', 'sam']:
        print("\n📋 Test 4: SAM Integration")
        if test_sam_integration():
            success_count += 1
        total_tests += 1
    
    if args.test in ['all', '3d']:
        print("\n📋 Test 5: 3D Reconstruction")
        if test_3d_reconstruction():
            success_count += 1
        total_tests += 1
    
    print("\n" + "="*60)
    print(f"🏁 TEST SUMMARY: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests passed! Ready for Phase 4.")
    else:
        print("⚠️  Some tests failed. Check requirements and setup.")
        print("\n💡 Setup recommendations:")
        print("   1. Install Intel RealSense SDK for best results")
        print("   2. Download SAM checkpoint (sam_vit_b.pth)")
        print("   3. Ensure webcam is connected and working")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 