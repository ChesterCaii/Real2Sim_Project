#!/usr/bin/env python3
"""
Real Camera Integration - Phase 4
Real-time RGB-D camera feeds with live object detection and simulation updates
"""

import cv2
import numpy as np
import open3d as o3d
import time
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️  Intel RealSense not available. Using webcam + synthetic depth.")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  SAM not available. Using traditional computer vision methods.")

@dataclass
class CameraFrame:
    """Container for camera frame data"""
    rgb: np.ndarray
    depth: np.ndarray
    timestamp: float
    frame_id: int

@dataclass
class DetectedObject:
    """Container for detected object data"""
    id: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center_3d: np.ndarray
    point_cloud: o3d.geometry.PointCloud
    confidence: float

class RealSenseCamera:
    """Intel RealSense camera interface"""
    
    def __init__(self, width=640, height=480, fps=30):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("Intel RealSense not available")
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure RGB and depth streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        self.profile = None
        self.align = None
        self.running = False
        
    def start(self):
        """Start camera streaming"""
        try:
            self.profile = self.pipeline.start(self.config)
            
            # Create alignment object to align depth to color
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Get camera intrinsics
            color_profile = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            self.running = True
            print(f"✅ RealSense camera started: {self.intrinsics.width}x{self.intrinsics.height}")
            
        except Exception as e:
            print(f"❌ Failed to start RealSense: {e}")
            raise
    
    def get_frame(self) -> Optional[CameraFrame]:
        """Get synchronized RGB-D frame"""
        if not self.running:
            return None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
            
            # Convert to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return CameraFrame(
                rgb=rgb_image,
                depth=depth_image,
                timestamp=time.time(),
                frame_id=color_frame.get_frame_number()
            )
            
        except RuntimeError as e:
            print(f"⚠️  Frame timeout: {e}")
            return None
    
    def stop(self):
        """Stop camera streaming"""
        if self.running:
            self.pipeline.stop()
            self.running = False
            print("🔴 RealSense camera stopped")

class WebcamCamera:
    """Webcam with synthetic depth interface"""
    
    def __init__(self, camera_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.width = width
        self.height = height
        self.frame_count = 0
        self.running = False
        
        # Synthetic camera intrinsics
        self.fx = self.fy = width  # Simple perspective
        self.cx, self.cy = width // 2, height // 2
    
    def start(self):
        """Start webcam"""
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        self.running = True
        print(f"✅ Webcam started: {self.width}x{self.height}")
    
    def create_synthetic_depth(self, rgb_image):
        """Create synthetic depth map from RGB image"""
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for smoothness
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Create depth map: darker areas = closer (lower depth values)
        # Scale to realistic depth range (0.3m to 3.0m)
        depth_map = 3000 - (blurred.astype(np.float32) / 255.0) * 2700 + 300
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, depth_map.shape)
        depth_map = np.clip(depth_map + noise, 300, 3000).astype(np.uint16)
        
        return depth_map
    
    def get_frame(self) -> Optional[CameraFrame]:
        """Get RGB frame with synthetic depth"""
        if not self.running:
            return None
        
        ret, rgb_image = self.cap.read()
        if not ret:
            return None
        
        # Create synthetic depth
        depth_image = self.create_synthetic_depth(rgb_image)
        
        self.frame_count += 1
        
        return CameraFrame(
            rgb=rgb_image,
            depth=depth_image,
            timestamp=time.time(),
            frame_id=self.frame_count
        )
    
    def stop(self):
        """Stop webcam"""
        if self.running:
            self.cap.release()
            self.running = False
            print("🔴 Webcam stopped")

class LiveObjectDetector:
    """Real-time object detection and segmentation"""
    
    def __init__(self, use_sam=True, min_area=1000):
        self.use_sam = use_sam and SAM_AVAILABLE
        self.min_area = min_area
        
        if self.use_sam:
            self._init_sam()
        else:
            self._init_opencv_detector()
    
    def _init_sam(self):
        """Initialize SAM for object detection"""
        try:
            # Use smaller model for real-time performance
            model_type = "vit_b"
            checkpoint_path = "sam_vit_b.pth"
            
            if os.path.exists(checkpoint_path):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                self.sam.to(device=device)
                self.mask_generator = SamAutomaticMaskGenerator(
                    self.sam,
                    points_per_side=16,  # Reduced for speed
                    pred_iou_thresh=0.8,
                    stability_score_thresh=0.85,
                    crop_n_layers=0,  # Disable cropping for speed
                    crop_n_points_downscale_factor=1,
                    min_mask_region_area=self.min_area
                )
                print("✅ SAM initialized for real-time detection")
            else:
                print("⚠️  SAM checkpoint not found, falling back to OpenCV")
                self.use_sam = False
                self._init_opencv_detector()
                
        except Exception as e:
            print(f"⚠️  SAM initialization failed: {e}")
            self.use_sam = False
            self._init_opencv_detector()
    
    def _init_opencv_detector(self):
        """Initialize OpenCV-based object detection"""
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=100
        )
        print("✅ OpenCV detector initialized")
    
    def detect_objects(self, rgb_image: np.ndarray) -> List[np.ndarray]:
        """Detect objects and return masks"""
        if self.use_sam:
            return self._detect_with_sam(rgb_image)
        else:
            return self._detect_with_opencv(rgb_image)
    
    def _detect_with_sam(self, rgb_image: np.ndarray) -> List[np.ndarray]:
        """Detect objects using SAM"""
        try:
            # Convert BGR to RGB for SAM
            rgb_for_sam = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            masks_data = self.mask_generator.generate(rgb_for_sam)
            
            # Extract mask arrays and filter by area
            masks = []
            for mask_data in masks_data:
                mask = mask_data['segmentation']
                area = mask_data['area']
                
                if area >= self.min_area:
                    masks.append(mask)
            
            return masks
            
        except Exception as e:
            print(f"⚠️  SAM detection failed: {e}")
            return []
    
    def _detect_with_opencv(self, rgb_image: np.ndarray) -> List[np.ndarray]:
        """Detect objects using OpenCV background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(rgb_image)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to masks
        masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                mask = np.zeros(rgb_image.shape[:2], dtype=bool)
                cv2.fillPoly(mask, [contour], True)
                masks.append(mask)
        
        return masks

class LiveReconstructionPipeline:
    """Real-time reconstruction pipeline"""
    
    def __init__(self, camera_type="auto"):
        # Initialize camera
        if camera_type == "auto":
            if REALSENSE_AVAILABLE:
                self.camera = RealSenseCamera()
                print("🎥 Using Intel RealSense camera")
            else:
                self.camera = WebcamCamera()
                print("🎥 Using webcam with synthetic depth")
        elif camera_type == "realsense":
            self.camera = RealSenseCamera()
        elif camera_type == "webcam":
            self.camera = WebcamCamera()
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        
        # Initialize detector
        self.detector = LiveObjectDetector()
        
        # Processing queues
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Control flags
        self.running = False
        self.processing_thread = None
        self.display_thread = None
        
        # Statistics
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def start(self):
        """Start the live reconstruction pipeline"""
        print("🚀 Starting live reconstruction pipeline...")
        
        # Start camera
        self.camera.start()
        
        # Start processing threads
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        
        self.processing_thread.start()
        self.display_thread.start()
        
        print("✅ Live pipeline started! Press 'q' to quit.")
    
    def _processing_loop(self):
        """Main processing loop for object detection and reconstruction"""
        while self.running:
            # Get frame from camera
            frame = self.camera.get_frame()
            if frame is None:
                continue
            
            # Add to processing queue (drop old frames if queue is full)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Remove old frame
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
            
            # Process frame
            try:
                frame_to_process = self.frame_queue.get_nowait()
                detected_objects = self._process_frame(frame_to_process)
                
                # Add results to display queue
                try:
                    self.result_queue.put_nowait({
                        'frame': frame_to_process,
                        'objects': detected_objects
                    })
                except queue.Full:
                    # Remove old result
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait({
                            'frame': frame_to_process,
                            'objects': detected_objects
                        })
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                time.sleep(0.001)  # Small delay to prevent busy waiting
    
    def _process_frame(self, frame: CameraFrame) -> List[DetectedObject]:
        """Process a single frame to detect and reconstruct objects"""
        # Detect objects
        masks = self.detector.detect_objects(frame.rgb)
        
        detected_objects = []
        for i, mask in enumerate(masks):
            # Get bounding box
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue
                
            bbox = (
                int(np.min(x_indices)),
                int(np.min(y_indices)),
                int(np.max(x_indices) - np.min(x_indices)),
                int(np.max(y_indices) - np.min(y_indices))
            )
            
            # Create point cloud for this object
            point_cloud = self._create_point_cloud(frame, mask)
            
            if point_cloud is not None and len(point_cloud.points) > 100:
                # Calculate 3D center
                center_3d = np.mean(np.asarray(point_cloud.points), axis=0)
                
                detected_objects.append(DetectedObject(
                    id=i,
                    mask=mask,
                    bbox=bbox,
                    center_3d=center_3d,
                    point_cloud=point_cloud,
                    confidence=1.0  # Simple confidence
                ))
        
        return detected_objects
    
    def _create_point_cloud(self, frame: CameraFrame, mask: np.ndarray) -> Optional[o3d.geometry.PointCloud]:
        """Create point cloud from RGB-D data and object mask"""
        # Get camera intrinsics
        if hasattr(self.camera, 'intrinsics'):
            # RealSense intrinsics
            fx = self.camera.intrinsics.fx
            fy = self.camera.intrinsics.fy
            cx = self.camera.intrinsics.ppx
            cy = self.camera.intrinsics.ppy
        else:
            # Webcam synthetic intrinsics
            fx = fy = self.camera.fx
            cx, cy = self.camera.cx, self.camera.cy
        
        # Create point cloud
        points = []
        colors = []
        
        height, width = frame.depth.shape
        
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    depth_value = frame.depth[y, x]
                    if depth_value > 0:
                        # Convert to meters (RealSense depth is in mm)
                        z = depth_value / 1000.0 if hasattr(self.camera, 'intrinsics') else depth_value / 1000.0
                        
                        # Skip invalid depths
                        if z < 0.1 or z > 5.0:
                            continue
                        
                        # Convert pixel to 3D point
                        px = (x - cx) * z / fx
                        py = (y - cy) * z / fy
                        
                        points.append([px, py, z])
                        
                        # Get color (BGR to RGB)
                        color = frame.rgb[y, x]
                        colors.append([color[2]/255.0, color[1]/255.0, color[0]/255.0])
        
        if len(points) == 0:
            return None
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        
        return pcd
    
    def _display_loop(self):
        """Display loop for visualization"""
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                self._display_results(result['frame'], result['objects'])
                
                # Update FPS
                self.fps_counter += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    
            except queue.Empty:
                continue
    
    def _display_results(self, frame: CameraFrame, objects: List[DetectedObject]):
        """Display detection and reconstruction results"""
        display_image = frame.rgb.copy()
        
        # Draw detected objects
        for obj in objects:
            # Draw bounding box
            x, y, w, h = obj.bbox
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw mask overlay
            mask_colored = np.zeros_like(display_image)
            mask_colored[obj.mask] = [0, 255, 255]  # Yellow overlay
            display_image = cv2.addWeighted(display_image, 0.7, mask_colored, 0.3, 0)
            
            # Draw object info
            info_text = f"Obj {obj.id}: {len(obj.point_cloud.points)} pts"
            cv2.putText(display_image, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw 3D center (projected back to 2D)
            center_2d = self._project_3d_to_2d(obj.center_3d)
            if center_2d is not None:
                cv2.circle(display_image, center_2d, 5, (255, 0, 0), -1)
        
        # Draw FPS and stats
        fps_text = f"FPS: {self.current_fps} | Objects: {len(objects)}"
        cv2.putText(display_image, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show depth image (normalized for display)
        depth_display = cv2.applyColorMap(
            cv2.convertScaleAbs(frame.depth, alpha=0.08), cv2.COLORMAP_JET
        )
        
        # Combine RGB and depth for display
        combined = np.hstack((display_image, depth_display))
        
        cv2.imshow('Live Object Detection & Reconstruction', combined)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop()
    
    def _project_3d_to_2d(self, point_3d: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project 3D point back to 2D image coordinates"""
        if hasattr(self.camera, 'intrinsics'):
            fx = self.camera.intrinsics.fx
            fy = self.camera.intrinsics.fy
            cx = self.camera.intrinsics.ppx
            cy = self.camera.intrinsics.ppy
        else:
            fx = fy = self.camera.fx
            cx, cy = self.camera.cx, self.camera.cy
        
        x, y, z = point_3d
        if z <= 0:
            return None
        
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)
        
        return (u, v)
    
    def stop(self):
        """Stop the live reconstruction pipeline"""
        print("🔴 Stopping live pipeline...")
        self.running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        
        self.camera.stop()
        cv2.destroyAllWindows()
        print("✅ Live pipeline stopped")

def main():
    """Main function for live reconstruction demo"""
    print("============================================================")
    print("🎥 LIVE CAMERA INTEGRATION - PHASE 4")
    print("============================================================")
    
    try:
        # Create and start pipeline
        pipeline = LiveReconstructionPipeline(camera_type="auto")
        pipeline.start()
        
        # Keep main thread alive
        while pipeline.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'pipeline' in locals():
            pipeline.stop()

if __name__ == "__main__":
    main() 