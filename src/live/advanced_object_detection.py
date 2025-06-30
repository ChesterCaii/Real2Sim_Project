#!/usr/bin/env python3
"""
Advanced Object Detection - Phase 5
Real-time YOLO-based object detection with classification
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available. Install with: pip install ultralytics")

@dataclass
class DetectedObject:
    """Enhanced object detection result with classification"""
    id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    mask: Optional[np.ndarray] = None
    
class YOLOObjectDetector:
    """YOLO-based object detection with real-time classification"""
    
    def __init__(self, model_size='n', confidence_threshold=0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if YOLO_AVAILABLE:
            try:
                print(f"🔄 Loading YOLOv8{model_size} model...")
                self.model = YOLO(f'yolov8{model_size}.pt')
                print("✅ YOLO model loaded successfully")
            except Exception as e:
                print(f"❌ YOLO model loading failed: {e}")
                self.model = None
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Colors for visualization (one per class)
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.randint(0, 255, (len(self.class_names), 3))
    
    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect objects in image using YOLO"""
        if self.model is None:
            return self._fallback_detection(image)
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            detected_objects = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Get detection data
                        box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        # Convert to our format
                        x1, y1, x2, y2 = box
                        bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                        center = (int((x1+x2)/2), int((y1+y2)/2))
                        class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                        
                        detected_objects.append(DetectedObject(
                            id=i,
                            class_name=class_name,
                            confidence=conf,
                            bbox=bbox,
                            center=center
                        ))
            
            return detected_objects
            
        except Exception as e:
            print(f"⚠️  YOLO detection failed: {e}")
            return self._fallback_detection(image)
    
    def _fallback_detection(self, image: np.ndarray) -> List[DetectedObject]:
        """Fallback to simple contour detection when YOLO fails"""
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                detected_objects.append(DetectedObject(
                    id=i,
                    class_name="unknown_object",
                    confidence=0.8,  # Default confidence
                    bbox=(x, y, w, h),
                    center=center
                ))
        
        return detected_objects
    
    def draw_detections(self, image: np.ndarray, objects: List[DetectedObject]) -> np.ndarray:
        """Draw detection results on image"""
        result_image = image.copy()
        
        for obj in objects:
            x, y, w, h = obj.bbox
            color = self.colors[hash(obj.class_name) % len(self.colors)].tolist()
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{obj.class_name}: {obj.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(result_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(result_image, obj.center, 5, color, -1)
        
        return result_image

class AdvancedSceneAnalyzer:
    """Advanced scene analysis with object relationships"""
    
    def __init__(self):
        self.detector = YOLOObjectDetector()
        self.object_history = {}  # Track objects over time
        self.next_object_id = 0
    
    def analyze_scene(self, image: np.ndarray) -> Dict:
        """Comprehensive scene analysis"""
        # Detect objects
        objects = self.detector.detect_objects(image)
        
        # Track objects over time (simple tracking by proximity)
        tracked_objects = self._track_objects(objects)
        
        # Analyze scene composition
        scene_stats = {
            'total_objects': len(tracked_objects),
            'object_classes': {},
            'manipulable_objects': [],
            'scene_complexity': self._calculate_complexity(tracked_objects),
            'suggested_actions': self._suggest_actions(tracked_objects)
        }
        
        # Count objects by class
        for obj in tracked_objects:
            if obj.class_name in scene_stats['object_classes']:
                scene_stats['object_classes'][obj.class_name] += 1
            else:
                scene_stats['object_classes'][obj.class_name] = 1
        
        # Identify manipulable objects (common graspable items)
        manipulable_classes = {
            'bottle', 'cup', 'bowl', 'book', 'cell phone', 'remote', 'scissors',
            'teddy bear', 'apple', 'orange', 'banana', 'sports ball'
        }
        
        for obj in tracked_objects:
            if obj.class_name in manipulable_classes:
                scene_stats['manipulable_objects'].append({
                    'class': obj.class_name,
                    'confidence': obj.confidence,
                    'center': obj.center,
                    'bbox': obj.bbox
                })
        
        return {
            'objects': tracked_objects,
            'statistics': scene_stats,
            'visualization': self.detector.draw_detections(image, tracked_objects)
        }
    
    def _track_objects(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """Simple object tracking by proximity"""
        # For now, just return objects (advanced tracking would use Kalman filters)
        return objects
    
    def _calculate_complexity(self, objects: List[DetectedObject]) -> float:
        """Calculate scene complexity score (0-1)"""
        if not objects:
            return 0.0
        
        # Factors: number of objects, diversity of classes, confidence levels
        num_objects = len(objects)
        unique_classes = len(set(obj.class_name for obj in objects))
        avg_confidence = np.mean([obj.confidence for obj in objects])
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (num_objects * 0.1) + (unique_classes * 0.15) + (avg_confidence * 0.2))
        return complexity
    
    def _suggest_actions(self, objects: List[DetectedObject]) -> List[str]:
        """Suggest possible robot actions based on detected objects"""
        suggestions = []
        
        if not objects:
            suggestions.append("No objects detected - explore environment")
            return suggestions
        
        # Check for graspable objects
        graspable = [obj for obj in objects if obj.class_name in {
            'bottle', 'cup', 'book', 'cell phone', 'apple', 'orange'
        }]
        
        if graspable:
            suggestions.append(f"Pick up {graspable[0].class_name}")
            if len(graspable) > 1:
                suggestions.append("Sort objects by type")
        
        # Check for containers
        containers = [obj for obj in objects if obj.class_name in {'bowl', 'cup'}]
        if containers and graspable:
            suggestions.append("Place objects in container")
        
        # General suggestions
        if len(objects) > 3:
            suggestions.append("Organize workspace")
        
        return suggestions[:3]  # Limit to top 3 suggestions

def main():
    """Demo of advanced object detection"""
    print("🚀 Advanced Object Detection Demo")
    print("=" * 50)
    
    analyzer = AdvancedSceneAnalyzer()
    
    # Test with webcam if available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available")
        return
    
    print("📹 Camera feed started. Press 'q' to quit, 's' to save detection")
    save_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze scene
        results = analyzer.analyze_scene(frame)
        
        # Display results
        cv2.imshow('Advanced Object Detection', results['visualization'])
        
        # Print stats every 30 frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            # Save detection results
            save_path = f"detection_result_{save_counter}.jpg"
            cv2.imwrite(save_path, results['visualization'])
            print(f"💾 Saved: {save_path}")
            save_counter += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo completed")

if __name__ == "__main__":
    main() 