#!/usr/bin/env python3
"""
Phase 5: Complete Integration Demo
Real-time YOLO detection + Intelligent grasp planning + Robot simulation
"""

import cv2
import numpy as np
import time
from typing import List, Dict
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from src.live.advanced_object_detection import AdvancedSceneAnalyzer, DetectedObject
    from src.control.intelligent_grasp_planner import IntelligentGraspPlanner
    ADVANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Advanced modules not available: {e}")
    ADVANCED_MODULES_AVAILABLE = False

class Phase5IntegratedSystem:
    """Complete Real2Sim system with YOLO + Grasp Planning + Simulation"""
    
    def __init__(self):
        """Initialize the integrated system"""
        print("🚀 Phase 5: Integrated Real2Sim System")
        print("=" * 60)
        
        if not ADVANCED_MODULES_AVAILABLE:
            print("❌ Advanced modules not available. Please check imports.")
            return
        
        # Initialize components
        self.scene_analyzer = AdvancedSceneAnalyzer()
        self.grasp_planner = IntelligentGraspPlanner()
        
        # System state
        self.last_detection_time = 0
        self.detection_interval = 1.0  # Detect every 1 second
        self.total_objects_processed = 0
        self.successful_grasps = 0
        
        print("✅ All systems initialized")
    
    def run_live_demo(self):
        """Run the complete live demo"""
        if not ADVANCED_MODULES_AVAILABLE:
            return
        
        print("\n🎬 Starting live demo...")
        print("Controls: 'q' to quit, 's' to save frame, 'g' to plan grasps")
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No camera available. Running static demo instead...")
            self.run_static_demo()
            return
        
        frame_count = 0
        last_grasp_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Periodic object detection
            if current_time - self.last_detection_time > self.detection_interval:
                results = self._process_frame(frame, frame_count)
                self.last_detection_time = current_time
                
                # Display detection info
                if results['statistics']['total_objects'] > 0:
                    print(f"🎯 Frame {frame_count}: {results['statistics']['total_objects']} objects detected")
                    for class_name, count in results['statistics']['object_classes'].items():
                        print(f"   • {count}x {class_name}")
            else:
                # Just show frame without processing
                results = {'visualization': frame}
            
            # Add system info overlay
            self._add_system_overlay(results['visualization'], frame_count)
            
            # Display frame
            cv2.imshow('Phase 5: Integrated Real2Sim System', results['visualization'])
            
            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"phase5_frame_{frame_count}.jpg"
                cv2.imwrite(filename, results['visualization'])
                print(f"💾 Saved: {filename}")
            elif key == ord('g') and current_time - last_grasp_time > 2.0:
                # Plan grasps for current detections
                if 'objects' in results and results['objects']:
                    self._demonstrate_grasp_planning(results['objects'])
                    last_grasp_time = current_time
        
        cap.release()
        cv2.destroyAllWindows()
        self._print_final_stats()
    
    def run_static_demo(self):
        """Run demo with static/simulated data"""
        print("\n🎭 Running static demo with simulated objects...")
        
        # Create simulated scene
        simulated_objects = self._create_simulated_scene()
        
        # Analyze the scene
        print("\n📊 Scene Analysis:")
        self._analyze_simulated_scene(simulated_objects)
        
        # Demonstrate grasp planning
        print("\n🤖 Grasp Planning Demo:")
        self._demonstrate_grasp_planning(simulated_objects)
        
        # Show integration benefits
        print("\n🔄 Integration Benefits:")
        self._show_integration_benefits()
    
    def _process_frame(self, frame: np.ndarray, frame_count: int) -> Dict:
        """Process a single frame through the complete pipeline"""
        try:
            # Run scene analysis (YOLO detection + scene understanding)
            results = self.scene_analyzer.analyze_scene(frame)
            
            # Update stats
            self.total_objects_processed += results['statistics']['total_objects']
            
            return results
        except Exception as e:
            print(f"⚠️  Frame processing error: {e}")
            return {'visualization': frame, 'statistics': {'total_objects': 0}}
    
    def _add_system_overlay(self, frame: np.ndarray, frame_count: int):
        """Add system information overlay to frame"""
        h, w = frame.shape[:2]
        
        # System info
        info_lines = [
            f"Phase 5: Integrated Real2Sim",
            f"Frame: {frame_count}",
            f"Objects processed: {self.total_objects_processed}",
            f"Successful grasps: {self.successful_grasps}",
            f"Success rate: {(self.successful_grasps/max(1, self.total_objects_processed)*100):.1f}%"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y = 30 + i * 25
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add controls info
        controls = ["Press 'q' to quit", "'s' to save", "'g' for grasps"]
        for i, control in enumerate(controls):
            y = h - 80 + i * 25
            cv2.putText(frame, control, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _create_simulated_scene(self) -> List:
        """Create simulated detected objects for demo"""
        from types import SimpleNamespace
        
        return [
            SimpleNamespace(
                class_name='apple',
                bbox=(100, 100, 80, 80),
                confidence=0.92,
                center=(140, 140)
            ),
            SimpleNamespace(
                class_name='bottle',
                bbox=(250, 150, 60, 120),
                confidence=0.87,
                center=(280, 210)
            ),
            SimpleNamespace(
                class_name='cup',
                bbox=(400, 80, 70, 90),
                confidence=0.91,
                center=(435, 125)
            ),
            SimpleNamespace(
                class_name='book',
                bbox=(150, 300, 120, 160),
                confidence=0.84,
                center=(210, 380)
            ),
            SimpleNamespace(
                class_name='cell phone',
                bbox=(350, 250, 50, 100),
                confidence=0.89,
                center=(375, 300)
            )
        ]
    
    def _analyze_simulated_scene(self, objects: List):
        """Analyze the simulated scene"""
        print(f"📋 Scene contains {len(objects)} objects:")
        
        for i, obj in enumerate(objects, 1):
            print(f"   {i}. {obj.class_name} (confidence: {obj.confidence:.2f})")
        
        # Group by manipulability
        manipulable = ['bottle', 'cup', 'book', 'cell phone', 'apple']
        manipulable_objects = [obj for obj in objects if obj.class_name in manipulable]
        
        print(f"🤏 Manipulable objects: {len(manipulable_objects)}/{len(objects)}")
        print(f"🏆 Average confidence: {np.mean([obj.confidence for obj in objects]):.2f}")
    
    def _demonstrate_grasp_planning(self, objects: List):
        """Demonstrate grasp planning for detected objects"""
        print(f"\n🎯 Planning grasps for {len(objects)} objects...")
        
        # Plan grasps
        grasp_plans = self.grasp_planner.plan_grasp_sequence(objects)
        
        # Show planning results
        for i, plan in enumerate(grasp_plans, 1):
            print(f"\n🔧 Object {i}: {plan.class_name}")
            print(f"   📍 3D Position: [{plan.center_3d[0]:.3f}, {plan.center_3d[1]:.3f}, {plan.center_3d[2]:.3f}]")
            print(f"   📏 Estimated size: {plan.estimated_size}")
            print(f"   🎯 Available grasps: {len(plan.preferred_grasps)}")
            
            if plan.preferred_grasps:
                best_grasp = max(plan.preferred_grasps, key=lambda x: x.confidence)
                print(f"   🏆 Best grasp: {best_grasp.grasp_type.value} (confidence: {best_grasp.confidence:.2f})")
                
                # Simulate execution
                result = self.grasp_planner.execute_grasp_plan(plan)
                success_icon = "✅" if result['success'] else "❌"
                print(f"   {success_icon} Execution: {result['success']} (time: {result['execution_time']:.1f}s)")
                
                if result['success']:
                    self.successful_grasps += 1
    
    def _show_integration_benefits(self):
        """Show the benefits of the integrated system"""
        benefits = [
            "🎯 Real-time object classification (80+ classes)",
            "🤖 Object-specific grasp strategies",
            "📊 Scene complexity analysis",
            "⚡ Automated action suggestions",
            "🔄 Continuous learning from failures",
            "📈 Performance metrics tracking",
            "🎮 Interactive real-time control",
            "🔧 Modular system architecture"
        ]
        
        print("\n🚀 Phase 5 Integration Benefits:")
        for benefit in benefits:
            print(f"   {benefit}")
    
    def _print_final_stats(self):
        """Print final system statistics"""
        print("\n📊 Final Statistics:")
        print("=" * 40)
        print(f"Total objects processed: {self.total_objects_processed}")
        print(f"Successful grasps: {self.successful_grasps}")
        if self.total_objects_processed > 0:
            success_rate = (self.successful_grasps / self.total_objects_processed) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print("✅ Phase 5 demo completed successfully!")

def main():
    """Main demo function"""
    try:
        # Initialize integrated system
        system = Phase5IntegratedSystem()
        
        # Choose demo mode
        print("\nSelect demo mode:")
        print("1. Live camera feed (if available)")
        print("2. Static simulated demo")
        print("3. Both modes")
        
        choice = input("Enter choice (1/2/3) [default: 3]: ").strip() or "3"
        
        if choice == "1":
            system.run_live_demo()
        elif choice == "2":
            system.run_static_demo()
        else:
            print("\n🎬 Running both demos...")
            system.run_static_demo()
            print("\n" + "="*60)
            system.run_live_demo()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 